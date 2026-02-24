from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np

# ============================================================================
# 프로젝트 의존성 임포트 (조건부)
# ============================================================================

try:
    from core.config import cfg  # type: ignore
except ImportError:
    cfg = None

from core.common_torch import resolve_node_idx  # [P4-DEDUP]
NODE_IDX = resolve_node_idx()


# ============================================================================
# 로깅 설정
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# 예외 클래스
# ============================================================================


class FightDetectionError(Exception):
    """교전 감지 관련 기본 예외"""


class InsufficientDataError(FightDetectionError):
    """데이터 부족 예외"""


class InvalidTeamMappingError(FightDetectionError):
    """팀 매핑 오류 예외"""


class ConfigurationError(FightDetectionError):
    """설정 오류 예외"""


# ============================================================================
# 상수 정의
# ============================================================================


class MapConstants:
    """리그 오브 레전드 맵 상수

    [FIX #10] config.py CFG 클래스의 BARON_PIT_XY / DRAGON_PIT_XY와 좌표 통일.
    """

    MAP_WIDTH: float = 16000.0
    MAP_HEIGHT: float = 16000.0
    MAP_MARGIN: float = 4000.0
    MAX_COORDINATE: float = MAP_WIDTH + MAP_MARGIN

    BLUE_BASE: Tuple[float, float] = (500.0, 500.0)
    RED_BASE: Tuple[float, float] = (14500.0, 14500.0)

    # default (CFG 없을 때 fallback)
    BARON_PIT: Tuple[float, float] = (5000.0, 10400.0)
    DRAGON_PIT: Tuple[float, float] = (9850.0, 4400.0)

    NORM_THRESHOLD: float = 2.0
    NORM_DIVISOR: float = 16000.0

    BASE_RADIUS: float = 3000.0
    OBJECTIVE_RADIUS: float = 1500.0
    TOWER_RADIUS: float = 1000.0


# [FIX #10] cfg 객체가 존재하면 cfg의 좌표로 MapConstants를 동적 갱신
if cfg is not None:
    _baron_xy = getattr(cfg, "BARON_PIT_XY", None)
    if isinstance(_baron_xy, (tuple, list)) and len(_baron_xy) == 2:
        MapConstants.BARON_PIT = (float(_baron_xy[0]), float(_baron_xy[1]))
    _dragon_xy = getattr(cfg, "DRAGON_PIT_XY", None)
    if isinstance(_dragon_xy, (tuple, list)) and len(_dragon_xy) == 2:
        MapConstants.DRAGON_PIT = (float(_dragon_xy[0]), float(_dragon_xy[1]))


class FightType(str, Enum):
    TEAMFIGHT = "teamfight"
    SKIRMISH = "skirmish"
    PICK = "pick"
    TOWER_DIVE = "tower_dive"
    OBJECTIVE_BARON = "objective_baron"
    OBJECTIVE_DRAGON = "objective_dragon"
    OBJECTIVE_RIFTHERALD = "objective_riftherald"
    OBJECTIVE_OTHER = "objective_other"
    BASE_FIGHT = "base_fight"
    UNKNOWN = "unknown"


# ============================================================================
# 타입 정의
# ============================================================================


class FightSegment(TypedDict):
    engage_ts: int
    centroid_x: float
    centroid_y: float
    first_kill_ts: int


class PlayerEngagement(TypedDict):
    player_idx: int
    team: str
    engagement_ratio: float
    avg_dist_to_enemy: float
    frames_engaged: int
    total_frames: int


class FightOutcome(TypedDict):
    winner: str
    blue_kills: int
    red_kills: int
    blue_deaths: int
    red_deaths: int
    kill_diff: int
    total_kills: int
    assists: Dict[str, int]
    blue_unique_deaths: int
    red_unique_deaths: int
    blue_survivors: int
    red_survivors: int
    blue_alive_end: int
    red_alive_end: int
    gold_blue_delta: float
    gold_red_delta: float
    gold_diff: float
    tower_blue: int
    tower_red: int
    tower_diff: int
    plate_blue: int
    plate_red: int
    plate_diff: int
    inhib_blue: int
    inhib_red: int
    inhib_diff: int
    objective_blue: int
    objective_red: int
    objective_diff: int
    objective_by_type: Dict[str, Dict[str, int]]


class FightVisualization(TypedDict):
    trajectory: List[Dict[str, Any]]
    heatmap_points: List[Tuple[float, float]]
    engagement_timeline: List[Dict[str, Any]]
    kill_markers: List[Dict[str, Any]]


# ============================================================================
# 설정 클래스
# ============================================================================


@dataclass
class FightDetectorConfig:
    """교전 감지기 설정

    [FIX #2] 기본값을 config.py의 CFG 클래스와 완전 일치시킴.
    """

    standoff_radius: float = 1800.0
    standoff_min_pairs: int = 3
    engage_min_dist_drop: float = 250.0
    engage_min_pair_gain: int = 2
    fight_min_gap_ms: int = 120000
    fight_context_min: int = 1
    detect_step_ms: int = 10000
    frame_ms: int = 60000

    # 병합 관련
    continuous_fight_merge: bool = True
    continuous_fight_max_gap_ms: int = 30000
    continuous_fight_merge_radius: float = 2000.0
    # [P2-2 FIX] 기본값을 config.py의 MAX_MERGED_FIGHT_DURATION_MS = 120000과 일치시킴.
    # 이전 값 300000ms(5분)는 config.py 값 120000ms(2분)과 불일치했음.
    # from_cfg() 없이 직접 dataclass를 생성하면 재현성이 흔들릴 수 있었음.
    max_merged_fight_duration_ms: int = 120000

    # kill-anchor / backtrack
    # Project rule: start-point must be engage-based, not kill-based.
    use_kill_anchor: bool = False
    kill_anchor_pre_sec: int = 15
    kill_anchor_cooldown_sec: int = 30
    # Fight validity check: require at least one kill in [engage_ts, engage_ts + horizon).
    verify_kill_in_horizon: bool = True
    fight_validation_rule: str = "kill_or_signal"
    min_damage_norm_in_horizon: float = 0.02
    min_summoner_spells_in_horizon: int = 1
    use_backtrack: bool = True
    # [P3-BT] 60s â†’ 30s: reduces noise, Phase 1 already covers long-range signals
    backtrack_max_ms: int = 30000
    backtrack_min_ms: int = 10000
    backtrack_min_pairs: int = 3

    # 구조적 가드
    # [P3-ALIVE] 2 â†’ 3: exclude 2v2 skirmishes (P(y|2v2) â‰  P(y|5v5))
    require_alive_per_team: int = 3
    require_engaged_per_team: int = 2
    require_lcc_total: int = 4
    require_lcc_per_team: int = 2
    cluster_max_diameter: float = 4000.0
    require_ward_actor_in_fight_radius: bool = True
    ward_actor_radius: float = 1800.0

    # 보간/스케일
    interp_method: str = "linear"
    coord_norm_div: float = 16000.0

    # 성능/안전
    chunk_size: int = 500
    strict_mode: bool = False

    # [FIX #6] 시간 스케일링 하한(거리 단위)
    engage_drop_floor: float = 30.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """[FIX #3] backtrack_min/max 교차 검증 추가, 에러 메시지 상세화."""
        errors: List[str] = []
        if self.standoff_radius <= 0:
            errors.append(f"standoff_radius must be positive, got {self.standoff_radius}")
        if not (1 <= self.standoff_min_pairs <= 25):
            errors.append(f"standoff_min_pairs must be 1-25, got {self.standoff_min_pairs}")
        if self.fight_min_gap_ms < 0:
            errors.append("fight_min_gap_ms must be non-negative")
        if self.continuous_fight_merge and self.continuous_fight_max_gap_ms >= self.fight_min_gap_ms:
            errors.append(
                f"continuous_fight_max_gap_ms ({self.continuous_fight_max_gap_ms}) "
                f"must be < fight_min_gap_ms ({self.fight_min_gap_ms})"
            )
        if self.backtrack_min_ms > self.backtrack_max_ms:
            errors.append(
                f"backtrack_min_ms ({self.backtrack_min_ms}) must be <= backtrack_max_ms ({self.backtrack_max_ms})"
            )
        if self.ward_actor_radius < 0:
            errors.append(f"ward_actor_radius must be >= 0, got {self.ward_actor_radius}")
        valid_rules = {"kill_only", "signal_only", "kill_or_signal", "kill_and_signal"}
        if str(self.fight_validation_rule).lower() not in valid_rules:
            errors.append(
                f"fight_validation_rule must be one of {sorted(valid_rules)}, got {self.fight_validation_rule!r}"
            )
        # Project rule: fight start-point must not be kill-anchored.
        # Keep other config knobs intact by forcing this one off instead of failing whole config load.
        if self.use_kill_anchor:
            logger.warning(
                "USE_KILL_ANCHOR=True is not allowed; forcing False. "
                "fight start-point remains engage-detected."
            )
            self.use_kill_anchor = False
        if errors:
            raise ConfigurationError("Config validation failed:\n" + "\n".join(errors))

    @classmethod
    def from_cfg(cls, cfg_obj: Any) -> "FightDetectorConfig":
        """cfg 객체로부터 설정 생성 (fallback은 CFG 기본값에 정렬).

        Project rule:
          - fight start-point: engage detection only
          - fight validity: require at least one kill in horizon
        """
        if cfg_obj is None:
            return cls()

        return cls(
            standoff_radius=float(getattr(cfg_obj, "STANDOFF_RADIUS", 1800.0)),
            standoff_min_pairs=int(getattr(cfg_obj, "STANDOFF_MIN_PAIRS", 3)),
            engage_min_dist_drop=float(getattr(cfg_obj, "ENGAGE_MIN_DIST_DROP", 250.0)),
            engage_min_pair_gain=int(getattr(cfg_obj, "ENGAGE_MIN_PAIR_GAIN", 2)),
            fight_min_gap_ms=int(
                getattr(cfg_obj, "FIGHT_MIN_GAP_MS", int(getattr(cfg_obj, "FIGHT_MIN_GAP_MIN", 2)) * 60000)
            ),
            fight_context_min=int(getattr(cfg_obj, "FIGHT_CONTEXT_MIN", 1)),
            detect_step_ms=int(getattr(cfg_obj, "DETECT_STEP_MS", int(getattr(cfg_obj, "BIN_MS", 5000)))),
            frame_ms=int(getattr(cfg_obj, "FRAME_MS", 60000)),
            continuous_fight_merge=bool(getattr(cfg_obj, "CONTINUOUS_FIGHT_MERGE", True)),
            continuous_fight_max_gap_ms=int(getattr(cfg_obj, "CONTINUOUS_FIGHT_MAX_GAP_MS", 30000)),
            continuous_fight_merge_radius=float(getattr(cfg_obj, "CONTINUOUS_FIGHT_MERGE_RADIUS", 2000.0)),
            max_merged_fight_duration_ms=int(getattr(cfg_obj, "MAX_MERGED_FIGHT_DURATION_MS", 120000)),
            use_kill_anchor=bool(getattr(cfg_obj, "USE_KILL_ANCHOR", False)),
            kill_anchor_pre_sec=int(getattr(cfg_obj, "KILL_ANCHOR_PRE_SEC", 15)),
            kill_anchor_cooldown_sec=int(getattr(cfg_obj, "KILL_ANCHOR_COOLDOWN_SEC", 30)),
            verify_kill_in_horizon=bool(getattr(cfg_obj, "VERIFY_KILL_IN_HORIZON", True)),
            fight_validation_rule=str(getattr(cfg_obj, "FIGHT_VALIDATION_RULE", "kill_or_signal")).lower(),
            min_damage_norm_in_horizon=float(getattr(cfg_obj, "MIN_DAMAGE_NORM_IN_HORIZON", 0.02)),
            min_summoner_spells_in_horizon=int(getattr(cfg_obj, "MIN_SUMMONER_SPELLS_IN_HORIZON", 1)),
            use_backtrack=bool(getattr(cfg_obj, "USE_BACKTRACK", True)),
            backtrack_max_ms=int(getattr(cfg_obj, "BACKTRACK_MAX_MS", 30000)),
            backtrack_min_ms=int(getattr(cfg_obj, "BACKTRACK_MIN_MS", 10000)),
            backtrack_min_pairs=int(getattr(cfg_obj, "BACKTRACK_MIN_PAIRS", 3)),
            require_alive_per_team=int(getattr(cfg_obj, "REQUIRE_ALIVE_PER_TEAM", 3) or 0),
            require_engaged_per_team=int(getattr(cfg_obj, "REQUIRE_ENGAGED_PER_TEAM", 2) or 0),
            require_lcc_total=int(getattr(cfg_obj, "REQUIRE_LCC_TOTAL", 4) or 0),
            require_lcc_per_team=int(getattr(cfg_obj, "REQUIRE_LCC_PER_TEAM", 2) or 0),
            cluster_max_diameter=float(getattr(cfg_obj, "CLUSTER_MAX_DIAMETER", 4000.0) or 0.0),
            require_ward_actor_in_fight_radius=bool(getattr(cfg_obj, "REQUIRE_WARD_ACTOR_IN_FIGHT_RADIUS", True)),
            ward_actor_radius=float(getattr(cfg_obj, "WARD_ACTOR_RADIUS", 1800.0) or 0.0),
            interp_method=str(getattr(cfg_obj, "INTERP_METHOD", "linear")).lower(),
            coord_norm_div=float(getattr(cfg_obj, "COORD_NORM_DIV", 16000.0)),
            chunk_size=int(getattr(cfg_obj, "CHUNK_SIZE", 500)),
            strict_mode=bool(getattr(cfg_obj, "STRICT_MODE", False)),
            engage_drop_floor=float(getattr(cfg_obj, "ENGAGE_DROP_FLOOR", 30.0)),
        )


# ============================================================================
# 유틸리티 함수
# ============================================================================


from core.common import safe_float


def safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(math.sqrt(dx * dx + dy * dy))


def _get_horizon_ms() -> int:
    """cfg에서 horizon_ms 가져오기"""
    if cfg is None:
        return 60000
    if hasattr(cfg, "FIGHT_HORIZON_SEC"):
        return int(getattr(cfg, "FIGHT_HORIZON_SEC", 60)) * 1000
    return int(getattr(cfg, "FIGHT_HORIZON_MIN", 1)) * 60000


# ============================================================================
# [P0 FIX] Post-merge spacing + non-overlap enforcement
#   - continuous_fight_merge=True 여도 fight_min_gap_ms를 강제로 적용
#   - label window는 [engage_ts, horizon_end_ts) 우선 (없으면 기본 horizon)
# ============================================================================


def _count_events_in_window(ts_sorted: np.ndarray, t0: int, t1_inclusive: int) -> int:
    """ts_sorted 오름차순일 때 [t0, t1] 포함 구간 개수."""
    if ts_sorted.size == 0:
        return 0
    l = int(np.searchsorted(ts_sorted, t0, side="left"))
    r = int(np.searchsorted(ts_sorted, t1_inclusive, side="right"))
    return max(0, r - l)


def _label_end_ts(fight: dict, horizon_ms: int) -> int:
    """Fight label-end ts (exclusive).

    Priority:
      1) fight["horizon_end_ts"] if valid (> engage_ts)
      2) engage_ts + horizon_ms fallback
    """
    t0 = int(fight.get("engage_ts", -1))
    if t0 < 0:
        return int(horizon_ms)
    fallback = int(t0 + int(horizon_ms))
    try:
        hend = int(fight.get("horizon_end_ts", fallback))
    except Exception:
        hend = fallback
    if hend <= t0:
        return fallback
    return hend


def _fight_priority_score(
    f: dict,
    *,
    kill_ts: np.ndarray,
    horizon_ms: int,
) -> float:
    """
    [P1-1 FIX] Observe-time-only priority score for conflict resolution.

    conflict(겹침/갭위반) 상황에서 어떤 fight를 남길지 결정하는 휴리스틱 점수.

    수학적 배경 (선택 편향 제거):
    ──────────────────────────────
    이전 구현:
        s(f) = 1000·|{k ∈ kills : k ∈ [t₀, t₀+H]}| + 10·prox + 5·seg + ...

    문제:
        가중치 1000·k가 다른 항을 완전히 지배하므로, 어떤 샘플이
        데이터셋에 남느냐가 미래 이벤트(라벨 구간의 kill)에 의해 결정됨.

        P(sample f retained) ∝ g(kills_in_horizon(f))

        이는 라벨-연관 변수로의 조건부 샘플링(selection bias):
            D_train ~ P(x, y | selected=1)  where  selected = h(y)
            → AUC 과대추정, "쉬운 전투"에 대한 과적합

    수정 후:
        s(f) = w₁·prox_pairs + w₂·n_segments + w₃·anchor + w₄·backtracked

        모든 항이 engage 시점까지 관측 가능한 특징만 사용.
        kill 정보를 배제하여 P(selected | y)와 y의 독립성 보장:

            selected ⊥ y | X_observed

    Note:
        kill_ts와 horizon_ms 파라미터는 호출 인터페이스 호환을 위해 유지하되,
        내부적으로 사용하지 않음 (향후 제거 시 deprecation warning 추가 가능).
    """
    try:
        t0 = int(f.get("engage_ts", -1))
    except Exception:
        t0 = -1
    if t0 < 0:
        return -1e18

    # [P1-1] Observe-time-only features (engage 시점까지 관측 가능)
    prox = int(f.get("det_prox_pairs", 0) or 0)
    seg = int(f.get("n_segments", 1) or 1)
    anchor = int(f.get("det_anchor", 0) or 0)
    back = int(f.get("det_backtracked", 0) or 0)

    # 가중치 설계:
    #   prox_pairs: 교전 밀도의 직접적 지표 → 최고 가중치
    #   anchor:     kill-anchor 기반 후보는 더 신뢰성 있는 교전 시점
    #   n_segments: 병합된 세그먼트 수 → 지속적 교전의 증거
    #   backtracked: backtrack으로 보정된 시점
    return float(100 * prox + 50 * anchor + 20 * seg + 10 * back)


def enforce_postmerge_spacing_and_nonoverlap(
    fights: List[dict],
    *,
    horizon_ms: int,
    fight_min_gap_ms: int,
    kill_ts: np.ndarray,
    location_radius: float = 0.0,
    diag: Optional[dict] = None,
) -> List[dict]:
    """
    [P0 FIX] continuous merge 결과에 대해 최종적으로:
      1) engage_ts 간 최소 간격 fight_min_gap_ms 강제
      2) 라벨 윈도우 [engage_ts, label_end_ts)가 겹치지 않도록 강제

    전략:
      - 시간순으로 스캔하며, prev와 overlap 또는 too_close면 더 "좋은" fight 1개만 남김.
      - [P1-1 FIX] '좋음'은 engage 시점까지 관측 가능한 특징(prox_pairs, anchor 등)만 사용.
        미래 kill 정보를 배제하여 라벨-연관 선택 편향을 방지.
    """
    if not fights:
        if diag is not None:
            diag.setdefault("postmerge_conflicts", 0)
            diag.setdefault("postmerge_removed", 0)
            diag.setdefault("postmerge_replaced", 0)
        return fights

    fs = sorted(fights, key=lambda x: int(x.get("engage_ts", -1)))
    kept: List[dict] = []

    conflicts = 0
    removed = 0
    replaced = 0

    for f in fs:
        t0 = int(f.get("engage_ts", -1))
        if t0 < 0:
            continue
        if not kept:
            kept.append(f)
            continue

        prev = kept[-1]
        p0 = int(prev.get("engage_ts", -1))
        if p0 < 0:
            kept[-1] = f
            continue

        prev_label_end = _label_end_ts(prev, horizon_ms)
        gap_from_prev_start = t0 - p0

        overlap = (t0 < prev_label_end)
        too_close = (gap_from_prev_start < int(fight_min_gap_ms))

        if overlap or too_close:
            # Keep simultaneous/nearby starts if they are clearly different locations.
            if float(location_radius) > 0:
                try:
                    pcx = float(prev.get("centroid_x", float("nan")))
                    pcy = float(prev.get("centroid_y", float("nan")))
                    ccx = float(f.get("centroid_x", float("nan")))
                    ccy = float(f.get("centroid_y", float("nan")))
                    if np.isfinite(pcx) and np.isfinite(pcy) and np.isfinite(ccx) and np.isfinite(ccy):
                        if _distance_2d((pcx, pcy), (ccx, ccy)) > float(location_radius):
                            kept.append(f)
                            continue
                except Exception:
                    pass

            conflicts += 1
            sp = _fight_priority_score(prev, kill_ts=kill_ts, horizon_ms=horizon_ms)
            sc = _fight_priority_score(f, kill_ts=kill_ts, horizon_ms=horizon_ms)
            if sc > sp:
                kept[-1] = f
                replaced += 1
            else:
                removed += 1
            continue

        kept.append(f)

    if diag is not None:
        diag["postmerge_conflicts"] = int(diag.get("postmerge_conflicts", 0) or 0) + int(conflicts)
        diag["postmerge_removed"] = int(diag.get("postmerge_removed", 0) or 0) + int(removed)
        diag["postmerge_replaced"] = int(diag.get("postmerge_replaced", 0) or 0) + int(replaced)

    return kept


# ============================================================================
# 입력 검증
# ============================================================================


def validate_team_mapping(tm: Dict[int, int], n_players: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """팀 매핑 검증 및 인덱스 추출"""
    if not tm:
        logger.warning("Empty team mapping, using defaults")
        tm = {i: 100 if i <= 5 else 200 for i in range(1, 11)}

    tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, n_players + 1)], dtype=np.int32)

    b = np.where(tids == 100)[0]
    r = np.where(tids == 200)[0]

    if len(b) == 0:
        unique = np.unique(tids)
        if len(unique) >= 2:
            b = np.where(tids == unique[0])[0]
            r = np.where(tids == unique[1])[0]
        else:
            b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
            r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

    return b, r


def detect_coordinate_scale(xy: np.ndarray) -> Tuple[bool, float]:
    """좌표 스케일 자동 감지"""
    valid_xy = xy[~np.isnan(xy)]
    if len(valid_xy) == 0:
        return False, 1.0

    max_val = float(np.max(valid_xy))
    min_val = float(np.min(valid_xy))

    if max_val <= MapConstants.NORM_THRESHOLD and min_val >= -1.0:
        return True, MapConstants.NORM_DIVISOR
    if max_val <= 100 and min_val >= 0:
        return True, (MapConstants.NORM_DIVISOR / max_val if max_val > 0 else MapConstants.NORM_DIVISOR)
    if max_val > 100:
        return False, 1.0

    return True, MapConstants.NORM_DIVISOR


# ============================================================================
# 이벤트 처리
# ============================================================================


def normalize_patch(game_version: str) -> str:
    s = str(game_version or "")
    parts = s.split(".")
    if len(parts) >= 2:
        patch_level = getattr(cfg, "PATCH_LEVEL", "major_minor") if cfg else "major_minor"
        if patch_level == "full":
            return s
        return f"{parts[0]}.{parts[1]}"
    return s or "0.0"


def _event_xy(e: dict) -> Optional[Tuple[float, float]]:
    if not isinstance(e, dict):
        return None
    pos = e.get("position", None)
    if isinstance(pos, dict):
        x = pos.get("x", None)
        y = pos.get("y", None)
        if x is not None and y is not None:
            return (safe_float(x), safe_float(y))
    if ("x" in e) and ("y" in e):
        return (safe_float(e.get("x")), safe_float(e.get("y")))
    return None


def build_anchors_from_events(events: List[dict]) -> Dict[str, Any]:
    obj = {k: [] for k in ["DRAGON", "BARON", "RIFTHERALD", "ATAKHAN", "HORDE"]}
    tower = {"TOWER_T100": [], "TOWER_T200": []}

    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type", "")).upper()
        xy = _event_xy(e)
        if xy is None:
            continue
        x, y = xy
        if not (0 <= x <= MapConstants.MAX_COORDINATE and 0 <= y <= MapConstants.MAX_COORDINATE):
            continue

        if et == "ELITE_MONSTER_KILL":
            mt = str(e.get("monsterType", "")).upper()
            if mt == "BARON_NASHOR":
                mt = "BARON"
            if mt in obj:
                obj[mt].append([x, y])
        elif et == "BUILDING_KILL":
            bt = str(e.get("buildingType", "")).upper()
            if "TOWER" in bt:
                victim_team = safe_int(e.get("teamId", 0))
                if victim_team in (100, 200):
                    tower[f"TOWER_T{victim_team}"].append([x, y])
        elif et == "TURRET_PLATE_DESTROYED":
            victim_team = safe_int(e.get("teamId", 0))
            if victim_team in (100, 200):
                tower[f"TOWER_T{victim_team}"].append([x, y])

    def _dedup(points: List[List[float]], r: float = 10.0) -> List[List[float]]:
        seen = set()
        out = []
        for px, py in points:
            k = (int(round(px / r)), int(round(py / r)))
            if k in seen:
                continue
            seen.add(k)
            out.append([float(px), float(py)])
        return out

    for k in list(obj.keys()):
        obj[k] = _dedup(obj[k])
    for k in list(tower.keys()):
        tower[k] = _dedup(tower[k])

    return {"obj": obj, "tower": tower}


def _extract_kill_events(events: List[dict]) -> List[dict]:
    kills: List[dict] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = ev.get("type", ev.get("eventType", ""))
        if str(et).upper() == "CHAMPION_KILL":
            ts = ev.get("timestamp", ev.get("ts"))
            if ts is not None:
                try:
                    kills.append(
                        {
                            "timestamp": int(ts),
                            "killer_id": safe_int(ev.get("killerId", 0)),
                            "victim_id": safe_int(ev.get("victimId", 0)),
                            "assisting_ids": ev.get("assistingParticipantIds", []),
                            "position": _event_xy(ev),
                        }
                    )
                except Exception:
                    pass
    kills.sort(key=lambda x: x["timestamp"])
    return kills


def _extract_kill_events_rich(events: List[dict]) -> List[dict]:
    """Extract CHAMPION_KILL events preserving victimDamageReceived/Dealt arrays.

    Returns enriched kill dicts with damage breakdown per participant,
    used by kill-chain detection to determine exact fight participants.
    """
    kills: List[dict] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = ev.get("type", ev.get("eventType", ""))
        if str(et).upper() == "CHAMPION_KILL":
            ts = ev.get("timestamp", ev.get("ts"))
            if ts is not None:
                try:
                    kills.append(
                        {
                            "timestamp": int(ts),
                            "killer_id": safe_int(ev.get("killerId", 0)),
                            "victim_id": safe_int(ev.get("victimId", 0)),
                            "assisting_ids": ev.get("assistingParticipantIds", []) or [],
                            "position": _event_xy(ev),
                            "victim_damage_received": ev.get("victimDamageReceived", []) or [],
                            "victim_damage_dealt": ev.get("victimDamageDealt", []) or [],
                            "bounty": safe_int(ev.get("bounty", 0)),
                            "kill_streak_length": safe_int(ev.get("killStreakLength", 0)),
                            "shutdown_bounty": safe_int(ev.get("shutdownBounty", 0)),
                        }
                    )
                except Exception:
                    pass
    kills.sort(key=lambda x: x["timestamp"])
    return kills


def _get_kill_participants(kill: dict) -> set:
    """Extract all participant IDs involved in a kill from damage arrays.

    Goes beyond killer/victim/assists: includes everyone who dealt damage
    to the victim (from victimDamageReceived) and everyone the victim
    damaged before dying (from victimDamageDealt).
    """
    pids: set = set()
    kid = safe_int(kill.get("killer_id", 0))
    vid = safe_int(kill.get("victim_id", 0))
    if 1 <= kid <= 10:
        pids.add(kid)
    if 1 <= vid <= 10:
        pids.add(vid)
    for aid in kill.get("assisting_ids", []) or []:
        a = safe_int(aid)
        if 1 <= a <= 10:
            pids.add(a)
    for dmg in kill.get("victim_damage_received", []) or []:
        if isinstance(dmg, dict):
            p = safe_int(dmg.get("participantId", 0))
            if 1 <= p <= 10:
                pids.add(p)
    for dmg in kill.get("victim_damage_dealt", []) or []:
        if isinstance(dmg, dict):
            p = safe_int(dmg.get("participantId", 0))
            if 1 <= p <= 10:
                pids.add(p)
    return pids


def _build_kill_chains(
    kills: List[dict],
    chain_window_ms: int = 30000,
) -> List[dict]:
    """Group kills into fight chains by participant overlap within time window.

    Two kills are chained if they share at least one participant (from damage
    arrays) AND occur within chain_window_ms of each other. Uses Union-Find
    for transitive chaining: if kill A chains to B and B chains to C, all
    three form one fight chain.

    Returns list of chain dicts sorted by start_ts, each containing:
      - kills: sorted list of kill events in the chain
      - participants: set of all participant IDs involved
      - start_ts, end_ts: first and last kill timestamps
      - centroid_x, centroid_y: mean of kill positions
      - n_kills: number of kills in chain
    """
    N = len(kills)
    if N == 0:
        return []

    kill_pids = [_get_kill_participants(k) for k in kills]

    uf = UnionFind(N)
    for i in range(N):
        ti = kills[i]["timestamp"]
        for j in range(i + 1, N):
            if kills[j]["timestamp"] - ti > chain_window_ms:
                break  # kills are sorted, all further j are beyond window
            if kill_pids[i] & kill_pids[j]:
                uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(N):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    chains: List[dict] = []
    for indices in groups.values():
        chain_kills = [kills[idx] for idx in indices]
        chain_kills.sort(key=lambda k: k["timestamp"])

        all_pids: set = set()
        for idx in indices:
            all_pids |= kill_pids[idx]

        positions = [k["position"] for k in chain_kills if k.get("position")]
        if positions:
            cx = float(np.mean([p[0] for p in positions]))
            cy = float(np.mean([p[1] for p in positions]))
        else:
            cx, cy = 0.0, 0.0

        chains.append(
            {
                "kills": chain_kills,
                "participants": all_pids,
                "start_ts": chain_kills[0]["timestamp"],
                "end_ts": chain_kills[-1]["timestamp"],
                "centroid_x": cx,
                "centroid_y": cy,
                "n_kills": len(chain_kills),
            }
        )

    chains.sort(key=lambda c: c["start_ts"])
    return chains


def _extract_summoner_spell_ts(events: List[dict]) -> np.ndarray:
    ts: List[int] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et not in ("SUMMONER_SPELL_USED", "SUMMONER_SPELL_CAST"):
            continue
        try:
            t = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            t = -1
        if t >= 0:
            ts.append(int(t))
    if not ts:
        return np.empty((0,), dtype=np.int64)
    ts.sort()
    return np.asarray(ts, dtype=np.int64)


def _extract_objective_building_ts(events: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    obj_ts: List[int] = []
    bld_ts: List[int] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        try:
            t = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            t = -1
        if t < 0:
            continue
        if et == "ELITE_MONSTER_KILL":
            obj_ts.append(int(t))
        elif et in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
            bld_ts.append(int(t))
    obj_ts.sort()
    bld_ts.sort()
    return (
        np.asarray(obj_ts, dtype=np.int64) if obj_ts else np.empty((0,), dtype=np.int64),
        np.asarray(bld_ts, dtype=np.int64) if bld_ts else np.empty((0,), dtype=np.int64),
    )


def _extract_ward_signal_ts(events: List[dict], tm: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ward-kill ts and ward-activity ts per team (100/200)."""
    ward_kill_ts: List[int] = []
    ward_t100_ts: List[int] = []
    ward_t200_ts: List[int] = []

    def _team_of_pid(pid: int) -> int:
        pid = int(pid or 0)
        tid = int(tm.get(pid, 0) or 0) if isinstance(tm, dict) else 0
        if tid in (100, 200):
            return tid
        if 1 <= pid <= 5:
            return 100
        if 6 <= pid <= 10:
            return 200
        return 0

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et not in ("WARD_KILL", "WARD_PLACED"):
            continue
        try:
            t = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            t = -1
        if t < 0:
            continue

        if et == "WARD_KILL":
            ward_kill_ts.append(int(t))
            pid = int(ev.get("killerId", 0) or ev.get("creatorId", 0) or 0)
        else:
            pid = int(ev.get("creatorId", 0) or ev.get("participantId", 0) or 0)

        tid = _team_of_pid(pid)
        if tid == 100:
            ward_t100_ts.append(int(t))
        elif tid == 200:
            ward_t200_ts.append(int(t))

    ward_kill_ts.sort()
    ward_t100_ts.sort()
    ward_t200_ts.sort()
    return (
        np.asarray(ward_kill_ts, dtype=np.int64) if ward_kill_ts else np.empty((0,), dtype=np.int64),
        np.asarray(ward_t100_ts, dtype=np.int64) if ward_t100_ts else np.empty((0,), dtype=np.int64),
        np.asarray(ward_t200_ts, dtype=np.int64) if ward_t200_ts else np.empty((0,), dtype=np.int64),
    )


def _extract_ward_actor_events(events: List[dict], tm: Dict[int, int]) -> List[Dict[str, int]]:
    """Extract ward activity events with actor identity/team for spatial validation."""
    out: List[Dict[str, int]] = []

    def _team_of_pid(pid: int) -> int:
        pid = int(pid or 0)
        tid = int(tm.get(pid, 0) or 0) if isinstance(tm, dict) else 0
        if tid in (100, 200):
            return tid
        if 1 <= pid <= 5:
            return 100
        if 6 <= pid <= 10:
            return 200
        return 0

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et not in ("WARD_KILL", "WARD_PLACED"):
            continue
        try:
            t = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            t = -1
        if t < 0:
            continue

        is_kill = 1 if et == "WARD_KILL" else 0
        if is_kill:
            pid = int(ev.get("killerId", 0) or ev.get("creatorId", 0) or 0)
        else:
            pid = int(ev.get("creatorId", 0) or ev.get("participantId", 0) or 0)

        if not (1 <= pid <= 10):
            continue
        tid = _team_of_pid(pid)
        if tid not in (100, 200):
            continue

        out.append(
            {
                "timestamp": int(t),
                "pid": int(pid),
                "team": int(tid),
                "is_kill": int(is_kill),
            }
        )

    out.sort(key=lambda e: (int(e["timestamp"]), int(e["pid"]), -int(e["is_kill"])))
    return out


def _extract_ace_ts(events: List[dict]) -> np.ndarray:
    """Extract CHAMPION_SPECIAL_KILL ACE timestamps (sorted unique)."""
    ts: List[int] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et != "CHAMPION_SPECIAL_KILL":
            continue
        kt = str(ev.get("killType", "")).upper()
        if "ACE" not in kt:
            continue
        try:
            t = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            t = -1
        if t >= 0:
            ts.append(int(t))
    if not ts:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(sorted(set(ts)), dtype=np.int64)


def _first_kill_in_window(kill_ts: np.ndarray, t0: int, t1_exclusive: int) -> Optional[int]:
    """Return first kill in half-open window [t0, t1_exclusive)."""
    if kill_ts.size == 0:
        return None
    i = int(np.searchsorted(kill_ts, t0, side="left"))
    if i < kill_ts.size and int(kill_ts[i]) < t1_exclusive:
        return int(kill_ts[i])
    return None


def _latest_ts_before(ts_sorted: np.ndarray, t_exclusive: int) -> Optional[int]:
    """Return latest timestamp < t_exclusive from sorted np array."""
    if not isinstance(ts_sorted, np.ndarray) or ts_sorted.size == 0:
        return None
    i = int(np.searchsorted(ts_sorted, int(t_exclusive), side="left")) - 1
    if i < 0:
        return None
    return int(ts_sorted[i])


def _ensure_start_before_recent_kill(
    *,
    engage_ts_val: int,
    ref_ts_exclusive: int,
    kill_ts: np.ndarray,
    t_min_ms: int,
    pre_ms: int,
    max_gap_ms: int,
    earliest_signal_before: Optional[Callable[[int, int], Optional[int]]] = None,
) -> Tuple[int, bool]:
    """If a recent prior kill exists and engage is not before it, shift engage before that kill."""
    prev_kill = _latest_ts_before(kill_ts, int(ref_ts_exclusive))
    if prev_kill is None:
        return int(engage_ts_val), False

    if int(max_gap_ms) > 0 and (int(ref_ts_exclusive) - int(prev_kill)) > int(max_gap_ms):
        return int(engage_ts_val), False

    if int(engage_ts_val) < int(prev_kill):
        return int(engage_ts_val), False

    new_start = int(engage_ts_val)
    if callable(earliest_signal_before):
        try:
            sig = earliest_signal_before(int(prev_kill), int(max_gap_ms))
        except Exception:
            sig = None
        if sig is not None and int(sig) < int(prev_kill):
            new_start = int(min(new_start, int(sig)))

    if new_start >= int(prev_kill):
        back_ms = max(1, int(pre_ms))
        new_start = int(min(new_start, max(int(t_min_ms), int(prev_kill) - back_ms)))
    if new_start >= int(prev_kill):
        new_start = int(max(int(t_min_ms), int(prev_kill) - 1))

    return int(new_start), (int(new_start) != int(engage_ts_val))


def _truncate_fights_at_ace(
    fights: List[dict],
    ace_ts: np.ndarray,
    *,
    horizon_ms: int,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """If ACE occurs in a fight window, clamp fight end to that ACE timestamp."""
    ace_cnt = int(ace_ts.size) if isinstance(ace_ts, np.ndarray) else 0
    clipped = 0

    if fights and ace_cnt > 0:
        for f in fights:
            try:
                t0 = int(f.get("engage_ts", -1))
            except Exception:
                t0 = -1
            if t0 < 0:
                continue

            t1 = int(_label_end_ts(f, int(horizon_ms)))
            if t1 <= t0:
                continue

            i = int(np.searchsorted(ace_ts, t0, side="left"))
            if i >= ace_ts.size:
                continue
            ace_t = int(ace_ts[i])
            if ace_t >= t1:
                continue

            # label window is [engage, end), so use ace_t + 1 to include same-ms events.
            new_end = int(max(t0 + 1, ace_t + 1))
            if new_end >= t1:
                continue

            f["horizon_end_ts"] = int(new_end)
            f["det_end_by_ace"] = 1
            f["det_ace_ts"] = int(ace_t)

            subs = f.get("sub_segments", None)
            if isinstance(subs, list) and subs:
                kept_subs = []
                for s in subs:
                    try:
                        st = int(s.get("engage_ts", -1))
                    except Exception:
                        st = -1
                    if st < 0 or st <= ace_t:
                        kept_subs.append(s)
                if len(kept_subs) != len(subs):
                    f["sub_segments"] = kept_subs
                    f["n_segments"] = int(1 + len(kept_subs))

            clipped += 1

    if diag is not None:
        diag["ace_events"] = int(ace_cnt)
        diag["ace_end_truncated"] = int(diag.get("ace_end_truncated", 0) or 0) + int(clipped)


# ============================================================================
# 보간 및 거리 계산
# ============================================================================


def _interp_xy_dense(
    xy_minute: np.ndarray,
    minute_ts: np.ndarray,
    step_ms: int,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(len(minute_ts))
    if T < 2:
        return minute_ts.copy(), xy_minute.astype(np.float32, copy=False)

    t_start = int(minute_ts[0])
    t_end = int(minute_ts[-1])
    dense_ts = np.arange(t_start, t_end + step_ms, step_ms, dtype=np.int64)
    Td = int(len(dense_ts))

    left = np.searchsorted(minute_ts, dense_ts, side="right") - 1
    left = np.clip(left, 0, T - 2)
    right = left + 1

    method = str(method or "none").lower()
    if method in ("none", "zoh"):
        return dense_ts, xy_minute[left].astype(np.float32, copy=False)

    tL = minute_ts[left].astype(np.float32)
    tR = minute_ts[right].astype(np.float32)
    denom = np.maximum(tR - tL, 1.0)
    a = ((dense_ts.astype(np.float32) - tL) / denom).reshape(Td, 1, 1)

    xyL = xy_minute[left].astype(np.float32)
    xyR = xy_minute[right].astype(np.float32)
    xy_dense = (1.0 - a) * xyL + a * xyR
    return dense_ts, xy_dense.astype(np.float32, copy=False)


def _map_ts_to_minute_idx(minute_ts: np.ndarray, ts: int) -> int:
    m = int(np.searchsorted(minute_ts, ts, side="right") - 1)
    return int(np.clip(m, 0, len(minute_ts) - 1))


def compute_distances_chunked(
    xy_dense: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
    chunk_size: int = 500,
) -> np.ndarray:
    """청크 단위 거리 계산 (메모리 최적화)"""
    Td = len(xy_dense)
    dists = np.empty((Td, 5, 5), dtype=np.float32)

    for start in range(0, Td, chunk_size):
        end = min(start + chunk_size, Td)
        xb = xy_dense[start:end, b, :]
        xr = xy_dense[start:end, r, :]
        diff = xb[:, :, None, :] - xr[:, None, :, :]
        dists[start:end] = np.sqrt(np.sum(diff * diff, axis=-1))
        del diff

    return dists


# ============================================================================
# 교전 분석 함수
# ============================================================================


def _get_engaged_player_indices(
    dists: np.ndarray, d_idx: int, R: float, b: np.ndarray, r: np.ndarray
) -> Tuple[List[int], List[int]]:
    dist_mat = dists[d_idx]
    blue_engaged = [int(b[i]) for i in range(5) if np.min(dist_mat[i, :]) <= R]
    red_engaged = [int(r[j]) for j in range(5) if np.min(dist_mat[:, j]) <= R]
    return blue_engaged, red_engaged


def _compute_fight_centroid(xy_dense: np.ndarray, d_idx: int, engaged_players: List[int]) -> Tuple[float, float]:
    if not engaged_players:
        return float(np.mean(xy_dense[d_idx, :, 0])), float(np.mean(xy_dense[d_idx, :, 1]))
    pts = xy_dense[d_idx, engaged_players, :]
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _compute_fight_centroid_from_dists(
    xy_dense: np.ndarray,
    dists: np.ndarray,
    dense_ts: np.ndarray,
    engage_ts: int,
    R: float,
    b: np.ndarray,
    r: np.ndarray,
) -> Tuple[float, float]:
    Td = len(dense_ts)
    d_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts, side="right") - 1, 0, Td - 1))
    blue_engaged, red_engaged = _get_engaged_player_indices(dists, d_idx, R, b, r)
    return _compute_fight_centroid(xy_dense, d_idx, blue_engaged + red_engaged)


def _is_continuous_fight(
    prev_fight: dict,
    curr_engage_ts: int,
    curr_centroid: Tuple[float, float],
    max_gap_ms: int,
    merge_radius: float,
) -> bool:
    """연속 교전 판정(레거시용).

    [FIX #5] gap_ms < 0 (시간 역전) 인 경우를 명시적으로 거부한다.
    """
    if "sub_segments" in prev_fight and len(prev_fight["sub_segments"]) > 0:
        last_ts = prev_fight["sub_segments"][-1].get("engage_ts", 0)
        last_cx = prev_fight["sub_segments"][-1].get("centroid_x", 0.0)
        last_cy = prev_fight["sub_segments"][-1].get("centroid_y", 0.0)
    else:
        last_ts = prev_fight.get("engage_ts", prev_fight.get("t_engage_ts", 0))
        last_cx = prev_fight.get("centroid_x", 0.0)
        last_cy = prev_fight.get("centroid_y", 0.0)

    gap_ms = curr_engage_ts - last_ts
    if gap_ms < 0:
        return False
    if gap_ms > max_gap_ms:
        return False
    return _distance_2d((last_cx, last_cy), curr_centroid) <= merge_radius


def backtrack_engage_ts(
    dists: np.ndarray,
    dense_ts: np.ndarray,
    prox_pairs: np.ndarray,
    kill_ts: int,
    *,
    R: float,
    min_pairs: int = 3,
    max_lookback_ms: int = 60000,
    min_lookback_ms: int = 5000,
) -> Tuple[int, bool]:
    """킬 시점으로부터 역추적하여 교전 시작점을 찾는다.

    [FIX #4] proximity에서 찾지 못한 경우에만 fallback을 사용.
    """
    Td = len(dense_ts)
    if Td == 0:
        return max(0, kill_ts - min_lookback_ms), False

    kill_idx = int(np.clip(np.searchsorted(dense_ts, kill_ts, side="right") - 1, 0, Td - 1))
    lookback_start_ts = kill_ts - max_lookback_ms
    lookback_start_idx = int(np.clip(np.searchsorted(dense_ts, lookback_start_ts, side="left"), 0, kill_idx))

    if lookback_start_idx >= kill_idx:
        return max(0, kill_ts - min_lookback_ms), False

    engage_idx: Optional[int] = None
    in_engage = prox_pairs[kill_idx] >= min_pairs

    if in_engage:
        engage_idx = kill_idx
        for i in range(kill_idx - 1, lookback_start_idx - 1, -1):
            if prox_pairs[i] >= min_pairs:
                engage_idx = i
            else:
                break
    else:
        for i in range(kill_idx - 1, lookback_start_idx - 1, -1):
            if prox_pairs[i] >= min_pairs:
                engage_idx = i
                break

    if engage_idx is None:
        return max(0, kill_ts - min_lookback_ms), False

    engage_ts = int(dense_ts[engage_idx])
    if engage_ts > kill_ts:
        return max(0, kill_ts - min_lookback_ms), False

    return max(0, engage_ts), True


# ============================================================================
# Union-Find (LCC 계산 + ST-DBSCAN 병합용)
# ============================================================================


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def _largest_bipartite_cc(dist_5x5: np.ndarray, Rthr: float) -> Tuple[int, int, int, List[int]]:
    uf = UnionFind(10)
    for i in range(5):
        for j in range(5):
            if dist_5x5[i, j] <= Rthr:
                uf.union(i, 5 + j)

    comp_members: Dict[int, List[int]] = {}
    for i in range(10):
        root = uf.find(i)
        comp_members.setdefault(root, []).append(i)

    best_comp = max(comp_members.values(), key=len, default=[])
    best_b = sum(1 for x in best_comp if x < 5)
    best_r = sum(1 for x in best_comp if x >= 5)
    return len(best_comp), best_b, best_r, best_comp


def _bipartite_components(dist_5x5: np.ndarray, Rthr: float) -> List[Tuple[List[int], int, int]]:
    """Return connected components (node ids in [0..9]) with at least one per team."""
    uf = UnionFind(10)
    for i in range(5):
        for j in range(5):
            if dist_5x5[i, j] <= Rthr:
                uf.union(i, 5 + j)

    comp_members: Dict[int, List[int]] = {}
    for i in range(10):
        root = uf.find(i)
        comp_members.setdefault(root, []).append(i)

    out: List[Tuple[List[int], int, int]] = []
    for nodes in comp_members.values():
        b_cnt = sum(1 for x in nodes if x < 5)
        r_cnt = sum(1 for x in nodes if x >= 5)
        if b_cnt <= 0 or r_cnt <= 0:
            continue
        out.append((nodes, int(b_cnt), int(r_cnt)))
    out.sort(key=lambda x: (-len(x[0]), -x[1], -x[2]))
    return out


# ============================================================================
# ST-DBSCAN(minPts=1) 병합: 후보 fight들을 시공간 ε-연결 컴포넌트로 병합
# ============================================================================


def _split_by_max_duration(sorted_cands: List[dict], horizon_ms: int, max_duration_ms: int) -> List[List[dict]]:
    """시간순 정렬된 후보들을 max_duration 제약으로 분할."""
    if not sorted_cands:
        return []
    groups: List[List[dict]] = [[sorted_cands[0]]]
    start_ts = int(sorted_cands[0]["engage_ts"])
    for f in sorted_cands[1:]:
        t = int(f["engage_ts"])
        potential_end = t + horizon_ms
        if potential_end - start_ts > max_duration_ms:
            groups.append([f])
            start_ts = t
        else:
            groups[-1].append(f)
    return groups


def _merge_group(group: List[dict], horizon_ms: int) -> dict:
    """동일 클러스터(또는 sub-group)를 하나의 교전으로 병합."""
    if len(group) == 1:
        g0 = group[0]
        g0.setdefault("n_segments", 1)
        g0.setdefault("horizon_end_ts", int(g0["engage_ts"]) + horizon_ms)
        return g0

    group_sorted = sorted(group, key=lambda x: int(x["engage_ts"]))
    primary = dict(group_sorted[0])  # shallow copy
    primary_ts = int(primary["engage_ts"])

    # sub_segments: primary 제외 나머지
    sub_segments: List[dict] = []
    for f in group_sorted[1:]:
        sub_segments.append(
            {
                "engage_ts": int(f.get("engage_ts", -1)),
                "centroid_x": float(f.get("centroid_x", 0.0)),
                "centroid_y": float(f.get("centroid_y", 0.0)),
                "first_kill_ts": int(f.get("first_kill_ts", -1) or -1),
                "det_anchor": int(f.get("det_anchor", 0) or 0),
                "det_backtracked": int(f.get("det_backtracked", 0) or 0),
                "det_prox_pairs": int(f.get("det_prox_pairs", 0) or 0),
            }
        )

    primary["sub_segments"] = sub_segments
    primary["n_segments"] = int(len(group_sorted))

    # horizon_end_ts: 멤버 중 최대 (연속교전 라벨의 종료시점 후보)
    max_end = primary.get("horizon_end_ts", primary_ts + horizon_ms)
    for f in group_sorted:
        max_end = max(int(max_end), int(f.get("horizon_end_ts", int(f["engage_ts"]) + horizon_ms)))
    primary["horizon_end_ts"] = int(max_end)

    # det flags: cluster 내 OR로 승격
    primary["det_anchor"] = int(max(int(f.get("det_anchor", 0) or 0) for f in group_sorted))
    primary["det_backtracked"] = int(max(int(f.get("det_backtracked", 0) or 0) for f in group_sorted))

    # first_kill_ts: 가장 이른 양수(없으면 -1)
    fk_vals = [int(f.get("first_kill_ts", -1) or -1) for f in group_sorted]
    fk_pos = [v for v in fk_vals if v >= 0]
    primary["first_kill_ts"] = int(min(fk_pos)) if fk_pos else -1

    # det_prox_pairs: primary의 frame 기준 값 유지 (하지만 cluster max도 별도 저장)
    try:
        primary["det_prox_pairs_max_cluster"] = int(max(int(f.get("det_prox_pairs", 0) or 0) for f in group_sorted))
    except Exception:
        pass

    return primary


def merge_fights_st_dbscan_unionfind(
    candidates: List[dict],
    *,
    eps_t_ms: int,
    eps_s: float,
    horizon_ms: int,
    max_duration_ms: int,
    diag: Optional[dict] = None,
) -> List[dict]:
    """Union-Find 기반 ST-DBSCAN(minPts=1).

    - 점: p_i = (t_i / alpha, x_i, y_i)
    - alpha = eps_t_ms / eps_s
    - eps = eps_s
    - minPts=1 → ε-이웃 그래프의 Connected Components와 동치
    """
    N = len(candidates)
    if N <= 1:
        return candidates

    if eps_s <= 0:
        return candidates

    alpha = float(eps_t_ms) / float(eps_s)  # ms per distance-unit
    if alpha <= 0:
        return candidates

    # 준비: 시간/좌표 추출
    t = np.array([float(c["engage_ts"]) for c in candidates], dtype=np.float64)  # ms
    x = np.array([float(c["centroid_x"]) for c in candidates], dtype=np.float64)
    y = np.array([float(c["centroid_y"]) for c in candidates], dtype=np.float64)

    eps2 = float(eps_s) * float(eps_s)
    uf = UnionFind(N)

    # O(N^2), N(경기당 후보수)은 작다는 가정
    for i in range(N):
        ti = t[i]
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, N):
            dt = (ti - t[j]) / alpha  # distance-unit
            dx = xi - x[j]
            dy = yi - y[j]
            if (dt * dt + dx * dx + dy * dy) <= eps2:
                uf.union(i, j)

    # 그룹화
    groups: Dict[int, List[int]] = {}
    for i in range(N):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    merged: List[dict] = []
    merged_segments = 0
    for idxs in groups.values():
        group = [candidates[i] for i in idxs]
        group.sort(key=lambda c: int(c["engage_ts"]))

        # max_duration 제약으로 분할
        for sub in _split_by_max_duration(group, horizon_ms, max_duration_ms):
            merged.append(_merge_group(sub, horizon_ms))
            if len(sub) > 1:
                merged_segments += (len(sub) - 1)

    merged.sort(key=lambda c: int(c["engage_ts"]))

    if diag is not None:
        diag["continuous_merged"] = int(merged_segments)
        diag["continuous_clusters"] = int(sum(1 for idxs in groups.values() if len(idxs) > 1))

    return merged


def merge_fights_temporal_unionfind(
    candidates: List[dict],
    *,
    max_gap_ms: int,
    horizon_ms: int,
    max_duration_ms: int,
    diag: Optional[dict] = None,
) -> List[dict]:
    """Time-only connected-components merge for event-driven detection."""
    if len(candidates) <= 1:
        return candidates

    if max_gap_ms < 0:
        max_gap_ms = 0

    cand = sorted(candidates, key=lambda x: int(x.get("engage_ts", -1)))
    groups: List[List[dict]] = []
    cur: List[dict] = []
    prev_t = -10**18

    for c in cand:
        t = int(c.get("engage_ts", -1))
        if t < 0:
            continue
        if (not cur) or (t - prev_t <= int(max_gap_ms)):
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
        prev_t = t
    if cur:
        groups.append(cur)

    merged: List[dict] = []
    merged_segments = 0
    for g in groups:
        g.sort(key=lambda x: int(x.get("engage_ts", -1)))
        for sub in _split_by_max_duration(g, horizon_ms, max_duration_ms):
            merged.append(_merge_group(sub, horizon_ms))
            if len(sub) > 1:
                merged_segments += (len(sub) - 1)

    merged.sort(key=lambda x: int(x.get("engage_ts", -1)))
    if diag is not None:
        diag["continuous_merged"] = int(merged_segments)
        diag["continuous_clusters"] = int(sum(1 for g in groups if len(g) > 1))
    return merged


# ============================================================================
# 교전 유형/결과/참여도/중요도 계산
# ============================================================================


def classify_fight_type(fight: dict, anchors: Dict[str, Any], is_norm: bool, scale_factor: float) -> str:
    cx = fight.get("centroid_x", 0.0)
    cy = fight.get("centroid_y", 0.0)

    if is_norm:
        cx *= scale_factor
        cy *= scale_factor

    centroid = (cx, cy)

    if _distance_2d(centroid, MapConstants.BARON_PIT) < MapConstants.OBJECTIVE_RADIUS:
        return FightType.OBJECTIVE_BARON.value
    if _distance_2d(centroid, MapConstants.DRAGON_PIT) < MapConstants.OBJECTIVE_RADIUS:
        return FightType.OBJECTIVE_DRAGON.value

    obj_positions = anchors.get("obj", {})
    for pos in obj_positions.get("BARON", []):
        if _distance_2d(centroid, tuple(pos)) < MapConstants.OBJECTIVE_RADIUS:
            return FightType.OBJECTIVE_BARON.value
    for pos in obj_positions.get("DRAGON", []):
        if _distance_2d(centroid, tuple(pos)) < MapConstants.OBJECTIVE_RADIUS:
            return FightType.OBJECTIVE_DRAGON.value
    for pos in obj_positions.get("RIFTHERALD", []):
        if _distance_2d(centroid, tuple(pos)) < MapConstants.OBJECTIVE_RADIUS:
            return FightType.OBJECTIVE_RIFTHERALD.value

    tower_positions = anchors.get("tower", {})
    for tower_key in ["TOWER_T100", "TOWER_T200"]:
        for pos in tower_positions.get(tower_key, []):
            if _distance_2d(centroid, tuple(pos)) < MapConstants.TOWER_RADIUS:
                return FightType.TOWER_DIVE.value

    if (_distance_2d(centroid, MapConstants.BLUE_BASE) < MapConstants.BASE_RADIUS) or (
        _distance_2d(centroid, MapConstants.RED_BASE) < MapConstants.BASE_RADIUS
    ):
        return FightType.BASE_FIGHT.value

    prox_pairs = int(fight.get("det_prox_pairs", 0) or 0)
    if prox_pairs >= 8:
        return FightType.TEAMFIGHT.value
    if prox_pairs >= 4:
        return FightType.SKIRMISH.value
    return FightType.PICK.value


def _team_of_pid(pid: int, tm: Dict[int, int]) -> int:
    pid = int(pid or 0)
    tid = int(tm.get(pid, 0) or 0) if isinstance(tm, dict) else 0
    if tid in (100, 200):
        return tid
    if 1 <= pid <= 5:
        return 100
    if 6 <= pid <= 10:
        return 200
    return 0


def _gold_team_at_ms(cache: Optional[Dict[str, Any]], q_ms: int) -> Optional[np.ndarray]:
    if not isinstance(cache, dict):
        return None

    ts_raw = cache.get("minute_ts", None)
    g_raw = cache.get("gold_team_minute", None)

    try:
        ts = np.asarray(ts_raw, dtype=np.int64)
        g = np.asarray(g_raw, dtype=np.float32)
    except Exception:
        return None

    if ts.ndim != 1 or ts.size <= 0:
        return None
    if g.ndim != 2 or g.shape[0] != ts.size or g.shape[1] < 2:
        return None

    if ts.size == 1:
        return g[0, :2].astype(np.float32, copy=False)

    q = int(q_ms)
    idx = int(np.searchsorted(ts, q, side="right") - 1)
    i = int(np.clip(idx, 0, ts.size - 1))
    j = int(np.clip(i + 1, 0, ts.size - 1))

    method = "ffill"
    if cfg is not None:
        method = str(
            getattr(
                cfg,
                "LABEL_GOLD_METHOD",
                getattr(cfg, "INTERP_SCALARS_METHOD", "ffill"),
            )
        ).lower().strip()

    if method in ("bfill",):
        return g[j, :2].astype(np.float32, copy=False)

    if method in ("linear",) and j > i and int(ts[j]) != int(ts[i]):
        alpha = float(q - int(ts[i])) / float(int(ts[j]) - int(ts[i]))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return ((1.0 - alpha) * g[i, :2] + alpha * g[j, :2]).astype(np.float32)

    # ffill / zoh / none / default
    return g[i, :2].astype(np.float32, copy=False)


def _window_resource_changes(events: List[dict], tm: Dict[int, int], t0: int, t1_exclusive: int) -> Dict[str, Any]:
    tower_blue = tower_red = 0
    plate_blue = plate_red = 0
    inhib_blue = inhib_red = 0
    obj_by_type = {
        "dragon": {"blue": 0, "red": 0},
        "baron": {"blue": 0, "red": 0},
        "herald": {"blue": 0, "red": 0},
        "atakhan": {"blue": 0, "red": 0},
        "horde": {"blue": 0, "red": 0},
        "other": {"blue": 0, "red": 0},
    }

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        try:
            ts = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            ts = -1
        if ts < int(t0) or ts >= int(t1_exclusive):
            continue

        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et == "ELITE_MONSTER_KILL":
            team = int(ev.get("killerTeamId", 0) or 0)
            if team not in (100, 200):
                team = _team_of_pid(
                    int(ev.get("killerId", 0) or ev.get("participantId", 0) or 0),
                    tm,
                )
            if team not in (100, 200):
                continue

            mt = str(ev.get("monsterType", "")).upper()
            key = "other"
            if mt == "DRAGON":
                key = "dragon"
            elif mt == "BARON_NASHOR":
                key = "baron"
            elif mt == "RIFTHERALD":
                key = "herald"
            elif mt == "ATAKHAN":
                key = "atakhan"
            elif mt == "HORDE":
                key = "horde"

            side = "blue" if team == 100 else "red"
            obj_by_type[key][side] += 1

        elif et == "BUILDING_KILL":
            bt = str(ev.get("buildingType", "")).upper()
            victim_team = int(ev.get("teamId", 0) or 0)
            if victim_team in (100, 200):
                taker_team = 100 if victim_team == 200 else 200
            else:
                taker_team = _team_of_pid(
                    int(ev.get("killerId", 0) or ev.get("participantId", 0) or 0),
                    tm,
                )
            if taker_team not in (100, 200):
                continue

            if "TOWER" in bt:
                if taker_team == 100:
                    tower_blue += 1
                else:
                    tower_red += 1
            elif "INHIBITOR" in bt:
                if taker_team == 100:
                    inhib_blue += 1
                else:
                    inhib_red += 1

        elif et == "TURRET_PLATE_DESTROYED":
            victim_team = int(ev.get("teamId", 0) or 0)
            if victim_team in (100, 200):
                taker_team = 100 if victim_team == 200 else 200
            else:
                taker_team = _team_of_pid(
                    int(ev.get("killerId", 0) or ev.get("participantId", 0) or 0),
                    tm,
                )
            if taker_team == 100:
                plate_blue += 1
            elif taker_team == 200:
                plate_red += 1

    obj_blue = int(sum(int(v["blue"]) for v in obj_by_type.values()))
    obj_red = int(sum(int(v["red"]) for v in obj_by_type.values()))

    return {
        "tower_blue": int(tower_blue),
        "tower_red": int(tower_red),
        "tower_diff": int(tower_blue - tower_red),
        "plate_blue": int(plate_blue),
        "plate_red": int(plate_red),
        "plate_diff": int(plate_blue - plate_red),
        "inhib_blue": int(inhib_blue),
        "inhib_red": int(inhib_red),
        "inhib_diff": int(inhib_blue - inhib_red),
        "objective_blue": int(obj_blue),
        "objective_red": int(obj_red),
        "objective_diff": int(obj_blue - obj_red),
        "objective_by_type": {
            k: {
                "blue": int(v["blue"]),
                "red": int(v["red"]),
                "diff": int(int(v["blue"]) - int(v["red"])),
            }
            for k, v in obj_by_type.items()
        },
    }


def compute_fight_outcome(
    fight: dict,
    kill_events: List[dict],
    tm: Dict[int, int],
    cache: Optional[Dict[str, Any]] = None,
    events: Optional[List[dict]] = None,
) -> FightOutcome:
    """Compute outcome on the fight label window.

    For continuous merged fights, uses [engage_ts, horizon_end_ts) when available.
    """
    engage_ts = int(fight["engage_ts"])
    horizon_end = _label_end_ts(fight, _get_horizon_ms())

    # Keep pipeline semantics: half-open interval [start, end)
    kills_in_fight = [k for k in kill_events if engage_ts <= int(k["timestamp"]) < horizon_end]

    blue_kills = red_kills = blue_deaths = red_deaths = blue_assists = red_assists = 0
    blue_dead_unique: set = set()
    red_dead_unique: set = set()

    for kill in kills_in_fight:
        killer_team = tm.get(kill.get("killer_id", 0), 0)
        victim_team = tm.get(kill.get("victim_id", 0), 0)

        if killer_team == 100:
            blue_kills += 1
        elif killer_team == 200:
            red_kills += 1

        if victim_team == 100:
            blue_deaths += 1
            blue_dead_unique.add(int(kill.get("victim_id", 0) or 0))
        elif victim_team == 200:
            red_deaths += 1
            red_dead_unique.add(int(kill.get("victim_id", 0) or 0))

        for assist_id in kill.get("assisting_ids", []) or []:
            assist_team = tm.get(assist_id, 0)
            if assist_team == 100:
                blue_assists += 1
            elif assist_team == 200:
                red_assists += 1

    kill_diff = blue_kills - red_kills
    winner = "blue" if kill_diff > 0 else ("red" if kill_diff < 0 else "draw")

    blue_survivors = int(max(0, 5 - len(blue_dead_unique)))
    red_survivors = int(max(0, 5 - len(red_dead_unique)))
    blue_alive_end = int(blue_survivors)
    red_alive_end = int(red_survivors)

    try:
        if isinstance(cache, dict):
            minute_ts = np.asarray(cache.get("minute_ts", []), dtype=np.int64)
            nm = cache.get("node_minute", None)
            alive_idx = NODE_IDX.get("alive", None)
            if (
                alive_idx is not None
                and isinstance(nm, np.ndarray)
                and nm.ndim == 3
                and minute_ts.ndim == 1
                and minute_ts.size > 0
                and nm.shape[0] == minute_ts.size
                and nm.shape[1] >= 10
                and int(alive_idx) < nm.shape[2]
            ):
                m_idx = _map_ts_to_minute_idx(minute_ts, max(0, int(horizon_end) - 1))
                alive_vec = nm[m_idx, :, int(alive_idx)].astype(np.float32)
                b_alive = 0.0
                r_alive = 0.0
                for pid in range(1, 11):
                    tid = _team_of_pid(pid, tm)
                    if tid == 100:
                        b_alive += float(alive_vec[pid - 1])
                    elif tid == 200:
                        r_alive += float(alive_vec[pid - 1])
                blue_alive_end = int(np.clip(round(b_alive), 0, 5))
                red_alive_end = int(np.clip(round(r_alive), 0, 5))
                blue_survivors = int(blue_alive_end)
                red_survivors = int(red_alive_end)
    except Exception:
        pass

    gold_blue_delta = 0.0
    gold_red_delta = 0.0
    gold_diff = 0.0
    g0 = _gold_team_at_ms(cache, int(engage_ts))
    g1 = _gold_team_at_ms(cache, max(int(engage_ts), int(horizon_end) - 1))
    if g0 is not None and g1 is not None:
        gold_blue_delta = float(g1[0] - g0[0])
        gold_red_delta = float(g1[1] - g0[1])
        gold_diff = float(gold_blue_delta - gold_red_delta)

    res = _window_resource_changes(events or [], tm, int(engage_ts), int(horizon_end))

    return FightOutcome(
        winner=winner,
        blue_kills=blue_kills,
        red_kills=red_kills,
        blue_deaths=blue_deaths,
        red_deaths=red_deaths,
        kill_diff=kill_diff,
        total_kills=len(kills_in_fight),
        assists={"blue": blue_assists, "red": red_assists},
        blue_unique_deaths=int(len(blue_dead_unique)),
        red_unique_deaths=int(len(red_dead_unique)),
        blue_survivors=int(blue_survivors),
        red_survivors=int(red_survivors),
        blue_alive_end=int(blue_alive_end),
        red_alive_end=int(red_alive_end),
        gold_blue_delta=float(gold_blue_delta),
        gold_red_delta=float(gold_red_delta),
        gold_diff=float(gold_diff),
        tower_blue=int(res.get("tower_blue", 0)),
        tower_red=int(res.get("tower_red", 0)),
        tower_diff=int(res.get("tower_diff", 0)),
        plate_blue=int(res.get("plate_blue", 0)),
        plate_red=int(res.get("plate_red", 0)),
        plate_diff=int(res.get("plate_diff", 0)),
        inhib_blue=int(res.get("inhib_blue", 0)),
        inhib_red=int(res.get("inhib_red", 0)),
        inhib_diff=int(res.get("inhib_diff", 0)),
        objective_blue=int(res.get("objective_blue", 0)),
        objective_red=int(res.get("objective_red", 0)),
        objective_diff=int(res.get("objective_diff", 0)),
        objective_by_type=res.get("objective_by_type", {}),
    )


def compute_player_engagement(
    fight: dict,
    xy_dense: np.ndarray,
    dists: np.ndarray,
    dense_ts: np.ndarray,
    R: float,
    b: np.ndarray,
    r: np.ndarray,
) -> List[PlayerEngagement]:
    """Compute engagement on the same label window used by training."""
    engage_ts = int(fight["engage_ts"])
    horizon_end = _label_end_ts(fight, _get_horizon_ms())

    Td = len(dense_ts)
    start_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts, side="left"), 0, Td - 1))
    end_idx = int(np.clip(np.searchsorted(dense_ts, horizon_end, side="left"), start_idx + 1, Td))

    end_idx = min(end_idx, len(dists))
    n_frames = max(1, end_idx - start_idx)

    dist_slice = dists[start_idx:end_idx]  # (n_frames, 5, 5)

    min_dist_b = np.min(dist_slice, axis=2)  # (n_frames, 5)
    frames_engaged_b = np.sum(min_dist_b <= R, axis=0)
    avg_dist_b = np.mean(min_dist_b, axis=0)

    min_dist_r = np.min(dist_slice, axis=1)  # (n_frames, 5)
    frames_engaged_r = np.sum(min_dist_r <= R, axis=0)
    avg_dist_r = np.mean(min_dist_r, axis=0)

    result: List[PlayerEngagement] = []
    for i, player_idx in enumerate(b):
        result.append(
            PlayerEngagement(
                player_idx=int(player_idx),
                team="blue",
                engagement_ratio=float(frames_engaged_b[i]) / n_frames,
                avg_dist_to_enemy=float(avg_dist_b[i]),
                frames_engaged=int(frames_engaged_b[i]),
                total_frames=n_frames,
            )
        )
    for j, player_idx in enumerate(r):
        result.append(
            PlayerEngagement(
                player_idx=int(player_idx),
                team="red",
                engagement_ratio=float(frames_engaged_r[j]) / n_frames,
                avg_dist_to_enemy=float(avg_dist_r[j]),
                frames_engaged=int(frames_engaged_r[j]),
                total_frames=n_frames,
            )
        )
    return result


def compute_fight_importance(fight: dict, outcome: FightOutcome, fight_type: str, game_duration_ms: int) -> float:
    score = 0.0
    score += min(int(outcome.get("total_kills", 0) or 0) * 6, 30)

    if game_duration_ms > 0:
        progress = float(int(fight["engage_ts"])) / float(game_duration_ms)
        if progress > 0.7:
            score += 20
        elif progress > 0.5:
            score += 10

    type_scores = {
        FightType.OBJECTIVE_BARON.value: 25,
        FightType.BASE_FIGHT.value: 25,
        FightType.OBJECTIVE_DRAGON.value: 15,
        FightType.OBJECTIVE_RIFTHERALD.value: 12,
        FightType.TOWER_DIVE.value: 8,
        FightType.TEAMFIGHT.value: 10,
        FightType.SKIRMISH.value: 5,
        FightType.PICK.value: 3,
    }
    score += type_scores.get(fight_type, 0)

    score += min(abs(int(outcome.get("kill_diff", 0) or 0)) * 5, 25)
    return float(min(score, 100.0))


def generate_fight_visualization(
    fight: dict,
    xy_dense: np.ndarray,
    dists: np.ndarray,
    dense_ts: np.ndarray,
    prox_pairs: np.ndarray,
    kill_events: List[dict],
    b: np.ndarray,
    r: np.ndarray,
    R: float,
    sample_interval: int = 5,
) -> FightVisualization:
    """Build visualization on the same label window used by training."""
    engage_ts = int(fight["engage_ts"])
    horizon_end = _label_end_ts(fight, _get_horizon_ms())

    Td = len(dense_ts)
    start_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts, side="left"), 0, Td - 1))
    end_idx = int(np.clip(np.searchsorted(dense_ts, horizon_end, side="left"), start_idx + 1, Td))

    trajectory: List[dict] = []
    for d_idx in range(start_idx, end_idx, sample_interval):
        if d_idx >= Td:
            break
        positions: Dict[str, Any] = {}
        for i, player_idx in enumerate(b):
            positions[f"blue_{i}"] = {"x": float(xy_dense[d_idx, player_idx, 0]), "y": float(xy_dense[d_idx, player_idx, 1])}
        for j, player_idx in enumerate(r):
            positions[f"red_{j}"] = {"x": float(xy_dense[d_idx, player_idx, 0]), "y": float(xy_dense[d_idx, player_idx, 1])}
        trajectory.append(
            {"timestamp": int(dense_ts[d_idx]), "positions": positions, "prox_pairs": int(prox_pairs[d_idx]) if d_idx < len(prox_pairs) else 0}
        )

    heatmap_points = [
        (float(xy_dense[d_idx, p, 0]), float(xy_dense[d_idx, p, 1]))
        for d_idx in range(start_idx, min(end_idx, Td))
        for p in range(10)
    ]

    engagement_timeline = [
        {
            "timestamp": int(dense_ts[d_idx]),
            "intensity": int(prox_pairs[d_idx]) if d_idx < len(prox_pairs) else 0,
            "normalized": float(prox_pairs[d_idx]) / 25.0 if d_idx < len(prox_pairs) else 0.0,
        }
        for d_idx in range(start_idx, min(end_idx, Td), sample_interval)
    ]

    kill_markers = [
        {
            "timestamp": int(kill["timestamp"]),
            "killer_id": kill.get("killer_id"),
            "victim_id": kill.get("victim_id"),
            "position": {"x": kill["position"][0], "y": kill["position"][1]} if kill.get("position") else None,
        }
        for kill in kill_events
        if engage_ts <= int(kill["timestamp"]) < horizon_end
    ]

    return FightVisualization(
        trajectory=trajectory,
        heatmap_points=heatmap_points,
        engagement_timeline=engagement_timeline,
        kill_markers=kill_markers,
    )


# ============================================================================
# 메인 감지 함수
# ============================================================================


def detect_fights_engage_v2(cache: Dict[str, Any], tm: Dict[int, int], config: Optional[FightDetectorConfig] = None) -> List[dict]:
    """교전 감지 메인 함수"""
    diag: Dict[str, Any] = {
        "Td": 0,
        "step_ms": 0,
        "candidates": 0,
        "accepted": 0,
        "rejected_gap": 0,
        "rejected_horizon": 0,
        "rejected_startctx": 0,
        "rejected_nokill": 0,
        "rejected_nosignal": 0,
        "rejected_noward_signal": 0,
        "rejected_noward_spatial": 0,
        "rejected_nocombat_signal": 0,
        "rejected_alive": 0,
        "rejected_engaged": 0,
        "rejected_lcc": 0,
        "rejected_compact": 0,
        "rejected_max_duration": 0,
        "rejected_negative_gap": 0,
        "accepted_by_anchor": 0,
        "accepted_by_signal": 0,
        "backtracked": 0,
        "backtrack_unreliable": 0,
        "continuous_merged": 0,
        "continuous_clusters": 0,
        "continuous_different_location": 0,
        "ace_events": 0,
        "ace_end_truncated": 0,
        "start_shifted_before_kill": 0,
        # [P0 FIX] post-merge enforcement stats
        "postmerge_conflicts": 0,
        "postmerge_removed": 0,
        "postmerge_replaced": 0,
        "errors": [],
    }

    # 설정 로드
    if config is None:
        try:
            config = FightDetectorConfig.from_cfg(cfg)
        except Exception as e:
            logger.warning(
                f"Config load failed: {e}. Using FightDetectorConfig defaults (aligned with CFG defaults)."
            )
            config = FightDetectorConfig()

    horizon_ms = int(_get_horizon_ms())

    # 팀 매핑 검증
    try:
        b, r = validate_team_mapping(tm)
    except Exception as e:
        logger.error(f"Team mapping error: {e}")
        diag["errors"].append({"type": "team_mapping", "message": str(e)})
        b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

    candidates_out: List[dict] = []  # pre-merge validated candidates
    fights: List[dict] = []  # post-merge fights

    def _compact_fight_result(f: dict) -> dict:
        out: Dict[str, Any] = {}
        try:
            out["engage_ts"] = int(f.get("engage_ts", -1))
            out["label_end_ts"] = int(_label_end_ts(f, horizon_ms))
            out["t_engage"] = int(f.get("t_engage", -1))
            out["fight_type"] = str(f.get("fight_type", "unknown"))
            out["importance_score"] = float(f.get("importance_score", 0.0) or 0.0)
            out["n_segments"] = int(f.get("n_segments", 1) or 1)
            out["det_anchor"] = int(f.get("det_anchor", 0) or 0)
            out["det_backtracked"] = int(f.get("det_backtracked", 0) or 0)
            out["first_kill_ts"] = int(f.get("first_kill_ts", -1) or -1)
            outcome = f.get("outcome", {}) if isinstance(f.get("outcome", {}), dict) else {}
            out["winner"] = str(outcome.get("winner", "unknown"))
            out["kill_diff"] = int(outcome.get("kill_diff", 0) or 0)
            out["total_kills"] = int(outcome.get("total_kills", 0) or 0)
            out["blue_deaths"] = int(outcome.get("blue_deaths", 0) or 0)
            out["red_deaths"] = int(outcome.get("red_deaths", 0) or 0)
            out["blue_survivors"] = int(outcome.get("blue_survivors", 0) or 0)
            out["red_survivors"] = int(outcome.get("red_survivors", 0) or 0)
            out["gold_diff"] = float(outcome.get("gold_diff", 0.0) or 0.0)
            out["gold_blue_delta"] = float(outcome.get("gold_blue_delta", 0.0) or 0.0)
            out["gold_red_delta"] = float(outcome.get("gold_red_delta", 0.0) or 0.0)
            out["tower_diff"] = int(outcome.get("tower_diff", 0) or 0)
            out["tower_blue"] = int(outcome.get("tower_blue", 0) or 0)
            out["tower_red"] = int(outcome.get("tower_red", 0) or 0)
            out["plate_diff"] = int(outcome.get("plate_diff", 0) or 0)
            out["objective_diff"] = int(outcome.get("objective_diff", 0) or 0)
            out["objective_blue"] = int(outcome.get("objective_blue", 0) or 0)
            out["objective_red"] = int(outcome.get("objective_red", 0) or 0)
        except Exception:
            pass
        return out

    def _pack_diagnostics():
        """fights 리스트가 완전히 채워진 뒤에만 호출."""
        try:
            diag["fight_summary"] = summarize_fights(fights)
            diag["fight_type_change_summary"] = summarize_fight_type_changes(fights)
            max_n = int(getattr(cfg, "DIAG_MAX_FIGHT_RESULTS", 50) or 50) if cfg else 50
            max_n = max(0, max_n)
            diag["fight_results_total"] = len(fights)
            diag["fight_results_truncated"] = int(len(fights) > max_n) if max_n > 0 else int(len(fights) > 0)
            diag["fight_results_brief"] = [_compact_fight_result(f) for f in (fights[:max_n] if max_n > 0 else [])]
            max_valid_n = int(getattr(cfg, "DIAG_MAX_VALIDATED_FIGHT_RESULTS", 10000) or 10000) if cfg else 10000
            max_valid_n = max(0, max_valid_n)
            diag["fight_results_validated_total"] = len(fights)
            diag["fight_results_validated_truncated"] = (
                int(len(fights) > max_valid_n) if max_valid_n > 0 else int(len(fights) > 0)
            )
            diag["fight_results_validated_brief"] = [
                _compact_fight_result(f) for f in (fights[:max_valid_n] if max_valid_n > 0 else [])
            ]
        except Exception as e:
            diag["errors"].append({"type": "diag_pack", "message": str(e)})

    # 위치 데이터 추출
    xy = cache.get("xy_raw_minute", None)
    if xy is None:
        xi = NODE_IDX.get("x_norm", 0)
        yi = NODE_IDX.get("y_norm", 1)
        try:
            xy = cache["node_minute"][:, :, [xi, yi]]
        except (KeyError, IndexError) as e:
            logger.error(f"Position data error: {e}")
            _pack_diagnostics()
            cache["fight_detect_diag"] = diag
            return fights

    minute_ts = np.asarray(cache["minute_ts"], dtype=np.int64)
    events = cache.get("events", [])
    Tm = int(len(minute_ts))

    if Tm < 3:
        logger.warning(f"Insufficient frames: {Tm}")
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    # 좌표 스케일 감지
    is_norm = bool(cache.get("meta", {}).get("anchor_is_norm", False))
    scale_factor = float(config.coord_norm_div)
    if not is_norm:
        is_norm, scale_factor = detect_coordinate_scale(xy)

    # 스케일 적용
    R = float(config.standoff_radius)
    engage_drop = float(config.engage_min_dist_drop)
    cluster_max_diam = float(config.cluster_max_diameter)
    continuous_merge_radius = float(config.continuous_fight_merge_radius)

    if is_norm:
        R /= scale_factor
        engage_drop /= scale_factor
        if cluster_max_diam > 0:
            cluster_max_diam /= scale_factor
        continuous_merge_radius /= scale_factor

    # Dense 보간
    step_ms = int(config.detect_step_ms)
    if step_ms < int(config.frame_ms) and str(config.interp_method).lower() != "none":
        dense_ts, xy_dense = _interp_xy_dense(xy, minute_ts, step_ms, config.interp_method)
    else:
        dense_ts = minute_ts.copy()
        xy_dense = xy.astype(np.float32, copy=False)
        step_ms = int(config.frame_ms)

    Td = int(len(dense_ts))
    if Td < 3:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    diag["Td"] = Td
    diag["step_ms"] = step_ms

    # 거리 계산
    dists = compute_distances_chunked(xy_dense, b, r, config.chunk_size)

    # Proximity metrics
    prox_pairs = np.sum(dists <= R, axis=(1, 2)).astype(np.int32)
    min_b = np.min(dists, axis=2)
    min_r = np.min(dists, axis=1)
    min_dist_mean = np.mean(np.concatenate([min_b, min_r], axis=1), axis=1).astype(np.float32)

    # 후보 탐지
    standoff = prox_pairs >= int(config.standoff_min_pairs)

    # engage_drop 시간 스케일링 + floor
    step_scale = float(step_ms) / 60000.0
    engage_drop_step = float(engage_drop) * step_scale
    drop_floor = float(config.engage_drop_floor)
    if is_norm:
        drop_floor /= scale_factor
    engage_drop_step = max(engage_drop_step, drop_floor)

    approach = (min_dist_mean[:-1] - min_dist_mean[1:]) >= engage_drop_step
    pair_gain = (prox_pairs[1:] - prox_pairs[:-1]) >= int(config.engage_min_pair_gain)
    cand = (standoff[1:] & standoff[:-1]) & (approach | pair_gain)
    cand_idx = np.where(cand)[0] + 1
    diag["candidates"] = int(cand_idx.size)

    # 킬 이벤트 추출
    kill_events = _extract_kill_events(events)
    kill_ts = (
        np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
        if kill_events
        else np.empty((0,), dtype=np.int64)
    )
    ace_ts = _extract_ace_ts(events)
    summ_spell_ts = _extract_summoner_spell_ts(events)
    dmg_idx = NODE_IDX.get("ds_totalDamageDoneToChampions", None)

    # 앵커 생성
    anchors = build_anchors_from_events(events)

    # 내부 변수
    t_min_ms = int(minute_ts[0])
    t_max_ms = int(minute_ts[-1])
    ctx_ms = int(config.fight_context_min) * 60000
    alive_idx = NODE_IDX.get("alive", None)
    diag["validation_rule"] = str(config.fight_validation_rule)
    diag["min_damage_norm_in_horizon"] = float(config.min_damage_norm_in_horizon)
    diag["min_summoner_spells_in_horizon"] = int(config.min_summoner_spells_in_horizon)
    kill_pre_ms = int(getattr(cfg, "EVENT_KILL_PRE_MS", 10000)) if cfg is not None else 10000
    late_kill_guard_ms = int(max(2 * int(step_ms), int(getattr(cfg, "EVENT_BURST_WINDOW_MS", 15000)))) if cfg is not None else int(max(2 * int(step_ms), 15000))
    diag["late_kill_guard_ms"] = int(late_kill_guard_ms)

    def _damage_norm_signal(engage_ts_val: int, horizon_end_ts: int) -> float:
        if dmg_idx is None:
            return 0.0
        try:
            i0 = _map_ts_to_minute_idx(minute_ts, int(engage_ts_val))
            i1 = _map_ts_to_minute_idx(minute_ts, int(horizon_end_ts))
            if i1 <= i0:
                return 0.0
            nm = cache.get("node_minute", None)
            if not isinstance(nm, np.ndarray) or nm.ndim != 3:
                return 0.0
            d0 = nm[i0, :, int(dmg_idx)].astype(np.float32)
            d1 = nm[i1, :, int(dmg_idx)].astype(np.float32)
            dd = np.maximum(d1 - d0, 0.0)
            return float(dd.sum())
        except Exception:
            return 0.0

    def _passes_cluster_guards(engage_ts_val: int) -> bool:
        if (
            int(config.require_engaged_per_team) <= 0
            and int(config.require_lcc_total) <= 0
            and int(config.require_lcc_per_team) <= 0
            and cluster_max_diam <= 0
        ):
            return True

        d_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts_val, side="right") - 1, 0, Td - 1))
        dist_mat = dists[d_idx]

        if int(config.require_engaged_per_team) > 0:
            engaged_b = int(np.sum(np.min(dist_mat, axis=1) <= R))
            engaged_r = int(np.sum(np.min(dist_mat, axis=0) <= R))
            if engaged_b < int(config.require_engaged_per_team) or engaged_r < int(config.require_engaged_per_team):
                diag["rejected_engaged"] += 1
                return False

        if int(config.require_lcc_total) > 0 or int(config.require_lcc_per_team) > 0 or cluster_max_diam > 0:
            lcc_total, lcc_b, lcc_r, lcc_nodes = _largest_bipartite_cc(dist_mat, R)

            if int(config.require_lcc_total) > 0 and lcc_total < int(config.require_lcc_total):
                diag["rejected_lcc"] += 1
                return False
            if int(config.require_lcc_per_team) > 0 and (
                lcc_b < int(config.require_lcc_per_team) or lcc_r < int(config.require_lcc_per_team)
            ):
                diag["rejected_lcc"] += 1
                return False

            if cluster_max_diam > 0 and lcc_total > 1 and len(lcc_nodes) > 1:
                orig_ids = [int(b[u]) if u < 5 else int(r[u - 5]) for u in lcc_nodes]
                pts = xy_dense[d_idx, orig_ids, :]
                maxd = 0.0
                for i in range(len(orig_ids)):
                    for j in range(i + 1, len(orig_ids)):
                        dx = float(pts[i, 0] - pts[j, 0])
                        dy = float(pts[i, 1] - pts[j, 1])
                        maxd = max(maxd, math.sqrt(dx * dx + dy * dy))
                if maxd > cluster_max_diam:
                    diag["rejected_compact"] += 1
                    return False

        return True

    def _check_alive_at_ts(engage_ts_val: int) -> bool:
        if int(config.require_alive_per_team) <= 0 or alive_idx is None:
            return True
        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)
        try:
            nm_alive = cache["node_minute"][m_idx, :, alive_idx]
            if float(nm_alive[b].sum()) < int(config.require_alive_per_team):
                return False
            if float(nm_alive[r].sum()) < int(config.require_alive_per_team):
                return False
        except Exception:
            return True
        return True

    def _try_add_candidate(
        engage_ts_val: int,
        *,
        anchor: bool = False,
        backtracked: bool = False,
        backtrack_reliable: bool = True,
    ):
        # Optional engage backtracking:
        # keep engage-driven detection, but if a kill exists in the candidate horizon,
        # allow moving start point earlier based on proximity continuity.
        if bool(config.use_backtrack) and not backtracked:
            fk_for_backtrack = _first_kill_in_window(kill_ts, engage_ts_val, engage_ts_val + horizon_ms)
            if fk_for_backtrack is not None:
                bt_ts, bt_ok = backtrack_engage_ts(
                    dists=dists,
                    dense_ts=dense_ts,
                    prox_pairs=prox_pairs,
                    kill_ts=int(fk_for_backtrack),
                    R=float(R),
                    min_pairs=int(config.backtrack_min_pairs),
                    max_lookback_ms=int(config.backtrack_max_ms),
                    min_lookback_ms=int(config.backtrack_min_ms),
                )
                bt_ts = int(bt_ts)
                if bt_ts < int(engage_ts_val):
                    engage_ts_val = bt_ts
                    backtracked = True
                    backtrack_reliable = bool(bt_ok)

        shifted_start, shifted = _ensure_start_before_recent_kill(
            engage_ts_val=int(engage_ts_val),
            ref_ts_exclusive=int(engage_ts_val),
            kill_ts=kill_ts,
            t_min_ms=int(t_min_ms),
            pre_ms=int(kill_pre_ms),
            max_gap_ms=int(late_kill_guard_ms),
            earliest_signal_before=None,
        )
        if shifted:
            engage_ts_val = int(shifted_start)
            diag["start_shifted_before_kill"] = int(diag.get("start_shifted_before_kill", 0) or 0) + 1

        # context/horizon guard
        if engage_ts_val - ctx_ms < t_min_ms:
            diag["rejected_startctx"] += 1
            return
        if engage_ts_val + horizon_ms > t_max_ms:
            diag["rejected_horizon"] += 1
            return
        if not _check_alive_at_ts(engage_ts_val):
            diag["rejected_alive"] += 1
            return

        horizon_end_ts = int(engage_ts_val + horizon_ms)
        fk = _first_kill_in_window(kill_ts, engage_ts_val, horizon_end_ts)
        kill_ok = bool(fk is not None)
        if (not bool(config.verify_kill_in_horizon)) and str(config.fight_validation_rule).lower() == "kill_only":
            kill_ok = True

        damage_norm = _damage_norm_signal(int(engage_ts_val), int(horizon_end_ts))
        spell_cnt = int(_count_events_in_window(summ_spell_ts, int(engage_ts_val), int(horizon_end_ts) - 1))
        signal_ok = bool(
            (float(damage_norm) >= float(config.min_damage_norm_in_horizon))
            or (int(spell_cnt) >= int(config.min_summoner_spells_in_horizon))
        )

        rule = str(config.fight_validation_rule).lower()
        if rule == "kill_only":
            valid = kill_ok
        elif rule == "signal_only":
            valid = signal_ok
        elif rule == "kill_and_signal":
            valid = bool(kill_ok and signal_ok)
        else:  # default: kill_or_signal
            valid = bool(kill_ok or signal_ok)

        if not valid:
            if not kill_ok:
                diag["rejected_nokill"] += 1
            if not signal_ok:
                diag["rejected_nosignal"] += 1
            return

        if not _passes_cluster_guards(engage_ts_val):
            return

        curr_centroid = _compute_fight_centroid_from_dists(xy_dense, dists, dense_ts, engage_ts_val, R, b, r)

        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)
        d_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts_val, side="right") - 1, 0, Td - 1))

        candidates_out.append(
            {
                "engage_ts": int(engage_ts_val),
                "t_engage": int(m_idx),
                "t_engage_ts": int(engage_ts_val),
                "first_kill_ts": int(fk) if fk is not None else -1,
                "centroid_x": float(curr_centroid[0]),
                "centroid_y": float(curr_centroid[1]),
                "horizon_end_ts": int(engage_ts_val + horizon_ms),
                "n_segments": 1,
                "det_step_ms": int(step_ms),
                "det_prox_pairs": int(prox_pairs[d_idx]) if d_idx < len(prox_pairs) else 0,
                "det_min_dist_mean": float(min_dist_mean[d_idx]) if d_idx < len(min_dist_mean) else 0.0,
                "det_anchor": int(anchor),
                "det_backtracked": int(backtracked),
                "det_damage_norm": float(damage_norm),
                "det_summoner_spells": int(spell_cnt),
                "det_signal_ok": int(signal_ok),
            }
        )

        diag["accepted"] += 1
        if anchor:
            diag["accepted_by_anchor"] += 1
        if signal_ok and not kill_ok:
            diag["accepted_by_signal"] += 1
        if backtracked:
            diag["backtracked"] += 1
            if not backtrack_reliable:
                diag["backtrack_unreliable"] += 1

    # Phase 1: Engage 후보
    for d_idx in cand_idx.tolist():
        _try_add_candidate(int(dense_ts[d_idx]), anchor=False, backtracked=False)

    # Phase 2: Kill-anchor start-points are intentionally disabled.
    if bool(config.use_kill_anchor):
        logger.warning(
            "USE_KILL_ANCHOR is ignored by design: "
            "fight start-point must be engage-detected, not kill-anchored."
        )

    # 후보가 비어있으면 진단 후 종료
    if not candidates_out:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    # 후보 정렬(시간)
    candidates_out.sort(key=lambda f: int(f["engage_ts"]))

    # continuous 병합 (order-independent)
    if bool(config.continuous_fight_merge):
        fights = merge_fights_st_dbscan_unionfind(
            candidates_out,
            eps_t_ms=int(config.continuous_fight_max_gap_ms),
            eps_s=float(continuous_merge_radius),
            horizon_ms=horizon_ms,
            max_duration_ms=int(config.max_merged_fight_duration_ms),
            diag=diag,
        )
        fights.sort(key=lambda f: int(f["engage_ts"]))
        for i in range(1, len(fights)):
            if int(fights[i]["engage_ts"]) - int(fights[i - 1]["engage_ts"]) < int(config.fight_min_gap_ms):
                diag["continuous_different_location"] += 1
    else:
        fights = []
        last_ts = -10**18
        for f in candidates_out:
            if int(f["engage_ts"]) - int(last_ts) < int(config.fight_min_gap_ms):
                diag["rejected_gap"] += 1
                continue
            fights.append(f)
            last_ts = int(f["engage_ts"])

    # If ACE occurs in the label horizon, end fight at ACE timestamp.
    _truncate_fights_at_ace(
        fights,
        ace_ts,
        horizon_ms=int(horizon_ms),
        diag=diag,
    )

    # ------------------------------------------------------------------
    # [P0 FIX] continuous merge ON/OFF와 무관하게 최종 fights에 대해:
    #   - fight_min_gap_ms 강제
    #   - label horizon non-overlap 강제
    # ------------------------------------------------------------------
    fights = enforce_postmerge_spacing_and_nonoverlap(
        fights,
        horizon_ms=int(horizon_ms),
        fight_min_gap_ms=int(config.fight_min_gap_ms),
        kill_ts=kill_ts,
        location_radius=float(max(0.0, cluster_max_diam)),
        diag=diag,
    )

    # Phase 3: 분석 결과 추가
    game_duration_ms = int(minute_ts[-1]) if len(minute_ts) > 0 else 0

    for fight in fights:
        try:
            fight["fight_type"] = classify_fight_type(fight, anchors, is_norm, scale_factor)
            outcome = compute_fight_outcome(fight, kill_events, tm, cache=cache, events=events)
            fight["outcome"] = outcome
            fight["importance_score"] = compute_fight_importance(fight, outcome, fight["fight_type"], game_duration_ms)
            fight["player_engagement"] = compute_player_engagement(fight, xy_dense, dists, dense_ts, R, b, r)
            fight["visualization"] = generate_fight_visualization(
                fight, xy_dense, dists, dense_ts, prox_pairs, kill_events, b, r, R
            )
        except Exception as e:
            logger.warning(f"Analysis failed for fight at {fight.get('engage_ts')}: {e}")
            diag["errors"].append({"type": "analysis", "engage_ts": fight.get("engage_ts"), "message": str(e)})

    _pack_diagnostics()
    cache["fight_detect_diag"] = diag
    return fights


def detect_fights_event_v1(cache: Dict[str, Any], tm: Dict[int, int], config: Optional[FightDetectorConfig] = None) -> List[dict]:
    """Event-driven teamfight detector.

    Detects fight start-points from short-window event bursts,
    then validates candidates with kill-centric pre/post window:
      - ward signal
      - objective/building around kill
      - structural guards (engaged/LCC/compactness)
    and merges temporally to form continuous fights.
    """
    diag: Dict[str, Any] = {
        "Td": 0,
        "step_ms": 0,
        "candidates": 0,
        "accepted": 0,
        "rejected_gap": 0,
        "rejected_horizon": 0,
        "rejected_startctx": 0,
        "rejected_nokill": 0,
        "rejected_nosignal": 0,
        "rejected_noward_signal": 0,
        "rejected_noward_spatial": 0,
        "rejected_post_nokill": 0,
        "rejected_noobjbuild": 0,
        "rejected_nocombat_signal": 0,
        "rejected_alive": 0,
        "rejected_engaged": 0,
        "rejected_lcc": 0,
        "rejected_compact": 0,
        "rejected_max_duration": 0,
        "rejected_negative_gap": 0,
        "accepted_by_anchor": 0,
        "accepted_by_signal": 0,
        "backtracked": 0,
        "backtrack_unreliable": 0,
        "continuous_merged": 0,
        "continuous_clusters": 0,
        "continuous_different_location": 0,
        "ace_events": 0,
        "ace_end_truncated": 0,
        "start_shifted_before_kill": 0,
        "postmerge_conflicts": 0,
        "postmerge_removed": 0,
        "postmerge_replaced": 0,
        "errors": [],
    }

    if config is None:
        try:
            config = FightDetectorConfig.from_cfg(cfg)
        except Exception as e:
            logger.warning(
                f"Config load failed: {e}. Using FightDetectorConfig defaults (aligned with CFG defaults)."
            )
            config = FightDetectorConfig()

    horizon_ms = int(_get_horizon_ms())

    try:
        b, r = validate_team_mapping(tm)
    except Exception as e:
        logger.error(f"Team mapping error: {e}")
        diag["errors"].append({"type": "team_mapping", "message": str(e)})
        b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

    candidates_out: List[dict] = []
    fights: List[dict] = []

    def _compact_fight_result(f: dict) -> dict:
        out: Dict[str, Any] = {}
        try:
            out["engage_ts"] = int(f.get("engage_ts", -1))
            out["label_end_ts"] = int(_label_end_ts(f, horizon_ms))
            out["t_engage"] = int(f.get("t_engage", -1))
            out["fight_type"] = str(f.get("fight_type", "unknown"))
            out["importance_score"] = float(f.get("importance_score", 0.0) or 0.0)
            out["n_segments"] = int(f.get("n_segments", 1) or 1)
            out["first_kill_ts"] = int(f.get("first_kill_ts", -1) or -1)
            out["det_event_score"] = float(f.get("det_event_score", 0.0) or 0.0)
            out["det_event_count"] = int(f.get("det_event_count", 0) or 0)
            outcome = f.get("outcome", {}) if isinstance(f.get("outcome", {}), dict) else {}
            out["winner"] = str(outcome.get("winner", "unknown"))
            out["kill_diff"] = int(outcome.get("kill_diff", 0) or 0)
            out["total_kills"] = int(outcome.get("total_kills", 0) or 0)
            out["blue_deaths"] = int(outcome.get("blue_deaths", 0) or 0)
            out["red_deaths"] = int(outcome.get("red_deaths", 0) or 0)
            out["blue_survivors"] = int(outcome.get("blue_survivors", 0) or 0)
            out["red_survivors"] = int(outcome.get("red_survivors", 0) or 0)
            out["gold_diff"] = float(outcome.get("gold_diff", 0.0) or 0.0)
            out["gold_blue_delta"] = float(outcome.get("gold_blue_delta", 0.0) or 0.0)
            out["gold_red_delta"] = float(outcome.get("gold_red_delta", 0.0) or 0.0)
            out["tower_diff"] = int(outcome.get("tower_diff", 0) or 0)
            out["tower_blue"] = int(outcome.get("tower_blue", 0) or 0)
            out["tower_red"] = int(outcome.get("tower_red", 0) or 0)
            out["plate_diff"] = int(outcome.get("plate_diff", 0) or 0)
            out["objective_diff"] = int(outcome.get("objective_diff", 0) or 0)
            out["objective_blue"] = int(outcome.get("objective_blue", 0) or 0)
            out["objective_red"] = int(outcome.get("objective_red", 0) or 0)
        except Exception:
            pass
        return out

    def _pack_diagnostics():
        try:
            diag["fight_summary"] = summarize_fights(fights)
            diag["fight_type_change_summary"] = summarize_fight_type_changes(fights)
            max_n = int(getattr(cfg, "DIAG_MAX_FIGHT_RESULTS", 50) or 50) if cfg else 50
            max_n = max(0, max_n)
            diag["fight_results_total"] = len(fights)
            diag["fight_results_truncated"] = int(len(fights) > max_n) if max_n > 0 else int(len(fights) > 0)
            diag["fight_results_brief"] = [_compact_fight_result(f) for f in (fights[:max_n] if max_n > 0 else [])]
            max_valid_n = int(getattr(cfg, "DIAG_MAX_VALIDATED_FIGHT_RESULTS", 10000) or 10000) if cfg else 10000
            max_valid_n = max(0, max_valid_n)
            diag["fight_results_validated_total"] = len(fights)
            diag["fight_results_validated_truncated"] = (
                int(len(fights) > max_valid_n) if max_valid_n > 0 else int(len(fights) > 0)
            )
            diag["fight_results_validated_brief"] = [
                _compact_fight_result(f) for f in (fights[:max_valid_n] if max_valid_n > 0 else [])
            ]
        except Exception as e:
            diag["errors"].append({"type": "diag_pack", "message": str(e)})

    xy = cache.get("xy_raw_minute", None)
    if xy is None:
        xi = NODE_IDX.get("x_norm", 0)
        yi = NODE_IDX.get("y_norm", 1)
        try:
            xy = cache["node_minute"][:, :, [xi, yi]]
        except (KeyError, IndexError) as e:
            logger.error(f"Position data error: {e}")
            _pack_diagnostics()
            cache["fight_detect_diag"] = diag
            return fights

    minute_ts = np.asarray(cache["minute_ts"], dtype=np.int64)
    events = cache.get("events", [])
    Tm = int(len(minute_ts))
    if Tm < 3:
        logger.warning(f"Insufficient frames: {Tm}")
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    is_norm = bool(cache.get("meta", {}).get("anchor_is_norm", False))
    scale_factor = float(config.coord_norm_div)
    if not is_norm:
        is_norm, scale_factor = detect_coordinate_scale(xy)

    # Structural guard defaults aligned to project rule:
    # at least 2 per team in contact and compact component (<= 4000).
    R = float(config.standoff_radius)
    cluster_max_diam = float(config.cluster_max_diameter)
    if cluster_max_diam <= 0:
        cluster_max_diam = 4000.0
    continuous_merge_radius = float(config.continuous_fight_merge_radius)
    if is_norm:
        R /= scale_factor
        if cluster_max_diam > 0:
            cluster_max_diam /= scale_factor
        if continuous_merge_radius > 0:
            continuous_merge_radius /= scale_factor

    # event_v1 does not need dense XY interpolation for candidate generation.
    # Use minute grid directly for faster cache building.
    dense_ts = minute_ts.copy()
    xy_dense = xy.astype(np.float32, copy=False)
    step_ms = int(config.frame_ms)

    Td = int(len(dense_ts))
    if Td < 3:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    diag["Td"] = Td
    diag["step_ms"] = step_ms
    diag["detector"] = "event_v1"

    dists = compute_distances_chunked(xy_dense, b, r, config.chunk_size)
    prox_pairs = np.sum(dists <= R, axis=(1, 2)).astype(np.int32)
    min_b = np.min(dists, axis=2)
    min_r = np.min(dists, axis=1)
    min_dist_mean = np.mean(np.concatenate([min_b, min_r], axis=1), axis=1).astype(np.float32)

    kill_events = _extract_kill_events(events)
    kill_ts = (
        np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
        if kill_events
        else np.empty((0,), dtype=np.int64)
    )
    ace_ts = _extract_ace_ts(events)
    summ_spell_ts = _extract_summoner_spell_ts(events)
    objective_ts, building_ts = _extract_objective_building_ts(events)
    ward_kill_ts, ward_t100_ts, ward_t200_ts = _extract_ward_signal_ts(events, tm)
    ward_actor_events = _extract_ward_actor_events(events, tm)
    dmg_idx = NODE_IDX.get("ds_totalDamageDoneToChampions", None)

    damage_ts = np.empty((0,), dtype=np.int64)
    try:
        if dmg_idx is not None:
            nm = cache.get("node_minute", None)
            if isinstance(nm, np.ndarray) and nm.ndim == 3 and nm.shape[0] >= 2:
                d0 = nm[:-1, :, int(dmg_idx)].astype(np.float32)
                d1 = nm[1:, :, int(dmg_idx)].astype(np.float32)
                dd = np.maximum(d1 - d0, 0.0).sum(axis=1)
                dmg_thr = float(getattr(cfg, "MIN_DAMAGE_NORM_IN_HORIZON", 0.02)) if cfg is not None else 0.02
                idx = np.where(dd >= dmg_thr)[0] + 1
                if idx.size > 0:
                    damage_ts = minute_ts[idx].astype(np.int64, copy=False)
    except Exception:
        damage_ts = np.empty((0,), dtype=np.int64)

    if (
        kill_ts.size
        + summ_spell_ts.size
        + objective_ts.size
        + building_ts.size
        + damage_ts.size
        + ward_kill_ts.size
        + ward_t100_ts.size
        + ward_t200_ts.size
    ) == 0:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    all_ts = np.unique(
        np.concatenate(
            [
                x
                for x in (
                    kill_ts,
                    summ_spell_ts,
                    objective_ts,
                    building_ts,
                    damage_ts,
                    ward_kill_ts,
                    ward_t100_ts,
                    ward_t200_ts,
                )
                if isinstance(x, np.ndarray) and x.size > 0
            ]
        )
    )
    diag["candidates"] = int(all_ts.size)

    anchors = build_anchors_from_events(events)
    t_min_ms = int(minute_ts[0])
    t_max_ms = int(minute_ts[-1])
    ctx_ms = int(config.fight_context_min) * 60000
    alive_idx = NODE_IDX.get("alive", None)

    event_window_ms = int(getattr(cfg, "EVENT_BURST_WINDOW_MS", 15000)) if cfg is not None else 15000
    kill_pre_ms = int(getattr(cfg, "EVENT_KILL_PRE_MS", 10000)) if cfg is not None else 10000
    kill_post_ms = int(getattr(cfg, "EVENT_KILL_POST_MS", 10000)) if cfg is not None else 10000
    require_post_kill_validation = bool(
        getattr(cfg, "EVENT_REQUIRE_POST_KILL_VALIDATION", True)
    ) if cfg is not None else True
    post_validate_pre_ms = int(getattr(cfg, "EVENT_POST_VALIDATE_PRE_MS", 45000)) if cfg is not None else 45000
    post_validate_post_ms = int(getattr(cfg, "EVENT_POST_VALIDATE_POST_MS", 45000)) if cfg is not None else 45000
    event_min_events = int(getattr(cfg, "EVENT_MIN_EVENTS_IN_WINDOW", 2)) if cfg is not None else 2
    event_score_thr = float(getattr(cfg, "EVENT_SCORE_THRESHOLD", 2.5)) if cfg is not None else 2.5
    w_kill = float(getattr(cfg, "EVENT_WEIGHT_KILL", 2.0)) if cfg is not None else 2.0
    w_spell = float(getattr(cfg, "EVENT_WEIGHT_SPELL", 0.35)) if cfg is not None else 0.35
    w_obj = float(getattr(cfg, "EVENT_WEIGHT_OBJECTIVE", 1.5)) if cfg is not None else 1.5
    w_build = float(getattr(cfg, "EVENT_WEIGHT_BUILDING", 1.5)) if cfg is not None else 1.5
    w_dmg = float(getattr(cfg, "EVENT_WEIGHT_DAMAGE", 1.0)) if cfg is not None else 1.0
    dmg_min = float(getattr(cfg, "MIN_DAMAGE_NORM_IN_HORIZON", 0.02)) if cfg is not None else 0.02
    spell_min = int(getattr(cfg, "MIN_SUMMONER_SPELLS_IN_HORIZON", 1)) if cfg is not None else 1
    req_engaged = int(max(2, int(config.require_engaged_per_team)))
    req_lcc_total = int(max(4, int(config.require_lcc_total)))
    req_lcc_per_team = int(max(2, int(config.require_lcc_per_team)))
    require_ward_actor_spatial = bool(config.require_ward_actor_in_fight_radius)
    ward_actor_radius = float(config.ward_actor_radius if config.ward_actor_radius > 0 else config.standoff_radius)
    if is_norm and ward_actor_radius > 0:
        ward_actor_radius /= scale_factor
    kill_ts_set = set(int(x) for x in kill_ts.tolist()) if kill_ts.size > 0 else set()

    if ward_actor_events:
        ward_evt_ts = np.asarray([int(e["timestamp"]) for e in ward_actor_events], dtype=np.int64)
        ward_evt_pid = np.asarray([int(e["pid"]) for e in ward_actor_events], dtype=np.int32)
        ward_evt_team = np.asarray([int(e["team"]) for e in ward_actor_events], dtype=np.int32)
        ward_evt_is_kill = np.asarray([int(e["is_kill"]) for e in ward_actor_events], dtype=np.int8)
    else:
        ward_evt_ts = np.empty((0,), dtype=np.int64)
        ward_evt_pid = np.empty((0,), dtype=np.int32)
        ward_evt_team = np.empty((0,), dtype=np.int32)
        ward_evt_is_kill = np.empty((0,), dtype=np.int8)

    diag["event_window_ms"] = int(event_window_ms)
    diag["event_kill_pre_ms"] = int(kill_pre_ms)
    diag["event_kill_post_ms"] = int(kill_post_ms)
    diag["event_require_post_kill_validation"] = int(require_post_kill_validation)
    diag["event_post_validate_pre_ms"] = int(post_validate_pre_ms)
    diag["event_post_validate_post_ms"] = int(post_validate_post_ms)
    diag["event_min_events"] = int(event_min_events)
    diag["event_score_threshold"] = float(event_score_thr)

    def _damage_norm_prewindow(t0: int, t1: int) -> float:
        if dmg_idx is None:
            return 0.0
        try:
            i0 = _map_ts_to_minute_idx(minute_ts, int(t0))
            i1 = _map_ts_to_minute_idx(minute_ts, int(t1))
            if i1 <= i0:
                return 0.0
            nm = cache.get("node_minute", None)
            if not isinstance(nm, np.ndarray) or nm.ndim != 3:
                return 0.0
            d0 = nm[i0, :, int(dmg_idx)].astype(np.float32)
            d1 = nm[i1, :, int(dmg_idx)].astype(np.float32)
            dd = np.maximum(d1 - d0, 0.0)
            return float(dd.sum())
        except Exception:
            return 0.0

    def _check_alive_at_ts(engage_ts_val: int) -> bool:
        if int(config.require_alive_per_team) <= 0 or alive_idx is None:
            return True
        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)
        try:
            nm_alive = cache["node_minute"][m_idx, :, alive_idx]
            if float(nm_alive[b].sum()) < int(config.require_alive_per_team):
                return False
            if float(nm_alive[r].sum()) < int(config.require_alive_per_team):
                return False
        except Exception:
            return True
        return True

    def _earliest_signal_before(anchor_ts: int, lookback_ms: int) -> Optional[int]:
        t0 = int(max(t_min_ms, anchor_ts - max(0, int(lookback_ms))))
        if t0 >= anchor_ts:
            return None
        earliest: Optional[int] = None
        for arr in (
            kill_ts,
            summ_spell_ts,
            objective_ts,
            building_ts,
            damage_ts,
            ward_kill_ts,
            ward_t100_ts,
            ward_t200_ts,
        ):
            if arr.size == 0:
                continue
            i = int(np.searchsorted(arr, t0, side="left"))
            if i < arr.size:
                v = int(arr[i])
                if v < anchor_ts and (earliest is None or v < earliest):
                    earliest = v
        return earliest

    def _nearest_kill_around(ts: int, pre_ms: int, post_ms: int) -> Optional[int]:
        if kill_ts.size == 0:
            return None
        lo = int(ts) - max(0, int(pre_ms))
        hi = int(ts) + max(0, int(post_ms))
        l = int(np.searchsorted(kill_ts, lo, side="left"))
        r_ = int(np.searchsorted(kill_ts, hi, side="right"))
        if r_ <= l:
            return None
        seg = kill_ts[l:r_]
        if seg.size <= 0:
            return None
        j = int(np.argmin(np.abs(seg.astype(np.int64) - int(ts))))
        return int(seg[j])

    def _ward_signal_spatial_for_component(
        d_idx: int,
        t0: int,
        t1: int,
        comp_players: List[int],
        comp_cx: float,
        comp_cy: float,
    ) -> Tuple[bool, bool, bool, int]:
        """Validate ward actors are inside fight radius of the candidate component."""
        if ward_evt_ts.size == 0:
            return False, False, False, 0
        if ward_actor_radius <= 0:
            return False, False, False, 0
        l = int(np.searchsorted(ward_evt_ts, int(t0), side="left"))
        r_ = int(np.searchsorted(ward_evt_ts, int(t1), side="right"))
        if r_ <= l:
            return False, False, False, 0

        comp_set = set(int(x) for x in comp_players)
        has_kill = False
        blue_seen = False
        red_seen = False
        in_radius_cnt = 0

        for i in range(l, r_):
            pid = int(ward_evt_pid[i])
            p_idx = int(pid - 1)
            if p_idx < 0 or p_idx >= 10:
                continue
            if p_idx in comp_set:
                in_radius = True
            else:
                px = float(xy_dense[d_idx, p_idx, 0])
                py = float(xy_dense[d_idx, p_idx, 1])
                if not np.isfinite(px) or not np.isfinite(py):
                    continue
                dx = px - float(comp_cx)
                dy = py - float(comp_cy)
                in_radius = bool((dx * dx + dy * dy) <= (ward_actor_radius * ward_actor_radius))

            if not in_radius:
                continue

            in_radius_cnt += 1
            if int(ward_evt_is_kill[i]) == 1:
                has_kill = True
            t_id = int(ward_evt_team[i])
            if t_id == 100:
                blue_seen = True
            elif t_id == 200:
                red_seen = True

        both_teams = bool(blue_seen and red_seen)
        ward_ok = bool(has_kill or both_teams)
        return ward_ok, has_kill, both_teams, int(in_radius_cnt)

    for ts in all_ts.tolist():
        anchor_ts = int(ts)
        is_kill_anchor = bool(anchor_ts in kill_ts_set)
        pre_sig_ts = _earliest_signal_before(anchor_ts, event_window_ms)
        backtracked = False
        backtrack_reliable = True

        if pre_sig_ts is not None:
            engage_ts_val = int(pre_sig_ts)
        elif is_kill_anchor:
            engage_ts_val = int(max(t_min_ms, int(anchor_ts) - max(1, int(kill_pre_ms))))
        else:
            engage_ts_val = int(anchor_ts)

        if is_kill_anchor and engage_ts_val >= int(anchor_ts):
            engage_ts_val = int(max(t_min_ms, int(anchor_ts) - 1))

        if bool(config.use_backtrack):
            fk_for_backtrack = _first_kill_in_window(kill_ts, engage_ts_val, engage_ts_val + horizon_ms)
            if fk_for_backtrack is not None:
                bt_ts, bt_ok = backtrack_engage_ts(
                    dists=dists,
                    dense_ts=dense_ts,
                    prox_pairs=prox_pairs,
                    kill_ts=int(fk_for_backtrack),
                    R=float(R),
                    min_pairs=int(config.backtrack_min_pairs),
                    max_lookback_ms=int(config.backtrack_max_ms),
                    min_lookback_ms=int(config.backtrack_min_ms),
                )
                bt_ts = int(bt_ts)
                if bt_ts < int(engage_ts_val):
                    engage_ts_val = bt_ts
                    backtracked = True
                    backtrack_reliable = bool(bt_ok)

        shifted_start, shifted = _ensure_start_before_recent_kill(
            engage_ts_val=int(engage_ts_val),
            ref_ts_exclusive=int(anchor_ts),
            kill_ts=kill_ts,
            t_min_ms=int(t_min_ms),
            pre_ms=int(kill_pre_ms),
            max_gap_ms=int(max(event_window_ms, kill_pre_ms)),
            earliest_signal_before=_earliest_signal_before,
        )
        if shifted:
            engage_ts_val = int(shifted_start)
            diag["start_shifted_before_kill"] = int(diag.get("start_shifted_before_kill", 0) or 0) + 1

        if engage_ts_val - ctx_ms < t_min_ms:
            diag["rejected_startctx"] += 1
            continue
        if engage_ts_val + horizon_ms > t_max_ms:
            diag["rejected_horizon"] += 1
            continue
        if not _check_alive_at_ts(engage_ts_val):
            diag["rejected_alive"] += 1
            continue

        # Stage-1 (realtime): signal check on pre-window around anchor.
        rt_t0 = int(max(t_min_ms, int(anchor_ts) - event_window_ms))
        rt_t1 = int(anchor_ts)
        if rt_t1 < rt_t0:
            diag["rejected_nosignal"] += 1
            continue

        kill_cnt = int(_count_events_in_window(kill_ts, rt_t0, rt_t1))
        spell_cnt = int(_count_events_in_window(summ_spell_ts, rt_t0, rt_t1))
        obj_cnt = int(_count_events_in_window(objective_ts, rt_t0, rt_t1))
        build_cnt = int(_count_events_in_window(building_ts, rt_t0, rt_t1))
        ward_kill_cnt = int(_count_events_in_window(ward_kill_ts, rt_t0, rt_t1))
        ward_b_cnt = int(_count_events_in_window(ward_t100_ts, rt_t0, rt_t1))
        ward_r_cnt = int(_count_events_in_window(ward_t200_ts, rt_t0, rt_t1))
        ward_both_teams = bool((ward_b_cnt > 0) and (ward_r_cnt > 0))
        evt_cnt = int(kill_cnt + spell_cnt + obj_cnt + build_cnt + ward_kill_cnt)
        dmg_norm = float(_damage_norm_prewindow(rt_t0, rt_t1))

        score = float(
            w_kill * kill_cnt
            + w_spell * spell_cnt
            + w_obj * obj_cnt
            + w_build * build_cnt
            + w_dmg * dmg_norm
        )

        score_ok = bool(
            (score >= event_score_thr)
            and ((evt_cnt >= event_min_events) or (dmg_norm >= dmg_min) or (spell_cnt >= spell_min))
        )
        ward_signal_time_ok = bool((ward_kill_cnt >= 1) or ward_both_teams)
        combat_signal_ok = bool((obj_cnt + build_cnt) >= 1)
        if not combat_signal_ok:
            diag["rejected_noobjbuild"] += 1
            diag["rejected_nocombat_signal"] += 1
            diag["rejected_nosignal"] += 1
            continue
        if not ward_signal_time_ok:
            diag["rejected_noward_signal"] += 1
            diag["rejected_nosignal"] += 1
            continue

        # Stage-2 (post validation): nearest kill around anchor and obj/build near that kill.
        kill_ref_ts = _nearest_kill_around(anchor_ts, post_validate_pre_ms, post_validate_post_ms)
        post_obj_cnt = 0
        post_build_cnt = 0
        if require_post_kill_validation:
            if kill_ref_ts is None:
                diag["rejected_post_nokill"] += 1
                diag["rejected_nokill"] += 1
                diag["rejected_nosignal"] += 1
                continue
            post_t0 = int(max(t_min_ms, int(kill_ref_ts) - post_validate_pre_ms))
            post_t1 = int(min(t_max_ms, int(kill_ref_ts) + post_validate_post_ms))
            if post_t1 < post_t0:
                diag["rejected_nosignal"] += 1
                continue
            post_obj_cnt = int(_count_events_in_window(objective_ts, post_t0, post_t1))
            post_build_cnt = int(_count_events_in_window(building_ts, post_t0, post_t1))
            if (post_obj_cnt + post_build_cnt) <= 0:
                diag["rejected_noobjbuild"] += 1
                diag["rejected_nocombat_signal"] += 1
                diag["rejected_nosignal"] += 1
                continue

        d_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts_val, side="right") - 1, 0, Td - 1))
        dist_mat = dists[d_idx]
        comps = _bipartite_components(dist_mat, R)
        if len(comps) == 0:
            diag["rejected_engaged"] += 1
            continue

        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)
        fk = _first_kill_in_window(kill_ts, engage_ts_val, engage_ts_val + horizon_ms)

        accepted_components = 0
        rejected_by_ward_spatial = 0
        for ci, (nodes, b_cnt, r_cnt) in enumerate(comps):
            comp_total = int(len(nodes))
            if req_engaged > 0 and (int(b_cnt) < req_engaged or int(r_cnt) < req_engaged):
                continue
            if req_lcc_total > 0 and comp_total < req_lcc_total:
                continue
            if req_lcc_per_team > 0 and (int(b_cnt) < req_lcc_per_team or int(r_cnt) < req_lcc_per_team):
                continue

            orig_ids = [int(b[u]) if u < 5 else int(r[u - 5]) for u in nodes]
            pts = xy_dense[d_idx, orig_ids, :]
            if cluster_max_diam > 0 and len(orig_ids) > 1:
                maxd = 0.0
                for i0 in range(len(orig_ids)):
                    for j0 in range(i0 + 1, len(orig_ids)):
                        dx = float(pts[i0, 0] - pts[j0, 0])
                        dy = float(pts[i0, 1] - pts[j0, 1])
                        maxd = max(maxd, math.sqrt(dx * dx + dy * dy))
                if maxd > cluster_max_diam:
                    continue

            prox_pairs_comp = 0
            for u in nodes:
                if u >= 5:
                    continue
                for v in nodes:
                    if v < 5:
                        continue
                    if dist_mat[u, v - 5] <= R:
                        prox_pairs_comp += 1
            if prox_pairs_comp <= 0:
                continue

            cx = float(np.mean(pts[:, 0])) if pts.size > 0 else float(np.mean(xy_dense[d_idx, :, 0]))
            cy = float(np.mean(pts[:, 1])) if pts.size > 0 else float(np.mean(xy_dense[d_idx, :, 1]))

            ward_signal_spatial_ok = bool(ward_signal_time_ok)
            ward_kill_spatial_ok = bool(ward_kill_cnt >= 1)
            ward_both_spatial_ok = bool(ward_both_teams)
            ward_actor_in_radius_cnt = -1
            if require_ward_actor_spatial:
                (
                    ward_signal_spatial_ok,
                    ward_kill_spatial_ok,
                    ward_both_spatial_ok,
                    ward_actor_in_radius_cnt,
                ) = _ward_signal_spatial_for_component(
                    d_idx=d_idx,
                    t0=rt_t0,
                    t1=rt_t1,
                    comp_players=orig_ids,
                    comp_cx=cx,
                    comp_cy=cy,
                )
            if not ward_signal_spatial_ok:
                rejected_by_ward_spatial += 1
                diag["rejected_noward_spatial"] += 1
                continue

            candidates_out.append(
                {
                    "engage_ts": int(engage_ts_val),
                    "t_engage": int(m_idx),
                    "t_engage_ts": int(engage_ts_val),
                    "first_kill_ts": int(fk) if fk is not None else -1,
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "horizon_end_ts": int(engage_ts_val + horizon_ms),
                    "n_segments": 1,
                    "det_step_ms": int(step_ms),
                    "det_prox_pairs": int(prox_pairs_comp),
                    "det_min_dist_mean": float(min_dist_mean[d_idx]) if d_idx < len(min_dist_mean) else 0.0,
                    "det_anchor": 0,
                    "det_backtracked": int(backtracked),
                    "det_backtrack_reliable": int(backtrack_reliable),
                    "det_damage_norm": float(dmg_norm),
                    "det_summoner_spells": int(spell_cnt),
                    "det_signal_ok": int(ward_signal_spatial_ok and combat_signal_ok),
                    "det_score_ok": int(score_ok),
                    "det_event_score": float(score),
                    "det_event_count": int(evt_cnt),
                    "det_kill_count_window": int(kill_cnt),
                    "det_objective_count_window": int(obj_cnt),
                    "det_building_count_window": int(build_cnt),
                    "det_post_objective_count_window": int(post_obj_cnt),
                    "det_post_building_count_window": int(post_build_cnt),
                    "det_ward_kill_count_window": int(ward_kill_cnt),
                    "det_ward_both_teams_window": int(ward_both_teams),
                    "det_ward_signal_time_ok": int(ward_signal_time_ok),
                    "det_ward_signal_spatial_ok": int(ward_signal_spatial_ok),
                    "det_ward_kill_spatial_ok": int(ward_kill_spatial_ok),
                    "det_ward_both_teams_spatial_ok": int(ward_both_spatial_ok),
                    "det_ward_actor_in_radius_count": int(max(0, ward_actor_in_radius_cnt)),
                    "det_ward_actor_radius": float(max(0.0, ward_actor_radius)),
                    "det_combat_signal_ok": int(combat_signal_ok),
                    "det_event_anchor_ts": int(anchor_ts),
                    "det_event_anchor_is_kill": int(is_kill_anchor),
                    "det_kill_ref_ts": int(kill_ref_ts) if kill_ref_ts is not None else -1,
                    "det_event_start_shift_ms": int(max(0, (int(kill_ref_ts) if kill_ref_ts is not None else int(anchor_ts)) - engage_ts_val)),
                    "det_component_idx": int(ci),
                    "det_component_size": int(comp_total),
                    "det_component_blue": int(b_cnt),
                    "det_component_red": int(r_cnt),
                }
            )
            accepted_components += 1

        if accepted_components <= 0:
            if rejected_by_ward_spatial > 0:
                diag["rejected_nosignal"] += 1
            else:
                diag["rejected_lcc"] += 1
            continue
        diag["accepted"] += int(accepted_components)
        diag["accepted_by_signal"] += int(accepted_components)
        if backtracked:
            diag["backtracked"] += int(accepted_components)
            if not backtrack_reliable:
                diag["backtrack_unreliable"] += int(accepted_components)

    if not candidates_out:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    candidates_out.sort(key=lambda f: int(f["engage_ts"]))
    if bool(config.continuous_fight_merge):
        merge_eps_s = float(cluster_max_diam if cluster_max_diam > 0 else continuous_merge_radius)
        if merge_eps_s > 0:
            fights = merge_fights_st_dbscan_unionfind(
                candidates_out,
                eps_t_ms=int(config.continuous_fight_max_gap_ms),
                eps_s=float(merge_eps_s),
                horizon_ms=horizon_ms,
                max_duration_ms=int(config.max_merged_fight_duration_ms),
                diag=diag,
            )
        else:
            fights = merge_fights_temporal_unionfind(
                candidates_out,
                max_gap_ms=int(config.continuous_fight_max_gap_ms),
                horizon_ms=horizon_ms,
                max_duration_ms=int(config.max_merged_fight_duration_ms),
                diag=diag,
            )
    else:
        fights = []
        last_ts = -10**18
        for f in candidates_out:
            if int(f["engage_ts"]) - int(last_ts) < int(config.fight_min_gap_ms):
                diag["rejected_gap"] += 1
                continue
            fights.append(f)
            last_ts = int(f["engage_ts"])

    # If ACE occurs in the label horizon, end fight at ACE timestamp.
    _truncate_fights_at_ace(
        fights,
        ace_ts,
        horizon_ms=int(horizon_ms),
        diag=diag,
    )

    fights = enforce_postmerge_spacing_and_nonoverlap(
        fights,
        horizon_ms=int(horizon_ms),
        fight_min_gap_ms=int(config.fight_min_gap_ms),
        kill_ts=kill_ts,
        location_radius=float(max(0.0, cluster_max_diam)),
        diag=diag,
    )

    game_duration_ms = int(minute_ts[-1]) if len(minute_ts) > 0 else 0
    for fight in fights:
        try:
            fight["fight_type"] = classify_fight_type(fight, anchors, is_norm, scale_factor)
            outcome = compute_fight_outcome(fight, kill_events, tm, cache=cache, events=events)
            fight["outcome"] = outcome
            fight["importance_score"] = compute_fight_importance(fight, outcome, fight["fight_type"], game_duration_ms)
            fight["player_engagement"] = compute_player_engagement(fight, xy_dense, dists, dense_ts, R, b, r)
            fight["visualization"] = generate_fight_visualization(
                fight, xy_dense, dists, dense_ts, prox_pairs, kill_events, b, r, R
            )
        except Exception as e:
            logger.warning(f"Analysis failed for fight at {fight.get('engage_ts')}: {e}")
            diag["errors"].append({"type": "analysis", "engage_ts": fight.get("engage_ts"), "message": str(e)})

    _pack_diagnostics()
    cache["fight_detect_diag"] = diag
    return fights


def detect_fights_killchain_v1(
    cache: Dict[str, Any],
    tm: Dict[int, int],
    config: Optional[FightDetectorConfig] = None,
) -> List[dict]:
    """Kill-chain teamfight detector.

    Detects fights by chaining CHAMPION_KILL events through participant
    overlap in victimDamageReceived/victimDamageDealt arrays.  Each chain
    becomes a fight candidate with engage_ts = first_kill - backtrack_ms.

    Key differences from event_v1:
      - Participants identified directly from ms-precision damage arrays,
        not from 60s-stale frame positions.
      - Fight location from kill-position centroids, not frame-position centroids.
      - No objective/building/ward gates -- objectives are used for
        classification only, not as detection requirements.
      - No bipartite-component structural check on frame distances.
    """
    diag: Dict[str, Any] = {
        "Td": 0,
        "step_ms": 0,
        "detector": "killchain_v1",
        "candidates": 0,
        "chains_total": 0,
        "chains_accepted": 0,
        "accepted": 0,
        "rejected_startctx": 0,
        "rejected_horizon": 0,
        "rejected_alive": 0,
        "rejected_too_few_per_team": 0,
        "rejected_gap": 0,
        "rejected_negative_gap": 0,
        "rejected_max_duration": 0,
        "backtracked": 0,
        "continuous_merged": 0,
        "continuous_clusters": 0,
        "ace_events": 0,
        "ace_end_truncated": 0,
        "postmerge_conflicts": 0,
        "postmerge_removed": 0,
        "postmerge_replaced": 0,
        "errors": [],
    }

    if config is None:
        try:
            config = FightDetectorConfig.from_cfg(cfg)
        except Exception as e:
            logger.warning(
                f"Config load failed: {e}. Using FightDetectorConfig defaults."
            )
            config = FightDetectorConfig()

    horizon_ms = int(_get_horizon_ms())

    try:
        b, r = validate_team_mapping(tm)
    except Exception as e:
        logger.error(f"Team mapping error: {e}")
        diag["errors"].append({"type": "team_mapping", "message": str(e)})
        b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

    candidates_out: List[dict] = []
    fights: List[dict] = []

    # --- compact result helper (same as event_v1) ---
    def _compact_fight_result(f: dict) -> dict:
        out: Dict[str, Any] = {}
        try:
            out["engage_ts"] = int(f.get("engage_ts", -1))
            out["label_end_ts"] = int(_label_end_ts(f, horizon_ms))
            out["t_engage"] = int(f.get("t_engage", -1))
            out["fight_type"] = str(f.get("fight_type", "unknown"))
            out["importance_score"] = float(f.get("importance_score", 0.0) or 0.0)
            out["n_segments"] = int(f.get("n_segments", 1) or 1)
            out["first_kill_ts"] = int(f.get("first_kill_ts", -1) or -1)
            out["det_event_score"] = float(f.get("det_event_score", 0.0) or 0.0)
            out["det_event_count"] = int(f.get("det_event_count", 0) or 0)
            outcome = f.get("outcome", {}) if isinstance(f.get("outcome", {}), dict) else {}
            out["winner"] = str(outcome.get("winner", "unknown"))
            out["kill_diff"] = int(outcome.get("kill_diff", 0) or 0)
            out["total_kills"] = int(outcome.get("total_kills", 0) or 0)
            out["blue_deaths"] = int(outcome.get("blue_deaths", 0) or 0)
            out["red_deaths"] = int(outcome.get("red_deaths", 0) or 0)
            out["blue_survivors"] = int(outcome.get("blue_survivors", 0) or 0)
            out["red_survivors"] = int(outcome.get("red_survivors", 0) or 0)
            out["gold_diff"] = float(outcome.get("gold_diff", 0.0) or 0.0)
            out["gold_blue_delta"] = float(outcome.get("gold_blue_delta", 0.0) or 0.0)
            out["gold_red_delta"] = float(outcome.get("gold_red_delta", 0.0) or 0.0)
            out["tower_diff"] = int(outcome.get("tower_diff", 0) or 0)
            out["tower_blue"] = int(outcome.get("tower_blue", 0) or 0)
            out["tower_red"] = int(outcome.get("tower_red", 0) or 0)
            out["plate_diff"] = int(outcome.get("plate_diff", 0) or 0)
            out["objective_diff"] = int(outcome.get("objective_diff", 0) or 0)
            out["objective_blue"] = int(outcome.get("objective_blue", 0) or 0)
            out["objective_red"] = int(outcome.get("objective_red", 0) or 0)
        except Exception:
            pass
        return out

    def _pack_diagnostics():
        try:
            diag["fight_summary"] = summarize_fights(fights)
            diag["fight_type_change_summary"] = summarize_fight_type_changes(fights)
            max_n = int(getattr(cfg, "DIAG_MAX_FIGHT_RESULTS", 50) or 50) if cfg else 50
            max_n = max(0, max_n)
            diag["fight_results_total"] = len(fights)
            diag["fight_results_truncated"] = int(len(fights) > max_n) if max_n > 0 else int(len(fights) > 0)
            diag["fight_results_brief"] = [_compact_fight_result(f) for f in (fights[:max_n] if max_n > 0 else [])]
            max_valid_n = int(getattr(cfg, "DIAG_MAX_VALIDATED_FIGHT_RESULTS", 10000) or 10000) if cfg else 10000
            max_valid_n = max(0, max_valid_n)
            diag["fight_results_validated_total"] = len(fights)
            diag["fight_results_validated_truncated"] = (
                int(len(fights) > max_valid_n) if max_valid_n > 0 else int(len(fights) > 0)
            )
            diag["fight_results_validated_brief"] = [
                _compact_fight_result(f) for f in (fights[:max_valid_n] if max_valid_n > 0 else [])
            ]
        except Exception as e:
            diag["errors"].append({"type": "diag_pack", "message": str(e)})

    # --- data setup ---
    xy = cache.get("xy_raw_minute", None)
    if xy is None:
        xi = NODE_IDX.get("x_norm", 0)
        yi = NODE_IDX.get("y_norm", 1)
        try:
            xy = cache["node_minute"][:, :, [xi, yi]]
        except (KeyError, IndexError) as e:
            logger.error(f"Position data error: {e}")
            _pack_diagnostics()
            cache["fight_detect_diag"] = diag
            return fights

    minute_ts = np.asarray(cache["minute_ts"], dtype=np.int64)
    events = cache.get("events", [])
    Tm = int(len(minute_ts))
    if Tm < 3:
        logger.warning(f"Insufficient frames: {Tm}")
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    is_norm = bool(cache.get("meta", {}).get("anchor_is_norm", False))
    scale_factor = float(config.coord_norm_div)
    if not is_norm:
        is_norm, scale_factor = detect_coordinate_scale(xy)

    R = float(config.standoff_radius)
    cluster_max_diam = float(config.cluster_max_diameter)
    if cluster_max_diam <= 0:
        cluster_max_diam = 4000.0
    continuous_merge_radius = float(config.continuous_fight_merge_radius)
    if is_norm:
        R /= scale_factor
        if cluster_max_diam > 0:
            cluster_max_diam /= scale_factor
        if continuous_merge_radius > 0:
            continuous_merge_radius /= scale_factor

    # Use minute grid for xy/dists (needed for engagement + visualization)
    dense_ts = minute_ts.copy()
    xy_dense = xy.astype(np.float32, copy=False)
    step_ms = int(config.frame_ms)

    Td = int(len(dense_ts))
    if Td < 3:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    diag["Td"] = Td
    diag["step_ms"] = step_ms

    dists = compute_distances_chunked(xy_dense, b, r, config.chunk_size)
    prox_pairs = np.sum(dists <= R, axis=(1, 2)).astype(np.int32)

    # --- extract events ---
    kill_events_rich = _extract_kill_events_rich(events)
    kill_events = _extract_kill_events(events)  # lean version for outcome
    kill_ts = (
        np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
        if kill_events
        else np.empty((0,), dtype=np.int64)
    )
    ace_ts = _extract_ace_ts(events)

    if not kill_events_rich:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    # --- config for killchain ---
    chain_window_ms = int(getattr(cfg, "KILLCHAIN_WINDOW_MS", 30000)) if cfg is not None else 30000
    backtrack_ms = int(getattr(cfg, "KILLCHAIN_BACKTRACK_MS", 10000)) if cfg is not None else 10000
    req_per_team = int(max(1, int(config.require_engaged_per_team)))

    t_min_ms = int(minute_ts[0])
    t_max_ms = int(minute_ts[-1])
    ctx_ms = int(config.fight_context_min) * 60000
    alive_idx = NODE_IDX.get("alive", None)

    def _check_alive_at_ts(engage_ts_val: int) -> bool:
        if int(config.require_alive_per_team) <= 0 or alive_idx is None:
            return True
        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)
        try:
            nm_alive = cache["node_minute"][m_idx, :, alive_idx]
            if float(nm_alive[b].sum()) < int(config.require_alive_per_team):
                return False
            if float(nm_alive[r].sum()) < int(config.require_alive_per_team):
                return False
        except Exception:
            return True
        return True

    # --- build kill chains ---
    chains = _build_kill_chains(kill_events_rich, chain_window_ms=chain_window_ms)
    diag["chains_total"] = len(chains)

    # --- convert chains to fight candidates ---
    # Map participant IDs to teams for per-team counting
    blue_set = set(int(x + 1) for x in b.tolist())
    red_set = set(int(x + 1) for x in r.tolist())

    for chain in chains:
        first_kill_ts = int(chain["start_ts"])
        engage_ts_val = int(max(t_min_ms, first_kill_ts - backtrack_ms))

        # Ensure engage < first kill
        if engage_ts_val >= first_kill_ts:
            engage_ts_val = int(max(t_min_ms, first_kill_ts - 1))

        # Context / horizon guards
        if engage_ts_val - ctx_ms < t_min_ms:
            diag["rejected_startctx"] += 1
            continue
        if engage_ts_val + horizon_ms > t_max_ms:
            diag["rejected_horizon"] += 1
            continue
        if not _check_alive_at_ts(engage_ts_val):
            diag["rejected_alive"] += 1
            continue

        # Per-team participant filter
        pids = chain["participants"]
        blue_in_fight = pids & blue_set
        red_in_fight = pids & red_set
        blue_cnt = len(blue_in_fight)
        red_cnt = len(red_in_fight)

        if blue_cnt < req_per_team or red_cnt < req_per_team:
            diag["rejected_too_few_per_team"] += 1
            continue

        # Compute cross-team pairs for compatibility with classify_fight_type
        cross_team_pairs = blue_cnt * red_cnt

        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)

        # Centroid from kill positions (ms precision)
        cx = float(chain["centroid_x"])
        cy = float(chain["centroid_y"])

        # Normalize centroid if coordinates are normalized
        if is_norm and scale_factor > 0:
            cx /= scale_factor
            cy /= scale_factor

        # Total damage from chain for scoring
        total_dmg = 0.0
        for k in chain["kills"]:
            for dmg in k.get("victim_damage_received", []):
                if isinstance(dmg, dict):
                    total_dmg += float(dmg.get("magicDamage", 0) or 0)
                    total_dmg += float(dmg.get("physicalDamage", 0) or 0)
                    total_dmg += float(dmg.get("trueDamage", 0) or 0)

        n_kills = chain["n_kills"]

        candidates_out.append(
            {
                "engage_ts": int(engage_ts_val),
                "t_engage": int(m_idx),
                "t_engage_ts": int(engage_ts_val),
                "first_kill_ts": int(first_kill_ts),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "horizon_end_ts": int(engage_ts_val + horizon_ms),
                "n_segments": 1,
                "det_step_ms": int(step_ms),
                "det_prox_pairs": int(cross_team_pairs),
                "det_min_dist_mean": 0.0,
                "det_anchor": 0,
                "det_backtracked": 1,
                "det_backtrack_reliable": 1,
                "det_damage_norm": 0.0,
                "det_summoner_spells": 0,
                "det_signal_ok": 1,
                "det_score_ok": 1,
                "det_event_score": float(n_kills),
                "det_event_count": int(n_kills),
                "det_kill_count_window": int(n_kills),
                "det_objective_count_window": 0,
                "det_building_count_window": 0,
                "det_combat_signal_ok": 1,
                "det_killchain_participants": int(len(pids)),
                "det_killchain_blue": int(blue_cnt),
                "det_killchain_red": int(red_cnt),
                "det_killchain_total_damage": float(total_dmg),
                "det_killchain_duration_ms": int(chain["end_ts"] - chain["start_ts"]),
            }
        )
        diag["chains_accepted"] += 1

    diag["candidates"] = len(candidates_out)
    diag["accepted"] = len(candidates_out)

    if not candidates_out:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    # --- merge candidates ---
    candidates_out.sort(key=lambda f: int(f["engage_ts"]))
    if bool(config.continuous_fight_merge):
        merge_eps_s = float(cluster_max_diam if cluster_max_diam > 0 else continuous_merge_radius)
        if merge_eps_s > 0:
            fights = merge_fights_st_dbscan_unionfind(
                candidates_out,
                eps_t_ms=int(config.continuous_fight_max_gap_ms),
                eps_s=float(merge_eps_s),
                horizon_ms=horizon_ms,
                max_duration_ms=int(config.max_merged_fight_duration_ms),
                diag=diag,
            )
        else:
            fights = merge_fights_temporal_unionfind(
                candidates_out,
                max_gap_ms=int(config.continuous_fight_max_gap_ms),
                horizon_ms=horizon_ms,
                max_duration_ms=int(config.max_merged_fight_duration_ms),
                diag=diag,
            )
    else:
        fights = []
        last_ts = -(10**18)
        for f in candidates_out:
            if int(f["engage_ts"]) - int(last_ts) < int(config.fight_min_gap_ms):
                diag["rejected_gap"] += 1
                continue
            fights.append(f)
            last_ts = int(f["engage_ts"])

    # ACE truncation
    _truncate_fights_at_ace(
        fights,
        ace_ts,
        horizon_ms=int(horizon_ms),
        diag=diag,
    )

    # Post-merge spacing enforcement
    fights = enforce_postmerge_spacing_and_nonoverlap(
        fights,
        horizon_ms=int(horizon_ms),
        fight_min_gap_ms=int(config.fight_min_gap_ms),
        kill_ts=kill_ts,
        location_radius=float(max(0.0, cluster_max_diam)),
        diag=diag,
    )

    # --- analysis: classify, outcome, importance, engagement, viz ---
    anchors = build_anchors_from_events(events)
    game_duration_ms = int(minute_ts[-1]) if len(minute_ts) > 0 else 0
    for fight in fights:
        try:
            fight["fight_type"] = classify_fight_type(fight, anchors, is_norm, scale_factor)
            outcome = compute_fight_outcome(fight, kill_events, tm, cache=cache, events=events)
            fight["outcome"] = outcome
            fight["importance_score"] = compute_fight_importance(fight, outcome, fight["fight_type"], game_duration_ms)
            fight["player_engagement"] = compute_player_engagement(fight, xy_dense, dists, dense_ts, R, b, r)
            fight["visualization"] = generate_fight_visualization(
                fight, xy_dense, dists, dense_ts, prox_pairs, kill_events, b, r, R
            )
        except Exception as e:
            logger.warning(f"Analysis failed for fight at {fight.get('engage_ts')}: {e}")
            diag["errors"].append({"type": "analysis", "engage_ts": fight.get("engage_ts"), "message": str(e)})

    _pack_diagnostics()
    cache["fight_detect_diag"] = diag
    return fights


def detect_fights(cache: Dict[str, Any], tm: Dict[int, int]) -> List[dict]:
    """교전 감지 진입점 (레거시 호환)"""
    algo = "event_v1"
    if cfg is not None:
        algo = str(getattr(cfg, "FIGHT_DETECTOR", getattr(cfg, "FIGHT_DETECT_ALGO", "event_v1"))).lower()

    if algo in ("killchain_v1", "killchain", "kc"):
        return detect_fights_killchain_v1(cache, tm)

    if algo in ("event_v1", "event", "v1"):
        return detect_fights_event_v1(cache, tm)

    if algo in ("engage_v2", "v2", "engage"):
        return detect_fights_engage_v2(cache, tm)

    logger.warning(f"Unknown fight detector algo={algo}. Falling back to event_v1.")
    return detect_fights_event_v1(cache, tm)


def summarize_fights(fights: List[dict]) -> Dict[str, Any]:
    """교전 요약 생성"""
    if not fights:
        return {"total_fights": 0, "by_type": {}, "by_winner": {}, "avg_importance": 0.0, "total_kills": 0}

    by_type: Dict[str, int] = {}
    by_winner: Dict[str, int] = {"blue": 0, "red": 0, "draw": 0}
    total_importance = 0.0
    total_kills = 0

    for fight in fights:
        ft = str(fight.get("fight_type", "unknown"))
        by_type[ft] = by_type.get(ft, 0) + 1

        outcome = fight.get("outcome", {})
        winner = "draw"
        if isinstance(outcome, dict):
            winner = str(outcome.get("winner", "draw"))
        by_winner[winner] = by_winner.get(winner, 0) + 1

        total_importance += float(fight.get("importance_score", 0.0) or 0.0)
        if isinstance(outcome, dict):
            total_kills += int(outcome.get("total_kills", 0) or 0)

    return {
        "total_fights": len(fights),
        "by_type": by_type,
        "by_winner": by_winner,
        "avg_importance": (total_importance / len(fights)) if fights else 0.0,
        "total_kills": total_kills,
    }


def summarize_fight_type_changes(fights: List[dict]) -> Dict[str, Any]:
    if not fights:
        return {}

    agg: Dict[str, Dict[str, float]] = {}
    for fight in fights:
        ft = str(fight.get("fight_type", "unknown"))
        outcome = fight.get("outcome", {})
        if not isinstance(outcome, dict):
            outcome = {}

        if ft not in agg:
            agg[ft] = {
                "count": 0.0,
                "blue_win": 0.0,
                "red_win": 0.0,
                "draw": 0.0,
                "total_kills": 0.0,
                "blue_deaths": 0.0,
                "red_deaths": 0.0,
                "blue_survivors": 0.0,
                "red_survivors": 0.0,
                "gold_diff": 0.0,
                "gold_blue_delta": 0.0,
                "gold_red_delta": 0.0,
                "tower_diff": 0.0,
                "tower_blue": 0.0,
                "tower_red": 0.0,
                "objective_diff": 0.0,
                "objective_blue": 0.0,
                "objective_red": 0.0,
            }

        rec = agg[ft]
        rec["count"] += 1.0

        winner = str(outcome.get("winner", "draw"))
        if winner == "blue":
            rec["blue_win"] += 1.0
        elif winner == "red":
            rec["red_win"] += 1.0
        else:
            rec["draw"] += 1.0

        rec["total_kills"] += float(outcome.get("total_kills", 0) or 0.0)
        rec["blue_deaths"] += float(outcome.get("blue_deaths", 0) or 0.0)
        rec["red_deaths"] += float(outcome.get("red_deaths", 0) or 0.0)
        rec["blue_survivors"] += float(outcome.get("blue_survivors", 0) or 0.0)
        rec["red_survivors"] += float(outcome.get("red_survivors", 0) or 0.0)
        rec["gold_diff"] += float(outcome.get("gold_diff", 0.0) or 0.0)
        rec["gold_blue_delta"] += float(outcome.get("gold_blue_delta", 0.0) or 0.0)
        rec["gold_red_delta"] += float(outcome.get("gold_red_delta", 0.0) or 0.0)
        rec["tower_diff"] += float(outcome.get("tower_diff", 0) or 0.0)
        rec["tower_blue"] += float(outcome.get("tower_blue", 0) or 0.0)
        rec["tower_red"] += float(outcome.get("tower_red", 0) or 0.0)
        rec["objective_diff"] += float(outcome.get("objective_diff", 0) or 0.0)
        rec["objective_blue"] += float(outcome.get("objective_blue", 0) or 0.0)
        rec["objective_red"] += float(outcome.get("objective_red", 0) or 0.0)

    out: Dict[str, Any] = {}
    for ft, rec in agg.items():
        c = float(max(1.0, rec.get("count", 0.0)))
        out[ft] = {
            "count": int(rec["count"]),
            "blue_win_rate": float(rec["blue_win"] / c),
            "red_win_rate": float(rec["red_win"] / c),
            "draw_rate": float(rec["draw"] / c),
            "avg_total_kills": float(rec["total_kills"] / c),
            "avg_blue_deaths": float(rec["blue_deaths"] / c),
            "avg_red_deaths": float(rec["red_deaths"] / c),
            "avg_blue_survivors": float(rec["blue_survivors"] / c),
            "avg_red_survivors": float(rec["red_survivors"] / c),
            "avg_gold_diff": float(rec["gold_diff"] / c),
            "avg_gold_blue_delta": float(rec["gold_blue_delta"] / c),
            "avg_gold_red_delta": float(rec["gold_red_delta"] / c),
            "avg_tower_diff": float(rec["tower_diff"] / c),
            "avg_tower_blue": float(rec["tower_blue"] / c),
            "avg_tower_red": float(rec["tower_red"] / c),
            "avg_objective_diff": float(rec["objective_diff"] / c),
            "avg_objective_blue": float(rec["objective_blue"] / c),
            "avg_objective_red": float(rec["objective_red"] / c),
        }
    return out
