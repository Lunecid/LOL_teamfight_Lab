from __future__ import annotations

import logging
import math
import re
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
from gameplay.fight_postmerge import (
    count_events_in_window as _count_events_in_window,
    distance_2d as _distance_2d,
    enforce_postmerge_spacing_and_nonoverlap,
    fight_priority_score as _fight_priority_score,
    label_end_ts as _label_end_ts,
    safe_int,
)
from gameplay.fight_metrics import (
    gold_team_at_ms as _gold_team_at_ms_impl,
    team_of_pid as _team_of_pid_impl,
    window_resource_changes as _window_resource_changes_impl,
)
from gameplay.fight_analysis import (
    compute_fight_importance_impl as _compute_fight_importance_impl,
    compute_fight_outcome_impl as _compute_fight_outcome_impl,
    compute_player_engagement_impl as _compute_player_engagement_impl,
    generate_fight_visualization_impl as _generate_fight_visualization_impl,
)
from gameplay.fight_clustering import (
    build_5s_position_grid as _build_5s_position_grid_impl,
    cluster_kills_temporal as _cluster_kills_temporal_impl,
    finalize_kill_cluster as _finalize_kill_cluster_impl,
    split_kill_cluster_spatial as _split_kill_cluster_spatial_impl,
)

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
    fight_min_gap_ms: int = 0
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
    require_alive_per_team: int = 2
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
        if (
            self.continuous_fight_merge
            and self.fight_min_gap_ms > 0
            and self.continuous_fight_max_gap_ms >= self.fight_min_gap_ms
        ):
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
                getattr(cfg_obj, "FIGHT_MIN_GAP_MS", int(getattr(cfg_obj, "FIGHT_MIN_GAP_MIN", 1)) * 60000)
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
            require_alive_per_team=int(getattr(cfg_obj, "REQUIRE_ALIVE_PER_TEAM", 2) or 0),
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

def _get_horizon_ms() -> int:
    """cfg에서 horizon_ms 가져오기"""
    if cfg is None:
        return 60000
    if hasattr(cfg, "FIGHT_HORIZON_SEC"):
        return int(getattr(cfg, "FIGHT_HORIZON_SEC", 60)) * 1000
    return int(getattr(cfg, "FIGHT_HORIZON_MIN", 1)) * 60000

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
    s = str(game_version or "").strip()
    if not s:
        return "0.0"

    patch_level = getattr(cfg, "PATCH_LEVEL", "major_minor") if cfg else "major_minor"
    if patch_level == "full":
        return s

    # Robust extraction: handles canonical "15.20.719.545" and noisy strings.
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        try:
            major = str(int(nums[0]))
            minor = str(int(nums[1]))
            return f"{major}.{minor}"
        except Exception:
            return f"{nums[0]}.{nums[1]}"

    parts = s.split(".")
    if len(parts) >= 2:
        return f"{parts[0].strip()}.{parts[1].strip()}"
    return s

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
    return _team_of_pid_impl(pid, tm)


def _gold_team_at_ms(cache: Optional[Dict[str, Any]], q_ms: int) -> Optional[np.ndarray]:
    return _gold_team_at_ms_impl(cache, q_ms, cfg_obj=cfg)


def _window_resource_changes(events: List[dict], tm: Dict[int, int], t0: int, t1_exclusive: int) -> Dict[str, Any]:
    return _window_resource_changes_impl(events, tm, t0, t1_exclusive)

def compute_fight_outcome(
    fight: dict,
    kill_events: List[dict],
    tm: Dict[int, int],
    cache: Optional[Dict[str, Any]] = None,
    events: Optional[List[dict]] = None,
) -> FightOutcome:
    out = _compute_fight_outcome_impl(
        fight,
        kill_events,
        tm,
        label_end_ts_fn=_label_end_ts,
        get_horizon_ms_fn=_get_horizon_ms,
        node_idx=NODE_IDX,
        map_ts_to_minute_idx_fn=_map_ts_to_minute_idx,
        team_of_pid_fn=_team_of_pid,
        gold_team_at_ms_fn=_gold_team_at_ms,
        window_resource_changes_fn=_window_resource_changes,
        cache=cache,
        events=events,
    )
    return FightOutcome(**out)


def compute_player_engagement(
    fight: dict,
    xy_dense: np.ndarray,
    dists: np.ndarray,
    dense_ts: np.ndarray,
    R: float,
    b: np.ndarray,
    r: np.ndarray,
) -> List[PlayerEngagement]:
    rows = _compute_player_engagement_impl(
        fight,
        xy_dense,
        dists,
        dense_ts,
        R,
        b,
        r,
        label_end_ts_fn=_label_end_ts,
        get_horizon_ms_fn=_get_horizon_ms,
    )
    return [PlayerEngagement(**row) for row in rows]


def compute_fight_importance(fight: dict, outcome: FightOutcome, fight_type: str, game_duration_ms: int) -> float:
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
    return _compute_fight_importance_impl(
        fight,
        outcome,
        fight_type,
        game_duration_ms,
        type_scores=type_scores,
    )


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
    out = _generate_fight_visualization_impl(
        fight,
        xy_dense,
        dists,
        dense_ts,
        prox_pairs,
        kill_events,
        b,
        r,
        R,
        label_end_ts_fn=_label_end_ts,
        get_horizon_ms_fn=_get_horizon_ms,
        sample_interval=sample_interval,
    )
    return FightVisualization(**out)


def _build_5s_position_grid(
    xy_minute: np.ndarray,
    minute_ts: np.ndarray,
    kill_events: List[dict],
    tm: Dict[int, int],
    *,
    cfg_obj: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return _build_5s_position_grid_impl(
        xy_minute=xy_minute,
        minute_ts=minute_ts,
        kill_events=kill_events,
        tm=tm,
        cfg_obj=cfg_obj,
    )


def _cluster_kills_temporal(
    kill_events: List[dict],
    gap_ms: int,
) -> List[dict]:
    return _cluster_kills_temporal_impl(kill_events, gap_ms)


def _split_kill_cluster_spatial(cluster: dict, max_diameter: float) -> List[dict]:
    return _split_kill_cluster_spatial_impl(cluster, max_diameter)


def _finalize_kill_cluster(kills: List[dict]) -> dict:
    return _finalize_kill_cluster_impl(kills)


def _validate_teamfight_at_engage(
    xy_dense: np.ndarray,
    dense_ts: np.ndarray,
    engage_ts: int,
    fight_center: Tuple[float, float],
    b: np.ndarray,
    r: np.ndarray,
    validity_radius: float,
    min_per_team: int,
    is_norm: bool,
    scale_factor: float,
) -> bool:
    """Check teamfight validity: at engage time, require min_per_team
    champions from each team within validity_radius of fight center.

    Uses the 5-second dense XY grid for spatial checks.
    """
    Td = int(len(dense_ts))
    d_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts, side="right") - 1, 0, Td - 1))

    R = float(validity_radius)
    if is_norm and scale_factor > 0:
        R /= scale_factor

    cx, cy = float(fight_center[0]), float(fight_center[1])
    if is_norm and scale_factor > 0:
        cx /= scale_factor
        cy /= scale_factor

    R_sq = R * R
    blue_in = 0
    red_in = 0

    for bi in b:
        dx = float(xy_dense[d_idx, int(bi), 0]) - cx
        dy = float(xy_dense[d_idx, int(bi), 1]) - cy
        if dx * dx + dy * dy <= R_sq:
            blue_in += 1

    for ri in r:
        dx = float(xy_dense[d_idx, int(ri), 0]) - cx
        dy = float(xy_dense[d_idx, int(ri), 1]) - cy
        if dx * dx + dy * dy <= R_sq:
            red_in += 1

    return blue_in >= min_per_team and red_in >= min_per_team

def _collect_interactions_in_radius(
    events: List[dict],
    fight_start: int,
    fight_end: int,
    fight_center: Tuple[float, float],
    interaction_radius: float,
    xy_dense: np.ndarray,
    dense_ts: np.ndarray,
    is_norm: bool,
    scale_factor: float,
) -> Tuple[List[dict], set]:
    """Collect non-kill events as interactions within fight time + radius 3000.

    Position-based events only (wards, spells, etc.).
    Objectives/tower events are excluded here — they are tracked only in
    the post-fight outcome window (Step 5) to prevent double-counting.
    Returns (interactions, additional_participant_ids).
    """
    R = float(interaction_radius)
    if is_norm and scale_factor > 0:
        R /= scale_factor

    cx, cy = float(fight_center[0]), float(fight_center[1])
    if is_norm and scale_factor > 0:
        cx /= scale_factor
        cy /= scale_factor

    R_sq = R * R
    Td = int(len(dense_ts))
    interactions: List[dict] = []
    extra_pids: set = set()

    obj_building_types = {
        "ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED",
    }

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et == "CHAMPION_KILL":
            continue  # kills already handled

        ts = ev.get("timestamp", ev.get("ts"))
        if ts is None:
            continue
        try:
            ts_val = int(ts)
        except (TypeError, ValueError):
            continue

        if ts_val < fight_start or ts_val > fight_end:
            continue

        # Objectives/towers tracked only in post-fight outcome (Step 5),
        # NOT counted as radius-3000 interactions (prevents double-counting).
        if et in obj_building_types:
            continue

        # Check spatial constraint (radius 3000)
        pos = _event_xy(ev)
        if pos is not None:
            dx = float(pos[0]) - cx
            dy = float(pos[1]) - cy
            if is_norm and scale_factor > 0:
                dx = float(pos[0]) / scale_factor - cx
                dy = float(pos[1]) / scale_factor - cy
            if dx * dx + dy * dy > R_sq:
                continue
        else:
            # Approximate position using actor's dense 5s XY
            actor_id = safe_int(ev.get("participantId", ev.get("killerId", ev.get("creatorId", 0))))
            if 1 <= actor_id <= 10:
                d_idx = int(np.clip(np.searchsorted(dense_ts, ts_val, side="right") - 1, 0, Td - 1))
                px = float(xy_dense[d_idx, actor_id - 1, 0])
                py = float(xy_dense[d_idx, actor_id - 1, 1])
                dx = px - cx
                dy = py - cy
                if dx * dx + dy * dy > R_sq:
                    continue
            else:
                continue

        interactions.append(ev)

        # Add actors to participant candidate set
        for key in ("participantId", "killerId", "creatorId"):
            pid = safe_int(ev.get(key, 0))
            if 1 <= pid <= 10:
                extra_pids.add(pid)

    return interactions, extra_pids

def _compute_postfight_outcome(
    events: List[dict],
    tm: Dict[int, int],
    cache: Dict[str, Any],
    fight_end_ts: int,
    post_window_ms: int,
) -> Dict[str, Any]:
    """Aggregate post-fight signals in a fixed window after fight end.

    Collects objectives/tower results (radius-independent per §4C).
    Gold delta from raw 60s snapshots only (no interpolation per §7).
    """
    post_start = int(fight_end_ts)
    post_end = int(fight_end_ts + post_window_ms)

    obj_blue = 0
    obj_red = 0
    tower_blue = 0
    tower_red = 0
    objectives: List[dict] = []

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        ts = ev.get("timestamp", ev.get("ts"))
        if ts is None:
            continue
        try:
            ts_val = int(ts)
        except (TypeError, ValueError):
            continue
        if ts_val < post_start or ts_val > post_end:
            continue

        et = str(ev.get("type", ev.get("eventType", ""))).upper()

        if et == "ELITE_MONSTER_KILL":
            killer_team = safe_int(ev.get("killerTeamId", 0))
            if killer_team == 100:
                obj_blue += 1
            elif killer_team == 200:
                obj_red += 1
            objectives.append(ev)

        elif et in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
            # Team that destroyed it
            killer_team = safe_int(ev.get("teamId", 0))
            # In building events, teamId is the team that LOST the building
            victim_team = killer_team
            if victim_team == 100:
                tower_red += 1  # red destroyed blue tower
            elif victim_team == 200:
                tower_blue += 1  # blue destroyed red tower

    # Gold delta from raw 60s snapshots only
    gold_diff = 0.0
    try:
        g0 = _gold_team_at_ms(cache, post_start)
        g1 = _gold_team_at_ms(cache, min(post_end, int(np.asarray(cache.get("minute_ts", [0])).max())))
        if g0 is not None and g1 is not None:
            gold_diff = float((g1[0] - g0[0]) - (g1[1] - g0[1]))
    except Exception:
        pass

    return {
        "post_obj_blue": obj_blue,
        "post_obj_red": obj_red,
        "post_obj_diff": obj_blue - obj_red,
        "post_tower_blue": tower_blue,
        "post_tower_red": tower_red,
        "post_tower_diff": tower_blue - tower_red,
        "post_gold_diff": gold_diff,
        "post_objectives": objectives,
    }

def detect_fights_teamfight_v2(
    cache: Dict[str, Any],
    tm: Dict[int, int],
    config: Optional[FightDetectorConfig] = None,
) -> List[dict]:
    """Kill-cluster-based teamfight detector (v2).

    Algorithm:
      1. Build 5-second position grid with baseline + pre-kill override.
      2. Cluster kill events temporally (gap threshold ~18s),
         then split temporal clusters spatially (diameter <= CLUSTER_MAX_DIAMETER).
      3. For each cluster:
         a. Fight center = first kill XY.
         b. Engage time = first_kill_ts - 10s.
         c. Validate teamfight: >=2 per team within radius 1800 at engage.
         d. Fight end = last kill in cluster.
      4. Collect interactions within radius 3000 (position-based events only;
         objectives/towers tracked in Step 5 post-fight outcome only).
      5. Post-fight outcome: 45s window for objectives/towers/gold.
      6. Model input: closest 60s snapshot before fight start, XY excluded.

    Key changes from legacy detectors:
      - Kills ONLY create fights (no ward/objective hard gates).
      - Single consistent definition: kills → temporal clustering → spatial split → radii → validation.
      - No multi-stage time-window guards.
    """
    diag: Dict[str, Any] = {
        "Td": 0,
        "step_ms": 5000,
        "detector": "teamfight_v2",
        "candidates": 0,
        "clusters_total": 0,
        "clusters_accepted": 0,
        "accepted": 0,
        "rejected_startctx": 0,
        "rejected_start_offset": 0,
        "rejected_horizon": 0,
        "rejected_alive": 0,
        "rejected_too_few_per_team": 0,
        "rejected_gap": 0,
        "rejected_max_duration": 0,
        "ace_events": 0,
        "ace_end_truncated": 0,
        "postmerge_conflicts": 0,
        "postmerge_removed": 0,
        "postmerge_replaced": 0,
        "postmerge_overlap_clipped": 0,
        "postmerge_overlap_dropped": 0,
        "errors": [],
    }

    if config is None:
        try:
            config = FightDetectorConfig.from_cfg(cfg)
        except Exception as e:
            logger.warning(f"Config load failed: {e}. Using defaults.")
            config = FightDetectorConfig()

    horizon_ms = int(_get_horizon_ms())

    try:
        b, r = validate_team_mapping(tm)
    except Exception as e:
        logger.error(f"Team mapping error: {e}")
        diag["errors"].append({"type": "team_mapping", "message": str(e)})
        b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

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
            out["last_kill_ts"] = int(f.get("last_kill_ts", -1) or -1)
            outcome = f.get("outcome", {}) if isinstance(f.get("outcome", {}), dict) else {}
            out["winner"] = str(outcome.get("winner", "unknown"))
            out["kill_diff"] = int(outcome.get("kill_diff", 0) or 0)
            out["total_kills"] = int(outcome.get("total_kills", 0) or 0)
            out["blue_deaths"] = int(outcome.get("blue_deaths", 0) or 0)
            out["red_deaths"] = int(outcome.get("red_deaths", 0) or 0)
            out["gold_diff"] = float(outcome.get("gold_diff", 0.0) or 0.0)
        except Exception:
            pass
        return out

    def _pack_diagnostics():
        try:
            diag["fight_summary"] = summarize_fights(fights)
            diag["fight_type_change_summary"] = summarize_fight_type_changes(fights)
            max_n = int(getattr(cfg, "DIAG_MAX_FIGHT_RESULTS", 50) or 50) if cfg else 50
            diag["fight_results_total"] = len(fights)
            diag["fight_results_truncated"] = int(len(fights) > max_n)
            diag["fight_results_brief"] = [_compact_fight_result(f) for f in fights[:max_n]]
            max_valid_n = int(getattr(cfg, "DIAG_MAX_VALIDATED_FIGHT_RESULTS", 10000) or 10000) if cfg else 10000
            diag["fight_results_validated_total"] = len(fights)
            diag["fight_results_validated_truncated"] = int(len(fights) > max_valid_n)
            diag["fight_results_validated_brief"] = [_compact_fight_result(f) for f in fights[:max_valid_n]]
        except Exception as e:
            diag["errors"].append({"type": "diag_pack", "message": str(e)})

    # --- Data setup ---
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

    # --- Extract kill events ---
    kill_events = _extract_kill_events(events)
    kill_ts = (
        np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
        if kill_events
        else np.empty((0,), dtype=np.int64)
    )
    ace_ts = _extract_ace_ts(events)

    if not kill_events:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    diag["candidates"] = len(kill_events)

    # --- Config parameters ---
    kill_cluster_gap_ms = int(getattr(cfg, "TF2_KILL_CLUSTER_GAP_MS", 18000)) if cfg else 18000
    engage_pre_kill_ms = int(getattr(cfg, "TF2_ENGAGE_PRE_KILL_MS", 10000)) if cfg else 10000
    validity_radius = float(getattr(cfg, "TF2_VALIDITY_RADIUS", 1800.0)) if cfg else 1800.0
    interaction_radius = float(getattr(cfg, "TF2_INTERACTION_RADIUS", 3000.0)) if cfg else 3000.0
    post_fight_window_ms = int(getattr(cfg, "TF2_POST_FIGHT_WINDOW_MS", 45000)) if cfg else 45000
    tail_buffer_ms = int(getattr(cfg, "TF2_TAIL_BUFFER_MS", 0)) if cfg else 0
    min_per_team = int(getattr(cfg, "TF2_MIN_PER_TEAM", 2)) if cfg else 2

    t_min_ms = int(minute_ts[0])
    t_max_ms = int(minute_ts[-1])
    ctx_ms = int(config.fight_context_min) * 60000
    start_offset_ms = int(getattr(cfg, "START_OFFSET_MIN", 2)) * 60000 if cfg else 120000
    alive_idx = NODE_IDX.get("alive", None)

    # --- Step 1: Build 5-second position grid ---
    dense_ts, xy_dense = _build_5s_position_grid(
        xy_minute=xy,
        minute_ts=minute_ts,
        kill_events=kill_events,
        tm=tm,
        cfg_obj=cfg,
    )
    Td = int(len(dense_ts))
    diag["Td"] = Td

    # Also compute distances for engagement/visualization compatibility
    R_compat = float(config.standoff_radius)
    if is_norm and scale_factor > 0:
        R_compat /= scale_factor
    dists = compute_distances_chunked(xy_dense, b, r, config.chunk_size)
    prox_pairs = np.sum(dists <= R_compat, axis=(1, 2)).astype(np.int32)

    # --- Step 2: Cluster kills (temporal -> spatial split) ---
    clusters = _cluster_kills_temporal(kill_events, kill_cluster_gap_ms)
    diag["clusters_total"] = len(clusters)

    spatial_added = 0
    spatial_diam = float(getattr(config, "cluster_max_diameter", 0.0) or 0.0)
    if spatial_diam > 0.0:
        split_clusters: List[dict] = []
        for cl in clusters:
            subs = _split_kill_cluster_spatial(cl, max_diameter=spatial_diam)
            if len(subs) > 1:
                spatial_added += (len(subs) - 1)
            split_clusters.extend(subs)
        clusters = sorted(split_clusters, key=lambda c: int(c.get("first_kill_ts", 0)))

    diag["clusters_spatial_added"] = int(spatial_added)
    diag["clusters_after_spatial"] = int(len(clusters))

    def _check_alive_at_ts(ts_val: int) -> bool:
        if int(config.require_alive_per_team) <= 0 or alive_idx is None:
            return True
        m_idx = _map_ts_to_minute_idx(minute_ts, ts_val)
        try:
            nm_alive = cache["node_minute"][m_idx, :, alive_idx]
            if float(nm_alive[b].sum()) < int(config.require_alive_per_team):
                return False
            if float(nm_alive[r].sum()) < int(config.require_alive_per_team):
                return False
        except Exception:
            return True
        return True

    # --- Step 3: Convert clusters to fight candidates ---
    blue_set = set(int(x + 1) for x in b.tolist())
    red_set = set(int(x + 1) for x in r.tolist())
    candidates_out: List[dict] = []

    for cluster in clusters:
        first_kill_ts = int(cluster["first_kill_ts"])
        last_kill_ts = int(cluster["last_kill_ts"])
        fight_center = cluster["fight_center"]

        # §3: engage time = ~10s before first kill
        engage_ts_val = int(max(t_min_ms, first_kill_ts - engage_pre_kill_ms))
        if engage_ts_val >= first_kill_ts:
            engage_ts_val = int(max(t_min_ms, first_kill_ts - 1))

        # Context / horizon guards
        if engage_ts_val - ctx_ms < t_min_ms:
            diag["rejected_startctx"] += 1
            continue
        # [FIX-6.3] Enforce START_OFFSET_MIN — reject fights too early in game.
        if engage_ts_val - t_min_ms < start_offset_ms:
            diag["rejected_start_offset"] += 1
            continue
        if engage_ts_val + horizon_ms > t_max_ms:
            diag["rejected_horizon"] += 1
            continue
        if not _check_alive_at_ts(engage_ts_val):
            diag["rejected_alive"] += 1
            continue

        # §4A: Validate teamfight — at least 2 per team within 1800 of fight center
        if not _validate_teamfight_at_engage(
            xy_dense=xy_dense,
            dense_ts=dense_ts,
            engage_ts=engage_ts_val,
            fight_center=fight_center,
            b=b, r=r,
            validity_radius=validity_radius,
            min_per_team=min_per_team,
            is_norm=is_norm,
            scale_factor=scale_factor,
        ):
            diag["rejected_too_few_per_team"] += 1
            continue

        # §5: fight time window
        fight_end_ts = last_kill_ts + tail_buffer_ms
        horizon_end_ts = int(max(fight_end_ts, engage_ts_val + horizon_ms))

        # Duration cap
        if fight_end_ts - engage_ts_val > int(config.max_merged_fight_duration_ms):
            diag["rejected_max_duration"] += 1
            continue

        # §6: Collect interactions within fight time + radius 3000
        interactions, extra_pids = _collect_interactions_in_radius(
            events=events,
            fight_start=engage_ts_val,
            fight_end=fight_end_ts,
            fight_center=fight_center,
            interaction_radius=interaction_radius,
            xy_dense=xy_dense,
            dense_ts=dense_ts,
            is_norm=is_norm,
            scale_factor=scale_factor,
        )

        # All participants: kill participants + interaction actors
        all_participants = cluster["participants"] | extra_pids
        blue_cnt = len(all_participants & blue_set)
        red_cnt = len(all_participants & red_set)

        m_idx = _map_ts_to_minute_idx(minute_ts, engage_ts_val)
        cx, cy = float(fight_center[0]), float(fight_center[1])
        if is_norm and scale_factor > 0:
            cx /= scale_factor
            cy /= scale_factor

        candidates_out.append({
            "engage_ts": int(engage_ts_val),
            "t_engage": int(m_idx),
            "t_engage_ts": int(engage_ts_val),
            "first_kill_ts": int(first_kill_ts),
            "last_kill_ts": int(last_kill_ts),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "horizon_end_ts": int(horizon_end_ts),
            "n_segments": 1,
            "det_step_ms": 5000,
            "det_prox_pairs": int(blue_cnt * red_cnt),
            "det_min_dist_mean": 0.0,
            "det_anchor": 0,
            "det_backtracked": 1,
            "det_backtrack_reliable": 1,
            "det_damage_norm": 0.0,
            "det_summoner_spells": 0,
            "det_signal_ok": 1,
            "det_score_ok": 1,
            "det_event_score": float(cluster["n_kills"]),
            "det_event_count": int(cluster["n_kills"]),
            "det_kill_count_window": int(cluster["n_kills"]),
            "det_combat_signal_ok": 1,
            "det_cluster_participants": int(len(all_participants)),
            "det_cluster_blue": int(blue_cnt),
            "det_cluster_red": int(red_cnt),
            "det_cluster_duration_ms": int(last_kill_ts - first_kill_ts),
            "det_interaction_count": int(len(interactions)),
        })
        diag["clusters_accepted"] += 1

    diag["accepted"] = len(candidates_out)

    if not candidates_out:
        _pack_diagnostics()
        cache["fight_detect_diag"] = diag
        return fights

    # --- Enforce minimum gap between fights ---
    candidates_out.sort(key=lambda f: int(f["engage_ts"]))
    fights = []
    last_ts = -(10**18)
    for f in candidates_out:
        if int(f["engage_ts"]) - int(last_ts) < int(config.fight_min_gap_ms):
            diag["rejected_gap"] += 1
            continue
        fights.append(f)
        last_ts = int(f["engage_ts"])

    # ACE truncation
    _truncate_fights_at_ace(fights, ace_ts, horizon_ms=int(horizon_ms), diag=diag)

    # Post-merge spacing enforcement
    postmerge_location_radius = float(getattr(config, "cluster_max_diameter", 0.0) or 0.0)
    if postmerge_location_radius > 0.0 and is_norm and scale_factor > 0:
        postmerge_location_radius /= scale_factor
    fights = enforce_postmerge_spacing_and_nonoverlap(
        fights,
        horizon_ms=int(horizon_ms),
        fight_min_gap_ms=int(config.fight_min_gap_ms),
        kill_ts=kill_ts,
        location_radius=float(max(0.0, postmerge_location_radius)),
        diag=diag,
    )

    # --- Analysis: classify, outcome, importance, engagement, viz ---
    anchors = build_anchors_from_events(events)
    game_duration_ms = int(minute_ts[-1]) if len(minute_ts) > 0 else 0
    for fight in fights:
        try:
            fight["fight_type"] = classify_fight_type(fight, anchors, is_norm, scale_factor)
            outcome = compute_fight_outcome(fight, kill_events, tm, cache=cache, events=events)
            fight["outcome"] = outcome
            fight["importance_score"] = compute_fight_importance(fight, outcome, fight["fight_type"], game_duration_ms)
            fight["player_engagement"] = compute_player_engagement(fight, xy_dense, dists, dense_ts, R_compat, b, r)
            fight["visualization"] = generate_fight_visualization(
                fight, xy_dense, dists, dense_ts, prox_pairs, kill_events, b, r, R_compat,
            )

            # §7: Post-fight outcome aggregation (45s after fight end)
            fight_end = int(fight.get("last_kill_ts", fight.get("first_kill_ts", fight["engage_ts"])))
            fight["post_fight_outcome"] = _compute_postfight_outcome(
                events=events,
                tm=tm,
                cache=cache,
                fight_end_ts=fight_end,
                post_window_ms=post_fight_window_ms,
            )
        except Exception as e:
            logger.warning(f"Analysis failed for fight at {fight.get('engage_ts')}: {e}")
            diag["errors"].append({"type": "analysis", "engage_ts": fight.get("engage_ts"), "message": str(e)})

    _pack_diagnostics()
    cache["fight_detect_diag"] = diag
    return fights

def detect_fights(cache: Dict[str, Any], tm: Dict[int, int]) -> List[dict]:
    """Fight detection entry point.

    Only teamfight_v2 (kill-cluster-based) detector is supported.
    Legacy detectors (engage_v2, event_v1, killchain_v1) have been removed.
    """
    return detect_fights_teamfight_v2(cache, tm)

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
