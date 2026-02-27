#!/usr/bin/env python3
"""
===============================================================================
 Teamfight Detection Algorithm — Paper Statistics Extractor
===============================================================================
 Description:
   5,000개의 *_fights.json 파일로부터 논문에 게재할 통계치를 체계적으로 추출합니다.
   출력은 세 가지 레벨로 구성됩니다:
     1) Match-level CSV   — 매치 단위 집계 (N = 5,000)
     2) Fight-level CSV   — 개별 팀파이트 단위 (N ≈ 23,000+)
     3) Paper Tables (LaTeX / Markdown / Console) — 논문 게재용 테이블

 Usage:
   python extract_paper_stats.py --input_dir /path/to/json_files \
                                  --output_dir /path/to/output

 Output files:
   ├── match_level_stats.csv          # 매치 단위 통계
   ├── fight_level_stats.csv          # 팀파이트 단위 통계
   ├── paper_tables.tex               # LaTeX 테이블 (IEEE CoG 형식)
   ├── paper_tables.md                # Markdown 테이블
   ├── paper_summary.json             # 기계 판독 가능 요약 통계
   └── extraction_log.txt             # 처리 로그
===============================================================================
"""

import json
import os
import sys
import glob
import argparse
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# pandas는 선택적 (없으면 CSV를 수동으로 작성)
# ---------------------------------------------------------------------------
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not found. CSV output will use manual writer.")

# ---------------------------------------------------------------------------
# scipy는 선택적 (통계적 검정에 사용)
# ---------------------------------------------------------------------------
try:
    from scipy import stats as sp_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# §1  Configuration & Constants
# ============================================================================

# JSON 파일의 예상 스키마 키들
EXPECTED_TOP_KEYS = {
    "match_id", "patch", "knobs", "diag",
    "n_fights_raw", "n_fights_kept",
    "fights_raw", "fights_kept",
}

FIGHT_TYPES_ORDERED = [
    "teamfight", "skirmish", "tower_dive", "pick",
    "objective_dragon", "objective_baron",
    "objective_riftherald", "objective_atakhan",
    "objective_horde", "other",
]

OBJECTIVE_TYPES = ["dragon", "baron", "herald", "atakhan", "horde", "other"]


# ============================================================================
# §2  Utility Functions
# ============================================================================

def safe_get(d: dict, *keys, default=None):
    """중첩 딕셔너리에서 안전하게 값을 추출합니다."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def compute_ci95(data: list, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Bootstrap 95% 신뢰구간을 계산합니다.

    수학적 정의:
        CI_{95} = [θ̂_{2.5%}, θ̂_{97.5%}]
        여기서 θ̂은 bootstrap 표본의 평균 분포
    """
    if len(data) < 2:
        m = np.mean(data) if data else 0.0
        return (m, m)
    arr = np.array(data, dtype=np.float64)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    return (float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


def fmt_ci(mean: float, ci_low: float, ci_high: float, precision: int = 2) -> str:
    """평균 [CI_low, CI_high] 형식의 문자열을 반환합니다."""
    return f"{mean:.{precision}f} [{ci_low:.{precision}f}, {ci_high:.{precision}f}]"


def fmt_mean_std(values: list, precision: int = 2) -> str:
    """μ ± σ 형식의 문자열을 반환합니다."""
    if not values:
        return "N/A"
    m, s = np.mean(values), np.std(values)
    return f"{m:.{precision}f} ± {s:.{precision}f}"


# ============================================================================
# §3  Match-Level Extractor
# ============================================================================

def extract_match_level(data: dict) -> dict:
    """
    단일 JSON 파일에서 매치 단위 통계를 추출합니다.

    Returns:
        dict: 매치 단위 피처 딕셔너리
    """
    diag = data.get("diag", {})
    knobs = data.get("knobs", {})
    cs_raw = data.get("continuous_stats_raw", {})
    cs_kept = data.get("continuous_stats_kept", {})
    fights = data.get("fights_kept", [])

    # --- 기본 메타데이터 ---
    row = {
        "match_id": data.get("match_id", ""),
        "patch": data.get("patch", ""),
        "patch_full": data.get("patch_full", ""),
        "run_tag": data.get("run_tag", ""),
        "detector": knobs.get("FIGHT_DETECTOR", ""),
        "game_duration_frames": diag.get("Td", 0),
        "step_ms": diag.get("step_ms", 0),
    }

    # --- 탐지 퍼널 (Detection Funnel) ---
    row["det_candidates"] = diag.get("candidates", 0)
    row["det_clusters_total"] = diag.get("clusters_total", 0)
    row["det_clusters_spatial"] = diag.get("clusters_spatial_added", 0)
    row["det_clusters_accepted"] = diag.get("clusters_accepted", 0)
    row["det_accepted"] = diag.get("accepted", 0)
    row["n_fights_raw"] = data.get("n_fights_raw", 0)
    row["n_fights_kept"] = data.get("n_fights_kept", 0)

    # --- 거부 사유 (Rejection Reasons) ---
    row["rej_startctx"] = diag.get("rejected_startctx", 0)
    row["rej_start_offset"] = diag.get("rejected_start_offset", 0)
    row["rej_horizon"] = diag.get("rejected_horizon", 0)
    row["rej_alive"] = diag.get("rejected_alive", 0)
    row["rej_too_few"] = diag.get("rejected_too_few_per_team", 0)
    row["rej_gap"] = diag.get("rejected_gap", 0)
    row["rej_max_duration"] = diag.get("rejected_max_duration", 0)

    # --- 에이스 / 포스트머지 ---
    row["ace_events"] = diag.get("ace_events", 0)
    row["ace_end_truncated"] = diag.get("ace_end_truncated", 0)
    row["postmerge_conflicts"] = diag.get("postmerge_conflicts", 0)
    row["postmerge_removed"] = diag.get("postmerge_removed", 0)
    row["postmerge_replaced"] = diag.get("postmerge_replaced", 0)
    row["postmerge_overlap_clip"] = diag.get("postmerge_overlap_clipped", 0)
    row["postmerge_overlap_drop"] = diag.get("postmerge_overlap_dropped", 0)

    # --- 연속 전투 통계 ---
    row["continuous_merged"] = cs_kept.get("merged_fights", 0)
    row["continuous_segments"] = cs_kept.get("total_segments", 0)
    row["continuous_avg_seg"] = cs_kept.get("avg_segments_per_fight", 0)
    row["continuous_max_seg"] = cs_kept.get("max_segments", 0)

    # --- 오류 ---
    row["n_errors"] = len(diag.get("errors", []))

    # --- 팀파이트 집계 ---
    if fights:
        engage_ts_list = [f["engage_ts"] for f in fights]
        sorted_ts = sorted(engage_ts_list)

        # 전투 간 간격 (inter-fight gap)
        if len(sorted_ts) > 1:
            gaps = [(sorted_ts[i + 1] - sorted_ts[i]) / 1000
                    for i in range(len(sorted_ts) - 1)]
            row["avg_gap_s"] = float(np.mean(gaps))
            row["min_gap_s"] = float(np.min(gaps))
            row["max_gap_s"] = float(np.max(gaps))
            row["std_gap_s"] = float(np.std(gaps))
        else:
            row["avg_gap_s"] = row["min_gap_s"] = row["max_gap_s"] = row["std_gap_s"] = np.nan

        # 라벨 윈도우 겹침 검사
        horizon_ends = sorted([(f["engage_ts"], f["horizon_end_ts"]) for f in fights])
        n_overlaps = 0
        for i in range(len(horizon_ends) - 1):
            if horizon_ends[i][1] > horizon_ends[i + 1][0]:
                n_overlaps += 1
        row["label_window_overlaps"] = n_overlaps

        # 유형별 집계
        types = Counter(f.get("fight_type", "unknown") for f in fights)
        for ft in FIGHT_TYPES_ORDERED:
            row[f"n_type_{ft}"] = types.get(ft, 0)
        row["n_type_unknown"] = sum(v for k, v in types.items()
                                    if k not in FIGHT_TYPES_ORDERED)

        # 승리 집계
        winners = Counter(f["outcome"]["winner"] for f in fights)
        row["n_win_blue"] = winners.get("blue", 0)
        row["n_win_red"] = winners.get("red", 0)
        row["n_draw"] = winners.get("draw", 0)

        # 매치 전체 킬/골드
        row["total_kills"] = sum(f["outcome"]["total_kills"] for f in fights)
        row["avg_kills"] = float(np.mean([f["outcome"]["total_kills"] for f in fights]))
        row["avg_gold_diff"] = float(np.mean([f["outcome"]["gold_diff"] for f in fights]))
        row["avg_importance"] = float(np.mean([f["importance_score"] for f in fights]))
        row["avg_participants"] = float(np.mean([f["det_cluster_participants"] for f in fights]))
        row["avg_duration_ms"] = float(np.mean([f["det_cluster_duration_ms"] for f in fights]))

        # 시간 분포 (빈: 0-5, 5-10, ..., 35-40, 40+)
        t_minutes = [f["t_engage"] for f in fights]
        for lo in range(0, 45, 5):
            hi = lo + 5
            row[f"fights_{lo}_{hi}min"] = sum(1 for t in t_minutes if lo <= t < hi)
        row["fights_45plus_min"] = sum(1 for t in t_minutes if t >= 45)

    else:
        # 팀파이트가 없는 매치
        row["avg_gap_s"] = row["min_gap_s"] = row["max_gap_s"] = row["std_gap_s"] = np.nan
        row["label_window_overlaps"] = 0
        for ft in FIGHT_TYPES_ORDERED:
            row[f"n_type_{ft}"] = 0
        row["n_type_unknown"] = 0
        row["n_win_blue"] = row["n_win_red"] = row["n_draw"] = 0
        row["total_kills"] = 0
        row["avg_kills"] = row["avg_gold_diff"] = row["avg_importance"] = np.nan
        row["avg_participants"] = row["avg_duration_ms"] = np.nan
        for lo in range(0, 45, 5):
            hi = lo + 5
            row[f"fights_{lo}_{hi}min"] = 0
        row["fights_45plus_min"] = 0

    return row


# ============================================================================
# §4  Fight-Level Extractor
# ============================================================================

def extract_fight_level(data: dict) -> List[dict]:
    """
    단일 JSON에서 모든 개별 팀파이트의 상세 통계를 추출합니다.

    Returns:
        List[dict]: 각 팀파이트에 대한 피처 딕셔너리 리스트
    """
    match_id = data.get("match_id", "")
    patch = data.get("patch", "")
    fights = data.get("fights_kept", [])
    rows = []

    for idx, f in enumerate(fights):
        oc = f.get("outcome", {})
        pfo = f.get("post_fight_outcome", {})
        pe_list = f.get("player_engagement", [])
        viz = f.get("visualization", {})

        row = {
            # --- 매치 메타데이터 ---
            "match_id": match_id,
            "patch": patch,
            "fight_idx": idx,

            # --- 시간 정보 ---
            "engage_ts": f.get("engage_ts", 0),
            "t_engage_min": f.get("t_engage", 0),
            "first_kill_ts": f.get("first_kill_ts", 0),
            "last_kill_ts": f.get("last_kill_ts", 0),
            "horizon_end_ts": f.get("horizon_end_ts", 0),

            # 파생 시간 피처 (derived temporal features)
            # Δt_kill = t_first_kill - t_engage (킬 지연 시간)
            "kill_delay_ms": f.get("first_kill_ts", 0) - f.get("engage_ts", 0),
            # Δt_fight = t_last_kill - t_first_kill (전투 지속 시간)
            "fight_duration_ms": f.get("last_kill_ts", 0) - f.get("first_kill_ts", 0),
            # Δt_horizon = t_horizon - t_engage (라벨 윈도우 크기)
            "horizon_duration_ms": f.get("horizon_end_ts", 0) - f.get("engage_ts", 0),

            # --- 공간 정보 ---
            "centroid_x": f.get("centroid_x", 0),
            "centroid_y": f.get("centroid_y", 0),

            # --- 탐지 메타데이터 ---
            "n_segments": f.get("n_segments", 0),
            "det_step_ms": f.get("det_step_ms", 0),
            "det_prox_pairs": f.get("det_prox_pairs", 0),
            "det_min_dist_mean": f.get("det_min_dist_mean", 0),
            "det_anchor": f.get("det_anchor", 0),
            "det_backtracked": f.get("det_backtracked", 0),
            "det_backtrack_reliable": f.get("det_backtrack_reliable", 0),
            "det_damage_norm": f.get("det_damage_norm", 0),
            "det_summoner_spells": f.get("det_summoner_spells", 0),
            "det_signal_ok": f.get("det_signal_ok", 0),
            "det_score_ok": f.get("det_score_ok", 0),
            "det_event_score": f.get("det_event_score", 0),
            "det_event_count": f.get("det_event_count", 0),
            "det_kill_count_window": f.get("det_kill_count_window", 0),
            "det_combat_signal_ok": f.get("det_combat_signal_ok", 0),
            "det_cluster_participants": f.get("det_cluster_participants", 0),
            "det_cluster_blue": f.get("det_cluster_blue", 0),
            "det_cluster_red": f.get("det_cluster_red", 0),
            "det_cluster_duration_ms": f.get("det_cluster_duration_ms", 0),
            "det_interaction_count": f.get("det_interaction_count", 0),

            # --- 분류 ---
            "fight_type": f.get("fight_type", "unknown"),
            "importance_score": f.get("importance_score", 0),

            # --- 결과 (Outcome) ---
            "winner": oc.get("winner", ""),
            "blue_kills": oc.get("blue_kills", 0),
            "red_kills": oc.get("red_kills", 0),
            "blue_deaths": oc.get("blue_deaths", 0),
            "red_deaths": oc.get("red_deaths", 0),
            "kill_diff": oc.get("kill_diff", 0),
            "total_kills": oc.get("total_kills", 0),
            "assists_blue": safe_get(oc, "assists", "blue", default=0),
            "assists_red": safe_get(oc, "assists", "red", default=0),
            "blue_unique_deaths": oc.get("blue_unique_deaths", 0),
            "red_unique_deaths": oc.get("red_unique_deaths", 0),
            "blue_survivors": oc.get("blue_survivors", 0),
            "red_survivors": oc.get("red_survivors", 0),
            "blue_alive_end": oc.get("blue_alive_end", 0),
            "red_alive_end": oc.get("red_alive_end", 0),
            "gold_blue_delta": oc.get("gold_blue_delta", 0),
            "gold_red_delta": oc.get("gold_red_delta", 0),
            "gold_diff": oc.get("gold_diff", 0),

            # --- 구조물 / 오브젝트 결과 ---
            "tower_blue": oc.get("tower_blue", 0),
            "tower_red": oc.get("tower_red", 0),
            "tower_diff": oc.get("tower_diff", 0),
            "plate_blue": oc.get("plate_blue", 0),
            "plate_red": oc.get("plate_red", 0),
            "plate_diff": oc.get("plate_diff", 0),
            "inhib_blue": oc.get("inhib_blue", 0),
            "inhib_red": oc.get("inhib_red", 0),
            "inhib_diff": oc.get("inhib_diff", 0),
            "objective_blue": oc.get("objective_blue", 0),
            "objective_red": oc.get("objective_red", 0),
            "objective_diff": oc.get("objective_diff", 0),
        }

        # 오브젝트 유형별 세분화
        obj_by_type = oc.get("objective_by_type", {})
        for otype in OBJECTIVE_TYPES:
            o = obj_by_type.get(otype, {})
            row[f"obj_{otype}_blue"] = o.get("blue", 0)
            row[f"obj_{otype}_red"] = o.get("red", 0)
            row[f"obj_{otype}_diff"] = o.get("diff", 0)

        # --- 포스트-파이트 결과 (Post-Fight Outcome) ---
        row["post_obj_blue"] = pfo.get("post_obj_blue", 0)
        row["post_obj_red"] = pfo.get("post_obj_red", 0)
        row["post_obj_diff"] = pfo.get("post_obj_diff", 0)
        row["post_tower_blue"] = pfo.get("post_tower_blue", 0)
        row["post_tower_red"] = pfo.get("post_tower_red", 0)
        row["post_tower_diff"] = pfo.get("post_tower_diff", 0)
        row["post_gold_diff"] = pfo.get("post_gold_diff", 0)

        # --- 플레이어 교전 참여 통계 (Player Engagement) ---
        if pe_list:
            blue_pe = [p for p in pe_list if p.get("team") == "blue"]
            red_pe = [p for p in pe_list if p.get("team") == "red"]

            row["pe_avg_engagement_all"] = float(np.mean(
                [p["engagement_ratio"] for p in pe_list]))
            row["pe_avg_engagement_blue"] = float(np.mean(
                [p["engagement_ratio"] for p in blue_pe])) if blue_pe else 0.0
            row["pe_avg_engagement_red"] = float(np.mean(
                [p["engagement_ratio"] for p in red_pe])) if red_pe else 0.0
            row["pe_avg_dist_enemy_all"] = float(np.mean(
                [p["avg_dist_to_enemy"] for p in pe_list]))
            row["pe_max_engagement"] = float(max(
                p["engagement_ratio"] for p in pe_list))
            row["pe_min_engagement"] = float(min(
                p["engagement_ratio"] for p in pe_list))
            row["pe_n_fully_engaged"] = sum(
                1 for p in pe_list if p["engagement_ratio"] >= 0.8)
        else:
            row["pe_avg_engagement_all"] = np.nan
            row["pe_avg_engagement_blue"] = np.nan
            row["pe_avg_engagement_red"] = np.nan
            row["pe_avg_dist_enemy_all"] = np.nan
            row["pe_max_engagement"] = np.nan
            row["pe_min_engagement"] = np.nan
            row["pe_n_fully_engaged"] = 0

        # --- 시각화 기반 메타 피처 ---
        kill_markers = viz.get("kill_markers", [])
        row["viz_n_kill_markers"] = len(kill_markers)
        timeline = viz.get("engagement_timeline", [])
        if timeline:
            intensities = [e.get("normalized", 0) for e in timeline]
            row["viz_peak_intensity"] = float(max(intensities))
            row["viz_avg_intensity"] = float(np.mean(intensities))
        else:
            row["viz_peak_intensity"] = 0.0
            row["viz_avg_intensity"] = 0.0

        rows.append(row)

    return rows


# ============================================================================
# §5  Aggregate Statistics for Paper Tables
# ============================================================================

def compute_paper_statistics(match_rows: List[dict],
                             fight_rows: List[dict]) -> dict:
    """
    논문에 게재할 통계 테이블을 생성합니다.

    Returns:
        dict: 구조화된 통계 요약
    """
    stats = {}
    N_matches = len(match_rows)
    N_fights = len(fight_rows)

    # -----------------------------------------------------------------------
    # Table 1: Dataset Overview
    # -----------------------------------------------------------------------
    n_kept_list = [m["n_fights_kept"] for m in match_rows]
    patches = list(set(m["patch"] for m in match_rows))

    stats["table1_overview"] = {
        "n_matches": N_matches,
        "n_fights_total": N_fights,
        "patches": patches,
        "server_tier": "KR Master+",
        "fights_per_match_mean": float(np.mean(n_kept_list)),
        "fights_per_match_std": float(np.std(n_kept_list)),
        "fights_per_match_median": float(np.median(n_kept_list)),
        "fights_per_match_min": int(np.min(n_kept_list)),
        "fights_per_match_max": int(np.max(n_kept_list)),
        "fights_per_match_ci95": compute_ci95(n_kept_list),
        "matches_with_zero": sum(1 for x in n_kept_list if x == 0),
    }

    # -----------------------------------------------------------------------
    # Table 2: Detection Funnel
    # -----------------------------------------------------------------------
    total_candidates = sum(m["det_candidates"] for m in match_rows)
    total_clusters = sum(m["det_clusters_total"] for m in match_rows)
    total_accepted = sum(m["det_accepted"] for m in match_rows)
    total_kept = sum(m["n_fights_kept"] for m in match_rows)

    # 거부 사유 합계
    rej_keys = [k for k in match_rows[0].keys() if k.startswith("rej_")]
    rej_totals = {k: sum(m[k] for m in match_rows) for k in rej_keys}
    total_rejected = sum(rej_totals.values())

    stats["table2_funnel"] = {
        "candidates": total_candidates,
        "clusters_total": total_clusters,
        "clusters_accepted": sum(m["det_clusters_accepted"] for m in match_rows),
        "accepted": total_accepted,
        "kept": total_kept,
        "acceptance_rate": total_accepted / total_candidates if total_candidates else 0,
        "retention_rate": total_kept / total_accepted if total_accepted else 0,
        "overall_rate": total_kept / total_candidates if total_candidates else 0,
        "rejection_breakdown": rej_totals,
        "total_rejected": total_rejected,
        "rejection_proportions": {
            k: v / total_rejected if total_rejected else 0
            for k, v in rej_totals.items()
        },
    }

    # -----------------------------------------------------------------------
    # Table 3: Fight Type Characteristics
    # -----------------------------------------------------------------------
    type_stats = {}
    for ft in FIGHT_TYPES_ORDERED + ["unknown"]:
        subset = [f for f in fight_rows if f["fight_type"] == ft]
        if not subset:
            continue
        n = len(subset)
        type_stats[ft] = {
            "count": n,
            "proportion": n / N_fights,
            "participants_mean": float(np.mean([f["det_cluster_participants"] for f in subset])),
            "participants_std": float(np.std([f["det_cluster_participants"] for f in subset])),
            "kills_mean": float(np.mean([f["total_kills"] for f in subset])),
            "kills_std": float(np.std([f["total_kills"] for f in subset])),
            "duration_ms_mean": float(np.mean([f["det_cluster_duration_ms"] for f in subset])),
            "duration_ms_std": float(np.std([f["det_cluster_duration_ms"] for f in subset])),
            "importance_mean": float(np.mean([f["importance_score"] for f in subset])),
            "importance_std": float(np.std([f["importance_score"] for f in subset])),
            "gold_diff_mean": float(np.mean([f["gold_diff"] for f in subset])),
            "gold_diff_std": float(np.std([f["gold_diff"] for f in subset])),
            "blue_win_rate": sum(1 for f in subset if f["winner"] == "blue") / n,
            "red_win_rate": sum(1 for f in subset if f["winner"] == "red") / n,
            "draw_rate": sum(1 for f in subset if f["winner"] == "draw") / n,
            "engage_time_mean": float(np.mean([f["t_engage_min"] for f in subset])),
            "engage_time_std": float(np.std([f["t_engage_min"] for f in subset])),
        }

    stats["table3_fight_types"] = type_stats

    # -----------------------------------------------------------------------
    # Table 4: Outcome & Gold Statistics
    # -----------------------------------------------------------------------
    all_gold = [f["gold_diff"] for f in fight_rows]
    all_kills = [f["total_kills"] for f in fight_rows]
    all_importance = [f["importance_score"] for f in fight_rows]

    winners = Counter(f["winner"] for f in fight_rows)
    stats["table4_outcomes"] = {
        "blue_wins": winners.get("blue", 0),
        "red_wins": winners.get("red", 0),
        "draws": winners.get("draw", 0),
        "blue_rate": winners.get("blue", 0) / N_fights,
        "red_rate": winners.get("red", 0) / N_fights,
        "draw_rate": winners.get("draw", 0) / N_fights,
        "gold_diff_mean": float(np.mean(all_gold)),
        "gold_diff_std": float(np.std(all_gold)),
        "gold_diff_median": float(np.median(all_gold)),
        "gold_diff_ci95": compute_ci95(all_gold),
        "kills_mean": float(np.mean(all_kills)),
        "kills_std": float(np.std(all_kills)),
        "kills_ci95": compute_ci95(all_kills),
        "importance_mean": float(np.mean(all_importance)),
        "importance_std": float(np.std(all_importance)),
        "importance_ci95": compute_ci95(all_importance),
    }

    # -----------------------------------------------------------------------
    # Table 5: Temporal Distribution
    # -----------------------------------------------------------------------
    time_bins = {}
    for lo in range(0, 50, 5):
        hi = lo + 5
        count = sum(1 for f in fight_rows if lo <= f["t_engage_min"] < hi)
        time_bins[f"{lo}-{hi}"] = count
    time_bins["50+"] = sum(1 for f in fight_rows if f["t_engage_min"] >= 50)

    stats["table5_temporal"] = {
        "bins": time_bins,
        "engage_time_mean": float(np.mean([f["t_engage_min"] for f in fight_rows])),
        "engage_time_std": float(np.std([f["t_engage_min"] for f in fight_rows])),
        "engage_time_median": float(np.median([f["t_engage_min"] for f in fight_rows])),
    }

    # -----------------------------------------------------------------------
    # Table 6: Quality & Integrity Metrics
    # -----------------------------------------------------------------------
    overlap_total = sum(m["label_window_overlaps"] for m in match_rows)
    gaps_all = [m["avg_gap_s"] for m in match_rows if not np.isnan(m.get("avg_gap_s", np.nan))]

    quality_flags = {}
    for flag in ["det_signal_ok", "det_score_ok", "det_combat_signal_ok",
                 "det_backtracked", "det_backtrack_reliable"]:
        vals = [f.get(flag, 0) for f in fight_rows]
        quality_flags[flag] = float(np.mean(vals)) if vals else 0.0

    stats["table6_quality"] = {
        "label_window_overlaps_total": overlap_total,
        "label_window_overlap_rate": overlap_total / N_fights if N_fights else 0,
        "inter_fight_gap_mean": float(np.mean(gaps_all)) if gaps_all else 0,
        "inter_fight_gap_std": float(np.std(gaps_all)) if gaps_all else 0,
        "inter_fight_gap_min": float(np.min(gaps_all)) if gaps_all else 0,
        "inter_fight_gap_max": float(np.max(gaps_all)) if gaps_all else 0,
        "quality_flags_pass_rate": quality_flags,
        "n_errors_total": sum(m.get("n_errors", 0) for m in match_rows),
        "postmerge_conflicts_total": sum(m.get("postmerge_conflicts", 0) for m in match_rows),
    }

    # -----------------------------------------------------------------------
    # Table 7: Player Engagement Statistics
    # -----------------------------------------------------------------------
    pe_all = [f["pe_avg_engagement_all"] for f in fight_rows
              if not np.isnan(f.get("pe_avg_engagement_all", np.nan))]
    pe_blue = [f["pe_avg_engagement_blue"] for f in fight_rows
               if not np.isnan(f.get("pe_avg_engagement_blue", np.nan))]
    pe_red = [f["pe_avg_engagement_red"] for f in fight_rows
              if not np.isnan(f.get("pe_avg_engagement_red", np.nan))]

    stats["table7_engagement"] = {
        "avg_engagement_all_mean": float(np.mean(pe_all)) if pe_all else 0,
        "avg_engagement_all_std": float(np.std(pe_all)) if pe_all else 0,
        "avg_engagement_blue_mean": float(np.mean(pe_blue)) if pe_blue else 0,
        "avg_engagement_red_mean": float(np.mean(pe_red)) if pe_red else 0,
        "fully_engaged_mean": float(np.mean(
            [f["pe_n_fully_engaged"] for f in fight_rows])),
    }

    # -----------------------------------------------------------------------
    # Table 8: Spatial Statistics
    # -----------------------------------------------------------------------
    cx = [f["centroid_x"] for f in fight_rows]
    cy = [f["centroid_y"] for f in fight_rows]
    stats["table8_spatial"] = {
        "centroid_x_mean": float(np.mean(cx)),
        "centroid_x_std": float(np.std(cx)),
        "centroid_y_mean": float(np.mean(cy)),
        "centroid_y_std": float(np.std(cy)),
        "prox_pairs_mean": float(np.mean(
            [f["det_prox_pairs"] for f in fight_rows])),
        "prox_pairs_std": float(np.std(
            [f["det_prox_pairs"] for f in fight_rows])),
    }

    # -----------------------------------------------------------------------
    # Normality Test (if scipy available)
    # -----------------------------------------------------------------------
    if HAS_SCIPY and len(all_gold) >= 20:
        _, p_gold = sp_stats.shapiro(np.random.choice(all_gold, min(5000, len(all_gold)), replace=False))
        _, p_kills = sp_stats.shapiro(np.random.choice(all_kills, min(5000, len(all_kills)), replace=False))
        stats["normality_tests"] = {
            "gold_diff_shapiro_p": float(p_gold),
            "kills_shapiro_p": float(p_kills),
        }

    return stats


# ============================================================================
# §6  LaTeX Table Generator
# ============================================================================

def generate_latex_tables(stats: dict) -> str:
    """
    IEEE CoG 형식의 LaTeX 테이블을 생성합니다.
    """
    lines = []
    lines.append("% ====================================================")
    lines.append("% Auto-generated Paper Tables")
    lines.append("% ====================================================")
    lines.append("")

    # --- Table 1: Dataset Overview ---
    t1 = stats["table1_overview"]
    ci = t1["fights_per_match_ci95"]
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Dataset Overview}")
    lines.append(r"\label{tab:dataset_overview}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} \\")
    lines.append(r"\midrule")
    lines.append(f"Server / Tier & {t1['server_tier']} \\\\")
    lines.append(f"Patch & {', '.join(str(p) for p in t1['patches'])} \\\\")
    lines.append(f"Total matches & {t1['n_matches']:,} \\\\")
    lines.append(f"Total teamfights & {t1['n_fights_total']:,} \\\\")
    lines.append(f"Fights/match ($\\mu \\pm \\sigma$) & "
                 f"${t1['fights_per_match_mean']:.2f} \\pm {t1['fights_per_match_std']:.2f}$ \\\\")
    lines.append(f"Fights/match (median) & {t1['fights_per_match_median']:.1f} \\\\")
    lines.append(f"Fights/match (range) & [{t1['fights_per_match_min']}, {t1['fights_per_match_max']}] \\\\")
    lines.append(f"95\\% CI & [{ci[0]:.2f}, {ci[1]:.2f}] \\\\")
    lines.append(f"Matches w/ 0 fights & {t1['matches_with_zero']} "
                 f"({t1['matches_with_zero'] / t1['n_matches'] * 100:.1f}\\%) \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 2: Detection Funnel ---
    t2 = stats["table2_funnel"]
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Detection Pipeline Funnel}")
    lines.append(r"\label{tab:detection_funnel}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Stage} & \textbf{Count} & \textbf{Rate} & \textbf{Cum. Rate} \\")
    lines.append(r"\midrule")
    lines.append(f"Candidate timesteps & {t2['candidates']:,} & 100.0\\% & 100.0\\% \\\\")
    lines.append(f"Spatial clusters & {t2['clusters_total']:,} & "
                 f"{t2['clusters_total'] / t2['candidates'] * 100:.1f}\\% & "
                 f"{t2['clusters_total'] / t2['candidates'] * 100:.1f}\\% \\\\")
    lines.append(f"Accepted clusters & {t2['accepted']:,} & "
                 f"{t2['acceptance_rate'] * 100:.1f}\\% & "
                 f"{t2['acceptance_rate'] * 100:.1f}\\% \\\\")
    lines.append(f"Final teamfights & {t2['kept']:,} & "
                 f"{t2['retention_rate'] * 100:.1f}\\% & "
                 f"{t2['overall_rate'] * 100:.1f}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 2b: Rejection Reasons ---
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Rejection Reasons}")
    lines.append(r"\label{tab:rejection_reasons}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Reason} & \textbf{Count} & \textbf{Proportion} \\")
    lines.append(r"\midrule")
    for k, v in sorted(t2["rejection_breakdown"].items(), key=lambda x: -x[1]):
        if v > 0:
            prop = t2["rejection_proportions"][k]
            label = k.replace("rej_", "").replace("_", " ").title()
            lines.append(f"{label} & {v:,} & {prop * 100:.1f}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 3: Fight Type Characteristics ---
    t3 = stats["table3_fight_types"]
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\caption{Teamfight Type Characteristics}")
    lines.append(r"\label{tab:fight_types}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Type} & \textbf{N} & \textbf{\%} & "
                 r"\textbf{Participants} & \textbf{Kills} & "
                 r"\textbf{Duration (s)} & \textbf{Importance} & "
                 r"\textbf{Gold $\Delta$} & \textbf{Blue WR} \\")
    lines.append(r"\midrule")
    for ft in FIGHT_TYPES_ORDERED:
        if ft not in t3:
            continue
        s = t3[ft]
        dur_s = s["duration_ms_mean"] / 1000
        dur_std = s["duration_ms_std"] / 1000
        lines.append(
            f"{ft.replace('_', ' ')} & {s['count']:,} & {s['proportion'] * 100:.1f} & "
            f"${s['participants_mean']:.1f} \\pm {s['participants_std']:.1f}$ & "
            f"${s['kills_mean']:.1f} \\pm {s['kills_std']:.1f}$ & "
            f"${dur_s:.1f} \\pm {dur_std:.1f}$ & "
            f"${s['importance_mean']:.1f} \\pm {s['importance_std']:.1f}$ & "
            f"${s['gold_diff_mean']:.0f} \\pm {s['gold_diff_std']:.0f}$ & "
            f"{s['blue_win_rate'] * 100:.1f}\\% \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    # --- Table 4: Quality Metrics ---
    t6 = stats["table6_quality"]
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Detection Quality and Temporal Integrity}")
    lines.append(r"\label{tab:quality_metrics}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} \\")
    lines.append(r"\midrule")
    lines.append(f"Label window overlaps & {t6['label_window_overlaps_total']} \\\\")
    lines.append(f"Label overlap rate & {t6['label_window_overlap_rate'] * 100:.2f}\\% \\\\")
    lines.append(f"Inter-fight gap ($\\mu \\pm \\sigma$) & "
                 f"${t6['inter_fight_gap_mean']:.1f} \\pm {t6['inter_fight_gap_std']:.1f}$s \\\\")
    lines.append(f"Min inter-fight gap & {t6['inter_fight_gap_min']:.1f}s \\\\")
    for flag, rate in t6["quality_flags_pass_rate"].items():
        label = flag.replace("det_", "").replace("_", " ").title()
        lines.append(f"{label} pass rate & {rate * 100:.1f}\\% \\\\")
    lines.append(f"Processing errors & {t6['n_errors_total']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================================
# §7  Markdown Table Generator
# ============================================================================

def generate_markdown_tables(stats: dict) -> str:
    """Markdown 형식의 논문 테이블을 생성합니다."""
    lines = []

    # Table 1
    t1 = stats["table1_overview"]
    ci = t1["fights_per_match_ci95"]
    lines.append("## Table 1: Dataset Overview")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Server / Tier | {t1['server_tier']} |")
    lines.append(f"| Patch | {', '.join(str(p) for p in t1['patches'])} |")
    lines.append(f"| Total matches | {t1['n_matches']:,} |")
    lines.append(f"| Total teamfights | {t1['n_fights_total']:,} |")
    lines.append(f"| Fights/match (μ ± σ) | {t1['fights_per_match_mean']:.2f} ± {t1['fights_per_match_std']:.2f} |")
    lines.append(f"| Fights/match (median) | {t1['fights_per_match_median']:.1f} |")
    lines.append(f"| Fights/match [min, max] | [{t1['fights_per_match_min']}, {t1['fights_per_match_max']}] |")
    lines.append(f"| 95% CI | [{ci[0]:.2f}, {ci[1]:.2f}] |")
    lines.append(
        f"| Matches w/ 0 fights | {t1['matches_with_zero']} ({t1['matches_with_zero'] / t1['n_matches'] * 100:.1f}%) |")
    lines.append("")

    # Table 2
    t2 = stats["table2_funnel"]
    lines.append("## Table 2: Detection Pipeline Funnel")
    lines.append("| Stage | Count | Stage Rate | Cumulative |")
    lines.append("|-------|------:|----------:|----------:|")
    lines.append(f"| Candidate timesteps | {t2['candidates']:,} | 100.0% | 100.0% |")
    lines.append(f"| Spatial clusters | {t2['clusters_total']:,} | "
                 f"{t2['clusters_total'] / t2['candidates'] * 100:.1f}% | "
                 f"{t2['clusters_total'] / t2['candidates'] * 100:.1f}% |")
    lines.append(f"| Accepted | {t2['accepted']:,} | "
                 f"{t2['acceptance_rate'] * 100:.1f}% | "
                 f"{t2['acceptance_rate'] * 100:.1f}% |")
    lines.append(f"| **Final fights** | **{t2['kept']:,}** | "
                 f"{t2['retention_rate'] * 100:.1f}% | "
                 f"**{t2['overall_rate'] * 100:.1f}%** |")
    lines.append("")

    # Table 2b: Rejections
    lines.append("## Table 2b: Rejection Reasons")
    lines.append("| Reason | Count | Proportion |")
    lines.append("|--------|------:|-----------:|")
    for k, v in sorted(t2["rejection_breakdown"].items(), key=lambda x: -x[1]):
        if v > 0:
            label = k.replace("rej_", "").replace("_", " ").title()
            lines.append(f"| {label} | {v:,} | {t2['rejection_proportions'][k] * 100:.1f}% |")
    lines.append("")

    # Table 3: Fight Types
    t3 = stats["table3_fight_types"]
    lines.append("## Table 3: Fight Type Characteristics")
    lines.append("| Type | N | % | Participants | Kills | Duration(s) | Importance | Gold Δ | Blue WR |")
    lines.append("|------|--:|--:|-----------:|------:|----------:|---------:|------:|-------:|")
    for ft in FIGHT_TYPES_ORDERED:
        if ft not in t3:
            continue
        s = t3[ft]
        dur_s = s["duration_ms_mean"] / 1000
        lines.append(
            f"| {ft} | {s['count']:,} | {s['proportion'] * 100:.1f} | "
            f"{s['participants_mean']:.1f}±{s['participants_std']:.1f} | "
            f"{s['kills_mean']:.1f}±{s['kills_std']:.1f} | "
            f"{dur_s:.1f} | "
            f"{s['importance_mean']:.1f}±{s['importance_std']:.1f} | "
            f"{s['gold_diff_mean']:.0f}±{s['gold_diff_std']:.0f} | "
            f"{s['blue_win_rate'] * 100:.1f}% |"
        )
    lines.append("")

    # Table 4: Outcomes
    t4 = stats["table4_outcomes"]
    lines.append("## Table 4: Outcome Statistics")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Blue wins | {t4['blue_wins']:,} ({t4['blue_rate'] * 100:.1f}%) |")
    lines.append(f"| Red wins | {t4['red_wins']:,} ({t4['red_rate'] * 100:.1f}%) |")
    lines.append(f"| Draws | {t4['draws']:,} ({t4['draw_rate'] * 100:.1f}%) |")
    lines.append(f"| Gold diff (μ ± σ) | {t4['gold_diff_mean']:.1f} ± {t4['gold_diff_std']:.1f} |")
    lines.append(f"| Gold diff 95% CI | [{t4['gold_diff_ci95'][0]:.1f}, {t4['gold_diff_ci95'][1]:.1f}] |")
    lines.append(f"| Total kills (μ ± σ) | {t4['kills_mean']:.2f} ± {t4['kills_std']:.2f} |")
    lines.append(f"| Importance (μ ± σ) | {t4['importance_mean']:.1f} ± {t4['importance_std']:.1f} |")
    lines.append("")

    # Table 5: Temporal
    t5 = stats["table5_temporal"]
    lines.append("## Table 5: Temporal Distribution")
    lines.append("| Time Bin | Count | Proportion |")
    lines.append("|----------|------:|-----------:|")
    total = sum(t5["bins"].values())
    for bin_name, count in t5["bins"].items():
        lines.append(f"| {bin_name} min | {count:,} | {count / total * 100:.1f}% |")
    lines.append(f"| **Mean engage time** | **{t5['engage_time_mean']:.1f} min** | σ = {t5['engage_time_std']:.1f} |")
    lines.append("")

    # Table 6: Quality
    t6 = stats["table6_quality"]
    lines.append("## Table 6: Quality & Integrity Metrics")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Label window overlaps | {t6['label_window_overlaps_total']} |")
    lines.append(f"| Label overlap rate | {t6['label_window_overlap_rate'] * 100:.3f}% |")
    lines.append(f"| Inter-fight gap (μ ± σ) | {t6['inter_fight_gap_mean']:.1f} ± {t6['inter_fight_gap_std']:.1f}s |")
    lines.append(f"| Min inter-fight gap | {t6['inter_fight_gap_min']:.1f}s |")
    for flag, rate in t6["quality_flags_pass_rate"].items():
        label = flag.replace("det_", "").replace("_", " ").title()
        lines.append(f"| {label} pass rate | {rate * 100:.1f}% |")
    lines.append(f"| Processing errors | {t6['n_errors_total']} |")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# §8  CSV Writer (fallback without pandas)
# ============================================================================

def write_csv_manual(rows: List[dict], filepath: str):
    """pandas 없이 CSV를 작성합니다."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(filepath, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float) and np.isnan(v):
                    vals.append("")
                elif isinstance(v, str) and "," in v:
                    vals.append(f'"{v}"')
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


# ============================================================================
# §9  Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract paper-ready statistics from teamfight JSON files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing *_fights.json files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for statistics files")
    parser.add_argument("--pattern", type=str, default="*_fights.json",
                        help="Glob pattern for JSON files (default: *_fights.json)")
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Number of bootstrap samples for CI (default: 10000)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "extraction_log.txt")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger(__name__)

    # Discover files
    json_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    logger.info(f"Found {len(json_files)} JSON files in {args.input_dir}")

    if not json_files:
        logger.error("No JSON files found. Check --input_dir and --pattern.")
        sys.exit(1)

    # Process files
    match_rows = []
    fight_rows = []
    errors = []
    t0 = time.time()

    for i, jf in enumerate(json_files):
        try:
            with open(jf, "r") as f:
                data = json.load(f)

            # Schema validation
            missing = EXPECTED_TOP_KEYS - set(data.keys())
            if missing:
                logger.warning(f"{jf}: Missing keys {missing}")

            # Extract
            m_row = extract_match_level(data)
            f_rows = extract_fight_level(data)

            match_rows.append(m_row)
            fight_rows.extend(f_rows)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(json_files) - i - 1) / rate
                logger.info(f"Processed {i + 1}/{len(json_files)} "
                            f"({rate:.0f} files/s, ETA {eta:.0f}s)")

        except Exception as e:
            errors.append((jf, str(e)))
            logger.error(f"Error processing {jf}: {e}")

    elapsed = time.time() - t0
    logger.info(f"Extraction complete: {len(match_rows)} matches, "
                f"{len(fight_rows)} fights in {elapsed:.1f}s")
    if errors:
        logger.warning(f"{len(errors)} files had errors")

    # -----------------------------------------------------------------------
    # Write Match-Level CSV
    # -----------------------------------------------------------------------
    match_csv = os.path.join(args.output_dir, "match_level_stats.csv")
    if HAS_PANDAS:
        pd.DataFrame(match_rows).to_csv(match_csv, index=False)
    else:
        write_csv_manual(match_rows, match_csv)
    logger.info(f"Wrote match-level CSV: {match_csv} ({len(match_rows)} rows)")

    # -----------------------------------------------------------------------
    # Write Fight-Level CSV
    # -----------------------------------------------------------------------
    fight_csv = os.path.join(args.output_dir, "fight_level_stats.csv")
    if HAS_PANDAS:
        pd.DataFrame(fight_rows).to_csv(fight_csv, index=False)
    else:
        write_csv_manual(fight_rows, fight_csv)
    logger.info(f"Wrote fight-level CSV: {fight_csv} ({len(fight_rows)} rows)")

    # -----------------------------------------------------------------------
    # Compute Paper Statistics
    # -----------------------------------------------------------------------
    logger.info("Computing aggregate statistics for paper tables...")
    paper_stats = compute_paper_statistics(match_rows, fight_rows)

    # Write JSON summary
    summary_json = os.path.join(args.output_dir, "paper_summary.json")
    with open(summary_json, "w") as f:
        json.dump(paper_stats, f, indent=2, default=str)
    logger.info(f"Wrote paper summary JSON: {summary_json}")

    # -----------------------------------------------------------------------
    # Generate LaTeX Tables
    # -----------------------------------------------------------------------
    latex_tables = generate_latex_tables(paper_stats)
    latex_path = os.path.join(args.output_dir, "paper_tables.tex")
    with open(latex_path, "w") as f:
        f.write(latex_tables)
    logger.info(f"Wrote LaTeX tables: {latex_path}")

    # -----------------------------------------------------------------------
    # Generate Markdown Tables
    # -----------------------------------------------------------------------
    md_tables = generate_markdown_tables(paper_stats)
    md_path = os.path.join(args.output_dir, "paper_tables.md")
    with open(md_path, "w") as f:
        f.write(md_tables)
    logger.info(f"Wrote Markdown tables: {md_path}")

    # -----------------------------------------------------------------------
    # Console Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PAPER STATISTICS — EXTRACTION COMPLETE")
    print("=" * 70)
    t1 = paper_stats["table1_overview"]
    t2 = paper_stats["table2_funnel"]
    t4 = paper_stats["table4_outcomes"]
    t6 = paper_stats["table6_quality"]

    print(f"\n  Matches:           {t1['n_matches']:>10,}")
    print(f"  Teamfights:        {t1['n_fights_total']:>10,}")
    print(f"  Fights/match:      {t1['fights_per_match_mean']:>10.2f} ± "
          f"{t1['fights_per_match_std']:.2f}")
    print(f"  Candidates:        {t2['candidates']:>10,}")
    print(f"  Acceptance rate:   {t2['acceptance_rate'] * 100:>10.1f}%")
    print(f"  Retention rate:    {t2['retention_rate'] * 100:>10.1f}%")
    print(f"  Blue WR:           {t4['blue_rate'] * 100:>10.1f}%")
    print(f"  Red WR:            {t4['red_rate'] * 100:>10.1f}%")
    print(f"  Draw rate:         {t4['draw_rate'] * 100:>10.1f}%")
    print(f"  Label overlaps:    {t6['label_window_overlaps_total']:>10}")
    print(f"  Errors:            {t6['n_errors_total']:>10}")

    print(f"\n  Output directory:  {args.output_dir}")
    print(f"  Files generated:")
    print(f"    - match_level_stats.csv  ({len(match_rows):,} rows)")
    print(f"    - fight_level_stats.csv  ({len(fight_rows):,} rows)")
    print(f"    - paper_summary.json")
    print(f"    - paper_tables.tex")
    print(f"    - paper_tables.md")
    print("=" * 70)


if __name__ == "__main__":
    main()