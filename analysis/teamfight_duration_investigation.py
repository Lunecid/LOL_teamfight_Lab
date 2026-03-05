"""teamfight_duration_investigation.py — Root-cause analysis for three data quality issues.

Issue 6.1: 838 fights with label_end_ts - t_start_ts > 60,000ms (max 111,497ms)
Issue 6.2: 11,037 overlapping engagement windows (3.67%)
Issue 6.3: 3,598 fights with t_start < START_OFFSET_MIN (1.20%)

This script traces each issue to the exact code path and configuration parameter,
determines whether the behavior is intended, and provides actionable recommendations.

Usage:
    python -m analysis.teamfight_duration_investigation [--cache-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from core.config import cfg, CACHE_DIR
except ImportError:
    cfg = None
    CACHE_DIR = None

logger = logging.getLogger(__name__)


# ===================================================================
#  Helpers
# ===================================================================

def _load_all_fight_refs(cache_dir: Path) -> List[Dict[str, Any]]:
    """Scan cache packs and extract fight metadata for analysis."""
    records: List[Dict[str, Any]] = []
    if not cache_dir.is_dir():
        logger.warning("Cache directory not found: %s", cache_dir)
        return records

    for pack_path in sorted(cache_dir.glob("*.npz")):
        try:
            data = dict(np.load(pack_path, allow_pickle=True))
        except Exception:
            continue

        meta = data.get("meta", None)
        if meta is not None and hasattr(meta, "item"):
            meta = meta.item()
        if not isinstance(meta, dict):
            continue

        match_id = meta.get("match_id", pack_path.stem)
        fights = data.get("fights", None)
        if fights is not None and hasattr(fights, "item"):
            fights = fights.item()
        if not isinstance(fights, list):
            continue

        minute_ts = data.get("minute_ts", np.array([]))
        if hasattr(minute_ts, "item"):
            minute_ts = np.asarray(minute_ts)

        for i, f in enumerate(fights):
            if not isinstance(f, dict):
                continue
            engage_ts = int(f.get("engage_ts", -1))
            horizon_end_ts = int(f.get("horizon_end_ts", -1))
            first_kill_ts = int(f.get("first_kill_ts", -1))
            last_kill_ts = int(f.get("last_kill_ts", -1))
            centroid_x = float(f.get("centroid_x", float("nan")))
            centroid_y = float(f.get("centroid_y", float("nan")))
            ace_truncated = int(f.get("det_end_by_ace", 0))

            if engage_ts < 0:
                continue

            # Compute duration
            if horizon_end_ts > engage_ts:
                label_end = horizon_end_ts
            else:
                label_end = engage_ts + 30000  # default horizon

            duration_ms = label_end - engage_ts

            # Minute index
            t_start_min = engage_ts / 60000.0

            records.append({
                "match_id": str(match_id),
                "fight_idx": i,
                "engage_ts": engage_ts,
                "label_end_ts": label_end,
                "horizon_end_ts": horizon_end_ts,
                "first_kill_ts": first_kill_ts,
                "last_kill_ts": last_kill_ts,
                "duration_ms": duration_ms,
                "cluster_span_ms": max(0, last_kill_ts - first_kill_ts) if last_kill_ts > 0 and first_kill_ts > 0 else 0,
                "t_start_min": t_start_min,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "ace_truncated": ace_truncated,
                "n_segments": int(f.get("n_segments", 1)),
                "det_prox_pairs": int(f.get("det_prox_pairs", 0)),
            })

    return records


# ===================================================================
#  Issue 6.1: Duration > 60,000ms
# ===================================================================

def analyze_long_duration(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Root-cause analysis for fights exceeding 60s duration.

    ROOT CAUSE (traced to gameplay/fights.py:1250):
    ────────────────────────────────────────────────
    horizon_end_ts = max(fight_end_ts, engage_ts + horizon_ms)

    where:
      fight_end_ts   = last_kill_ts + tail_buffer_ms   (tail_buffer=0)
      engage_ts      = first_kill_ts - 10,000ms        (TF2_ENGAGE_PRE_KILL_MS)
      horizon_ms     = 60,000ms                        (FIGHT_HORIZON_SEC * 1000)

    Duration = horizon_end_ts - engage_ts
             = max(last_kill_ts - engage_ts, 60,000)
             = max((last_kill_ts - first_kill_ts) + 10,000, 60,000)

    Duration > 60,000ms  ⟺  cluster_span > 50,000ms
        (consecutive kills within TF2_KILL_CLUSTER_GAP_MS=18s each,
         but total cluster span exceeds 50s)

    Duration cap (gameplay/fights.py:1253):
      fight_end_ts - engage_ts > MAX_MERGED_FIGHT_DURATION_MS (120,000ms)
      → cluster_span + 10,000 > 120,000  →  cluster_span > 110,000ms

    So any cluster spanning 50s-110s produces a fight with duration 60s-120s.
    The maximum observed 111,497ms implies a cluster_span of ~101,497ms.

    VERDICT: Intended behavior — extended kill clusters naturally extend
    the label window. The 120s cap is enforced correctly.
    """
    long = [r for r in records if r["duration_ms"] > 60000]
    if not long:
        return {"count": 0, "message": "No fights exceed 60s duration"}

    durations = np.array([r["duration_ms"] for r in long])
    cluster_spans = np.array([r["cluster_span_ms"] for r in long])

    # Verify the formula: duration ≈ cluster_span + 10000
    expected_durations = cluster_spans + 10000  # engage_pre_kill_ms
    formula_match = np.abs(durations - expected_durations) < 2000  # allow 2s tolerance
    formula_match_pct = float(np.mean(formula_match)) * 100

    # Breakdown by duration range
    ranges = [
        ("60-70s", 60000, 70000),
        ("70-80s", 70000, 80000),
        ("80-90s", 80000, 90000),
        ("90-100s", 90000, 100000),
        ("100-110s", 100000, 110000),
        ("110-120s", 110000, 120000),
    ]
    breakdown = {}
    for label, lo, hi in ranges:
        cnt = int(np.sum((durations >= lo) & (durations < hi)))
        if cnt > 0:
            breakdown[label] = cnt

    # Check ace-truncated among long fights
    ace_cnt = sum(1 for r in long if r["ace_truncated"])

    return {
        "count": len(long),
        "duration_stats": {
            "min_ms": int(np.min(durations)),
            "max_ms": int(np.max(durations)),
            "mean_ms": float(np.mean(durations)),
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
        },
        "cluster_span_stats": {
            "min_ms": int(np.min(cluster_spans)),
            "max_ms": int(np.max(cluster_spans)),
            "mean_ms": float(np.mean(cluster_spans)),
        },
        "formula_verification": {
            "formula": "duration ≈ cluster_span + TF2_ENGAGE_PRE_KILL_MS (10s)",
            "match_pct": round(formula_match_pct, 1),
        },
        "duration_breakdown": breakdown,
        "ace_truncated_count": ace_cnt,
        "root_cause": "Extended kill clusters (span > 50s) with consecutive kills "
                      "within TF2_KILL_CLUSTER_GAP_MS=18s each",
        "code_path": "gameplay/fights.py:1250 — horizon_end_ts = max(fight_end_ts, engage_ts + horizon_ms)",
        "verdict": "INTENDED — label window correctly extends to include all cluster kills; "
                   "cap at MAX_MERGED_FIGHT_DURATION_MS=120s is enforced",
        "recommendation": (
            "Keep in training data. These represent legitimate extended teamfights "
            "(e.g., baron dance, base siege). For models sensitive to fixed-length "
            "windows, normalize by actual duration or add duration as a feature. "
            "Exclude only if model assumes i.i.d. 60s windows."
        ),
    }


# ===================================================================
#  Issue 6.2: Overlapping engagement windows
# ===================================================================

def analyze_overlap(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Root-cause analysis for overlapping engagement windows.

    ROOT CAUSE (traced to gameplay/fight_postmerge.py:113-125):
    ──────────────────────────────────────────────────────────
    The post-merge spacing function has a LOCATION-BASED EXCEPTION:

        if distance_2d((pcx, pcy), (ccx, ccy)) > location_radius:
            kept.append(f)   # ← overlap allowed if fights are far apart
            continue

    When two consecutive fights overlap temporally but are spatially
    distant (> cluster_max_diameter), both are INTENTIONALLY kept.

    Additionally, the overlap check only compares against the immediately
    previous kept fight (kept[-1]), not all preceding fights.

    Overlap definition used in the report:
      fight[i].label_end_ts > fight[i+1].engage_ts
      (horizon end of i-th fight exceeds engagement start of (i+1)-th)

    VERDICT: Intended for i.i.d. models (treat each fight independently).
    Problematic for sequential models (RNN, Transformer) that process
    fights in temporal order within a match.
    """
    # Group by match
    by_match: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_match.setdefault(r["match_id"], []).append(r)

    overlap_cases: List[Dict[str, Any]] = []
    total_pairs = 0

    for match_id, fights in by_match.items():
        fights_sorted = sorted(fights, key=lambda x: x["engage_ts"])
        for i in range(len(fights_sorted) - 1):
            total_pairs += 1
            curr = fights_sorted[i]
            nxt = fights_sorted[i + 1]

            curr_end = curr["label_end_ts"]
            nxt_start = nxt["engage_ts"]

            if curr_end > nxt_start:
                overlap_ms = curr_end - nxt_start
                # Compute spatial distance
                dx = curr["centroid_x"] - nxt["centroid_x"]
                dy = curr["centroid_y"] - nxt["centroid_y"]
                dist = float(np.sqrt(dx * dx + dy * dy))

                overlap_cases.append({
                    "match_id": match_id,
                    "fight_i_engage": curr["engage_ts"],
                    "fight_i_end": curr_end,
                    "fight_j_engage": nxt_start,
                    "overlap_ms": overlap_ms,
                    "spatial_distance": dist,
                    "fight_i_duration": curr["duration_ms"],
                })

    if not overlap_cases:
        return {"count": 0, "message": "No overlapping engagement windows found"}

    overlaps_ms = np.array([c["overlap_ms"] for c in overlap_cases])
    distances = np.array([c["spatial_distance"] for c in overlap_cases])

    # How many overlaps are from spatially distant fights?
    # cluster_max_diameter default = 4000 (raw), but normalized = 4000/16000 = 0.25
    dist_threshold_raw = 4000.0
    coord_norm_div = float(getattr(cfg, "COORD_NORM_DIV", 16000.0)) if cfg else 16000.0
    # Check if distances are in normalized or raw scale
    max_dist = float(np.max(distances)) if len(distances) > 0 else 0
    if max_dist < 2.0:
        # Likely normalized coordinates
        dist_threshold = dist_threshold_raw / coord_norm_div
    else:
        dist_threshold = dist_threshold_raw

    far_apart = int(np.sum(distances > dist_threshold))
    close_together = len(overlap_cases) - far_apart

    return {
        "count": len(overlap_cases),
        "total_consecutive_pairs": total_pairs,
        "overlap_pct": round(len(overlap_cases) / max(1, total_pairs) * 100, 2),
        "overlap_stats": {
            "min_ms": int(np.min(overlaps_ms)),
            "max_ms": int(np.max(overlaps_ms)),
            "mean_ms": float(np.mean(overlaps_ms)),
            "median_ms": float(np.median(overlaps_ms)),
        },
        "spatial_analysis": {
            "far_apart_count": far_apart,
            "close_together_count": close_together,
            "far_apart_pct": round(far_apart / max(1, len(overlap_cases)) * 100, 1),
            "distance_threshold_used": dist_threshold,
        },
        "root_cause": (
            "fight_postmerge.py:113-125 — Location-based exception allows "
            "overlapping fights when centroids are > cluster_max_diameter apart. "
            "Also, only checks against immediately previous fight (kept[-1])."
        ),
        "code_path": "gameplay/fight_postmerge.py:enforce_postmerge_spacing_and_nonoverlap()",
        "verdict": (
            "INTENDED for i.i.d. models — spatially separated simultaneous fights "
            "are valid distinct events. PROBLEMATIC for sequential models where "
            "temporal ordering matters."
        ),
        "recommendation": (
            "For sequential models (RNN, Transformer): apply one of:\n"
            "  (a) Clipping: clip fight[i].label_end_ts to fight[i+1].engage_ts\n"
            "  (b) Masking: skip overlapping fight pairs during sequence training\n"
            "  (c) Merging: merge overlapping fights into a single extended event\n"
            "For i.i.d. models (LightGBM, individual fight prediction): no action needed."
        ),
    }


# ===================================================================
#  Issue 6.3: t_start < START_OFFSET_MIN
# ===================================================================

def analyze_early_fights(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Root-cause analysis for fights occurring before START_OFFSET_MIN.

    ROOT CAUSE (code path analysis):
    ─────────────────────────────────
    START_OFFSET_MIN = 2 (config.py:446) is DEFINED but NEVER ENFORCED
    in the fight detection pipeline.

    The actual filter in gameplay/fights.py:1223-1225:

        ctx_ms = fight_context_sec * 1000   # = 30 * 1000 = 30000
        if engage_ts - ctx_ms < t_min_ms:
            continue

    This only requires engage_ts >= t_min_ms + 30,000ms (30s into game).

    engage_ts = first_kill_ts - TF2_ENGAGE_PRE_KILL_MS (10,000ms)

    So a kill at 40,000ms → engage at 30,000ms → passes the filter.
    t_start_min = 30,000 / 60,000 = 0.5 minute.

    START_OFFSET_MIN is only logged in diagnostics.py:147 for reporting,
    but NOT used as a filter condition anywhere in the pipeline.

    VERDICT: Bug — configuration intent (min 2 minutes) is not enforced.
    The early fights are accepted because only FIGHT_CONTEXT_SEC=30
    is checked, not START_OFFSET_MIN=2.
    """
    start_offset_min = int(getattr(cfg, "START_OFFSET_MIN", 2)) if cfg else 2
    threshold_ms = start_offset_min * 60000

    early = [r for r in records if r["engage_ts"] < threshold_ms]
    if not early:
        return {"count": 0, "message": "No fights before START_OFFSET_MIN"}

    engage_times = np.array([r["engage_ts"] for r in early])
    t_start_mins = np.array([r["t_start_min"] for r in early])

    # Check: how many are exactly at 1 minute?
    at_1min = int(np.sum((t_start_mins >= 0.9) & (t_start_mins < 1.1)))
    before_1min = int(np.sum(t_start_mins < 0.9))

    return {
        "count": len(early),
        "start_offset_min_configured": start_offset_min,
        "threshold_ms": threshold_ms,
        "engage_ts_stats": {
            "min_ms": int(np.min(engage_times)),
            "max_ms": int(np.max(engage_times)),
            "mean_ms": float(np.mean(engage_times)),
        },
        "minute_distribution": {
            "at_1min_count": at_1min,
            "before_1min_count": before_1min,
        },
        "root_cause": (
            "START_OFFSET_MIN=2 is defined in config.py:446 but NEVER "
            "enforced in the detection pipeline. The only time filter is "
            "FIGHT_CONTEXT_SEC=30 (gameplay/fights.py), which requires "
            "engage_ts >= t_min + 30s. START_OFFSET_MIN is only logged in "
            "diagnostics.py:147 — it is a dead configuration parameter."
        ),
        "code_path": (
            "Missing enforcement in gameplay/fights.py — no check for "
            "engage_ts >= t_min + START_OFFSET_MIN * 60000"
        ),
        "verdict": "BUG — intent-vs-implementation mismatch",
        "recommendation": (
            "Option A (enforce): Add filter in detect_fights_teamfight_v2() after line 1225:\n"
            "    start_offset_ms = int(getattr(cfg, 'START_OFFSET_MIN', 2)) * 60000\n"
            "    if engage_ts_val - t_min_ms < start_offset_ms:\n"
            "        diag['rejected_start_offset'] += 1\n"
            "        continue\n"
            "\n"
            "Option B (relax config): Change START_OFFSET_MIN to 1 to match actual behavior.\n"
            "\n"
            "Option C (post-filter): Exclude these 1.2% during dataset construction "
            "rather than detection. This preserves cache reproducibility."
        ),
    }


# ===================================================================
#  Summary report
# ===================================================================

def generate_report(
    records: List[Dict[str, Any]],
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate comprehensive investigation report."""
    total = len(records)
    report = {
        "total_fights_analyzed": total,
        "issue_6_1_long_duration": analyze_long_duration(records),
        "issue_6_2_overlap": analyze_overlap(records),
        "issue_6_3_early_fights": analyze_early_fights(records),
    }

    # Print summary
    print("=" * 72)
    print("TEAMFIGHT DURATION INVESTIGATION REPORT")
    print("=" * 72)
    print(f"\nTotal fights analyzed: {total:,}")

    print("\n" + "-" * 72)
    print("6.1  Duration > 60,000ms")
    print("-" * 72)
    r61 = report["issue_6_1_long_duration"]
    print(f"  Count:      {r61['count']:,}")
    if r61["count"] > 0:
        stats = r61["duration_stats"]
        print(f"  Min:        {stats['min_ms']:,}ms")
        print(f"  Max:        {stats['max_ms']:,}ms")
        print(f"  Mean:       {stats['mean_ms']:,.0f}ms")
        print(f"  Median:     {stats['median_ms']:,.0f}ms")
        fv = r61["formula_verification"]
        print(f"  Formula:    {fv['formula']}")
        print(f"  Match rate: {fv['match_pct']}%")
        print(f"  Verdict:    {r61['verdict']}")

    print("\n" + "-" * 72)
    print("6.2  Engagement Overlap")
    print("-" * 72)
    r62 = report["issue_6_2_overlap"]
    print(f"  Count:      {r62['count']:,}")
    if r62["count"] > 0:
        sa = r62["spatial_analysis"]
        print(f"  Far apart:  {sa['far_apart_count']:,} ({sa['far_apart_pct']}%)")
        print(f"  Close:      {sa['close_together_count']:,}")
        print(f"  Verdict:    {r62['verdict']}")

    print("\n" + "-" * 72)
    print("6.3  t_start < START_OFFSET_MIN")
    print("-" * 72)
    r63 = report["issue_6_3_early_fights"]
    print(f"  Count:      {r63['count']:,}")
    if r63["count"] > 0:
        print(f"  Config:     START_OFFSET_MIN={r63['start_offset_min_configured']}")
        print(f"  Verdict:    {r63['verdict']}")

    print("\n" + "=" * 72)
    print("RECOMMENDED ACTIONS")
    print("=" * 72)
    print("""
  6.1 [P3 — Low]  No code change needed.
      Extended fights are correctly capped at 120s.
      Add duration_ms as model feature for length-aware prediction.

  6.2 [P1 — High for sequential models]
      Document the overlap source in paper.
      For sequential models: implement clipping or masking strategy.

  6.3 [P2 — Medium]  Bug fix required.
      START_OFFSET_MIN=2 is never enforced.
      Apply Option A (add enforcement filter) or Option C (post-filter).
""")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        print(f"  Report saved to: {output_path}")

    return report


# ===================================================================
#  CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Teamfight duration investigation")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Path to match cache directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON report")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    if cache_dir is None:
        print("ERROR: No cache directory specified and CACHE_DIR not available.")
        print("Usage: python -m analysis.teamfight_duration_investigation --cache-dir PATH")
        sys.exit(1)

    print(f"Loading fight data from: {cache_dir}")
    records = _load_all_fight_refs(cache_dir)
    if not records:
        print("No fight records found. Check cache directory path.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    generate_report(records, output_path=output_path)


if __name__ == "__main__":
    main()
