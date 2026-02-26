from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Allow "python app/detection_quality_report.py" from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import CACHE_DIR, cfg
from data.cache_io import load_match_cache
from gameplay.fights import detect_fights


FULL_INTERP_OVERLAY: Dict[str, Any] = {
    "BIN_MS": 5000,
    "DETECT_STEP_MS": 5000,
    "INTERP_XY": True,
    "INTERP_SCALARS_METHOD": "ffill",
    "TF2_GRID_STEP_MS": 5000,
    "TF2_USE_FRAME_INTERP": True,
    "TF2_USE_KILL_TRAJECTORY_INTERP": True,
}

NO_KILL_TRAJ_OVERLAY: Dict[str, Any] = {
    "BIN_MS": 5000,
    "DETECT_STEP_MS": 5000,
    "INTERP_XY": True,
    "INTERP_SCALARS_METHOD": "ffill",
    "TF2_GRID_STEP_MS": 5000,
    "TF2_USE_FRAME_INTERP": True,
    "TF2_USE_KILL_TRAJECTORY_INTERP": False,
}

NO_FRAME_INTERP_5S_OVERLAY: Dict[str, Any] = {
    "BIN_MS": 5000,
    "DETECT_STEP_MS": 5000,
    "INTERP_XY": False,
    "INTERP_SCALARS_METHOD": "ffill",
    "TF2_GRID_STEP_MS": 5000,
    "TF2_USE_FRAME_INTERP": False,
    "TF2_USE_KILL_TRAJECTORY_INTERP": False,
}

SNAPSHOT_60S_OVERLAY: Dict[str, Any] = {
    "BIN_MS": 60000,
    "DETECT_STEP_MS": 60000,
    "INTERP_XY": False,
    "INTERP_SCALARS_METHOD": "ffill",
    "TF2_GRID_STEP_MS": 60000,
    "TF2_USE_FRAME_INTERP": False,
    "TF2_USE_KILL_TRAJECTORY_INTERP": False,
}

VARIANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "full_interp": dict(FULL_INTERP_OVERLAY),
    "no_kill_traj_interp": dict(NO_KILL_TRAJ_OVERLAY),
    "no_frame_interp_5s": dict(NO_FRAME_INTERP_5S_OVERLAY),
    "snapshot_60s_no_interp": dict(SNAPSHOT_60S_OVERLAY),
    # Existing treatment aliases
    "t8": dict(SNAPSHOT_60S_OVERLAY),
    "t9": dict(NO_KILL_TRAJ_OVERLAY),
    "t10": dict(SNAPSHOT_60S_OVERLAY),
}

VARIANT_ALIASES: Dict[str, str] = {
    "baseline": "full_interp",
    "full": "full_interp",
    "interp": "full_interp",
    "kill_traj_off": "no_kill_traj_interp",
    "frame_interp_off": "no_frame_interp_5s",
    "snapshot_60s": "snapshot_60s_no_interp",
    "no_interp_60s": "snapshot_60s_no_interp",
}

_MISSING = object()


@dataclass
class FightDigest:
    match_id: str
    engage_ts: int
    end_ts: int
    first_kill_ts: int
    last_kill_ts: int
    centroid_x: float
    centroid_y: float
    total_kills: int
    kill_diff: int
    post_obj_diff: int
    post_tower_diff: int
    post_gold_diff: float


@dataclass
class MatchPair:
    ref_idx: int
    cand_idx: int
    score: float
    temporal_iou: float
    center_distance: float
    engage_shift_ms: int
    first_kill_shift_ms: int


class CfgOverlay:
    """Temporary cfg overlay helper."""

    def __init__(self, overlay: Mapping[str, Any]) -> None:
        self.overlay = dict(overlay)
        self._snapshot: Dict[str, Any] = {}

    def __enter__(self) -> "CfgOverlay":
        for key, value in self.overlay.items():
            self._snapshot[key] = getattr(cfg, key, _MISSING)
            setattr(cfg, key, value)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, prev in self._snapshot.items():
            if prev is _MISSING:
                try:
                    delattr(cfg, key)
                except Exception:
                    pass
            else:
                setattr(cfg, key, prev)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def temporal_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    if end_a <= start_a or end_b <= start_b:
        return 0.0
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _fight_end_ts(fight: Mapping[str, Any], default_horizon_ms: int) -> int:
    engage_ts = _safe_int(fight.get("engage_ts"), -1)
    if engage_ts < 0:
        engage_ts = _safe_int(fight.get("t_engage_ts"), -1)
    horizon_end = _safe_int(fight.get("horizon_end_ts"), -1)
    label_end = _safe_int(fight.get("label_end_ts"), -1)
    last_kill_ts = _safe_int(fight.get("last_kill_ts"), -1)

    end_ts = horizon_end
    if end_ts <= engage_ts:
        end_ts = label_end
    if end_ts <= engage_ts and last_kill_ts > engage_ts:
        end_ts = last_kill_ts
    if end_ts <= engage_ts and engage_ts >= 0:
        end_ts = engage_ts + int(default_horizon_ms)
    return end_ts


def _digest_fight(match_id: str, fight: Mapping[str, Any], default_horizon_ms: int) -> Optional[FightDigest]:
    engage_ts = _safe_int(fight.get("engage_ts"), -1)
    if engage_ts < 0:
        engage_ts = _safe_int(fight.get("t_engage_ts"), -1)
    if engage_ts < 0:
        return None

    end_ts = _fight_end_ts(fight, default_horizon_ms=default_horizon_ms)
    first_kill_ts = _safe_int(fight.get("first_kill_ts"), engage_ts)
    last_kill_ts = _safe_int(fight.get("last_kill_ts"), first_kill_ts)
    centroid_x = _safe_float(fight.get("centroid_x"), 0.0)
    centroid_y = _safe_float(fight.get("centroid_y"), 0.0)

    outcome = fight.get("outcome", {})
    if not isinstance(outcome, dict):
        outcome = {}
    post = fight.get("post_fight_outcome", {})
    if not isinstance(post, dict):
        post = {}

    return FightDigest(
        match_id=str(match_id),
        engage_ts=int(engage_ts),
        end_ts=int(end_ts),
        first_kill_ts=int(first_kill_ts),
        last_kill_ts=int(last_kill_ts),
        centroid_x=float(centroid_x),
        centroid_y=float(centroid_y),
        total_kills=int(_safe_int(outcome.get("total_kills"), 0)),
        kill_diff=int(_safe_int(outcome.get("kill_diff"), 0)),
        post_obj_diff=int(_safe_int(post.get("post_obj_diff"), 0)),
        post_tower_diff=int(_safe_int(post.get("post_tower_diff"), 0)),
        post_gold_diff=float(_safe_float(post.get("post_gold_diff"), 0.0)),
    )


def summarize_fight_set(fights: Sequence[FightDigest], gold_impact_threshold: float) -> Dict[str, float]:
    n = int(len(fights))
    if n == 0:
        return {
            "n_fights": 0,
            "mean_total_kills": 0.0,
            "mean_abs_kill_diff": 0.0,
            "mean_abs_post_gold_diff": 0.0,
            "objective_impact_rate": 0.0,
            "tower_impact_rate": 0.0,
            "gold_impact_rate": 0.0,
            "impact_rate": 0.0,
        }

    kills = np.asarray([float(f.total_kills) for f in fights], dtype=np.float64)
    kill_diff = np.asarray([abs(float(f.kill_diff)) for f in fights], dtype=np.float64)
    post_gold = np.asarray([abs(float(f.post_gold_diff)) for f in fights], dtype=np.float64)
    post_obj = np.asarray([abs(float(f.post_obj_diff)) for f in fights], dtype=np.float64)
    post_tower = np.asarray([abs(float(f.post_tower_diff)) for f in fights], dtype=np.float64)

    objective_flag = post_obj > 0.0
    tower_flag = post_tower > 0.0
    gold_flag = post_gold >= float(gold_impact_threshold)
    impact_flag = objective_flag | tower_flag | gold_flag

    return {
        "n_fights": float(n),
        "mean_total_kills": float(np.mean(kills)),
        "mean_abs_kill_diff": float(np.mean(kill_diff)),
        "mean_abs_post_gold_diff": float(np.mean(post_gold)),
        "objective_impact_rate": float(np.mean(objective_flag)),
        "tower_impact_rate": float(np.mean(tower_flag)),
        "gold_impact_rate": float(np.mean(gold_flag)),
        "impact_rate": float(np.mean(impact_flag)),
    }


def _center_distance(a: FightDigest, b: FightDigest) -> float:
    dx = float(a.centroid_x) - float(b.centroid_x)
    dy = float(a.centroid_y) - float(b.centroid_y)
    return float(math.hypot(dx, dy))


def match_fight_sets(
    refs: Sequence[FightDigest],
    cands: Sequence[FightDigest],
    *,
    engage_tol_ms: int,
    center_tol: float,
    iou_min: float,
) -> List[MatchPair]:
    if not refs or not cands:
        return []

    tol_ms = max(1, int(engage_tol_ms))
    center_tol_eff = float(center_tol) if float(center_tol) > 0 else float("inf")
    iou_gate = float(max(0.0, min(1.0, iou_min)))

    potentials: List[MatchPair] = []
    for i, rf in enumerate(refs):
        for j, cf in enumerate(cands):
            engage_shift = abs(int(rf.engage_ts) - int(cf.engage_ts))
            first_kill_shift = abs(int(rf.first_kill_ts) - int(cf.first_kill_ts))
            if min(engage_shift, first_kill_shift) > tol_ms:
                continue

            center_dist = _center_distance(rf, cf)
            if not math.isfinite(center_dist) or center_dist > center_tol_eff:
                continue

            tiou = temporal_iou(rf.engage_ts, rf.end_ts, cf.engage_ts, cf.end_ts)
            if tiou < iou_gate:
                continue

            first_bonus = 1.0 if first_kill_shift <= 2000 else 0.0
            center_penalty = 0.0 if math.isinf(center_tol_eff) else (center_dist / max(1.0, center_tol_eff))
            score = (
                first_bonus
                + 2.0 * float(tiou)
                - 0.35 * float(center_penalty)
                - 0.10 * (float(engage_shift) / float(tol_ms))
            )
            potentials.append(
                MatchPair(
                    ref_idx=i,
                    cand_idx=j,
                    score=float(score),
                    temporal_iou=float(tiou),
                    center_distance=float(center_dist),
                    engage_shift_ms=int(engage_shift),
                    first_kill_shift_ms=int(first_kill_shift),
                )
            )

    potentials.sort(
        key=lambda x: (x.score, x.temporal_iou, -x.center_distance, -x.engage_shift_ms, -x.first_kill_shift_ms),
        reverse=True,
    )

    matched_ref = set()
    matched_cand = set()
    out: List[MatchPair] = []
    for pair in potentials:
        if pair.ref_idx in matched_ref or pair.cand_idx in matched_cand:
            continue
        matched_ref.add(pair.ref_idx)
        matched_cand.add(pair.cand_idx)
        out.append(pair)
    return out


def _normalize_team_map(raw: Any) -> Dict[int, int]:
    tm: Dict[int, int] = {}
    raw_map = raw if isinstance(raw, dict) else {}
    for pid in range(1, 11):
        value = raw_map.get(pid, raw_map.get(str(pid)))
        if value is None:
            value = 100 if pid <= 5 else 200
        tid = _safe_int(value, 100 if pid <= 5 else 200)
        if tid not in (100, 200):
            tid = 100 if pid <= 5 else 200
        tm[pid] = tid
    return tm


def _digest_fights_for_variant(
    *,
    match_id: str,
    cache_pack: Dict[str, Any],
    overlay: Mapping[str, Any],
    default_horizon_ms: int,
) -> Tuple[List[FightDigest], Dict[str, Any]]:
    team_map = _normalize_team_map(cache_pack.get("meta", {}).get("team_map", {}))
    with CfgOverlay(overlay):
        fights = detect_fights(cache_pack, team_map)
        diag = cache_pack.get("fight_detect_diag", {})

    digests: List[FightDigest] = []
    for f in fights or []:
        if not isinstance(f, dict):
            continue
        row = _digest_fight(match_id, f, default_horizon_ms=default_horizon_ms)
        if row is not None:
            digests.append(row)

    out_diag = diag if isinstance(diag, dict) else {}
    return digests, dict(out_diag)


def _summarize_diags(diags: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = [
        "clusters_total",
        "clusters_after_spatial",
        "clusters_accepted",
        "accepted",
        "rejected_startctx",
        "rejected_horizon",
        "rejected_alive",
        "rejected_too_few_per_team",
        "rejected_gap",
        "rejected_max_duration",
    ]
    agg = {k: 0 for k in keys}

    n = 0
    for d in diags:
        if not isinstance(d, Mapping):
            continue
        n += 1
        for k in keys:
            agg[k] += _safe_int(d.get(k), 0)

    denom = agg["clusters_after_spatial"] if agg["clusters_after_spatial"] > 0 else agg["clusters_total"]
    accepted = agg["clusters_accepted"] if agg["clusters_accepted"] > 0 else agg["accepted"]
    return {
        "n_matches_with_diag": int(n),
        "clusters_total": int(agg["clusters_total"]),
        "clusters_after_spatial": int(agg["clusters_after_spatial"]),
        "clusters_accepted": int(accepted),
        "cluster_accept_rate": float(accepted) / float(denom) if denom > 0 else 0.0,
        "rejections": {
            "start_context": int(agg["rejected_startctx"]),
            "horizon": int(agg["rejected_horizon"]),
            "alive": int(agg["rejected_alive"]),
            "min_team_participants": int(agg["rejected_too_few_per_team"]),
            "gap": int(agg["rejected_gap"]),
            "max_duration": int(agg["rejected_max_duration"]),
        },
    }


def compare_variants_against_reference(
    *,
    reference_fights: Mapping[str, Sequence[FightDigest]],
    candidate_fights: Mapping[str, Sequence[FightDigest]],
    engage_tol_ms: int,
    center_tol: float,
    iou_min: float,
    gold_impact_threshold: float,
) -> Dict[str, Any]:
    match_ids = sorted(set(reference_fights.keys()) | set(candidate_fights.keys()))

    n_ref = 0
    n_cand = 0
    n_match = 0
    iou_vals: List[float] = []
    center_vals: List[float] = []
    engage_shift_vals: List[float] = []

    matched_ref_fights: List[FightDigest] = []
    matched_cand_fights: List[FightDigest] = []
    ref_only_fights: List[FightDigest] = []
    cand_only_fights: List[FightDigest] = []

    for mid in match_ids:
        refs = list(reference_fights.get(mid, []))
        cands = list(candidate_fights.get(mid, []))
        n_ref += len(refs)
        n_cand += len(cands)

        pairs = match_fight_sets(
            refs,
            cands,
            engage_tol_ms=engage_tol_ms,
            center_tol=center_tol,
            iou_min=iou_min,
        )
        n_match += len(pairs)

        matched_ref_idx = {p.ref_idx for p in pairs}
        matched_cand_idx = {p.cand_idx for p in pairs}

        for p in pairs:
            iou_vals.append(float(p.temporal_iou))
            center_vals.append(float(p.center_distance))
            engage_shift_vals.append(float(p.engage_shift_ms))
            matched_ref_fights.append(refs[p.ref_idx])
            matched_cand_fights.append(cands[p.cand_idx])

        for idx, rf in enumerate(refs):
            if idx not in matched_ref_idx:
                ref_only_fights.append(rf)
        for idx, cf in enumerate(cands):
            if idx not in matched_cand_idx:
                cand_only_fights.append(cf)

    precision = float(n_match) / float(n_cand) if n_cand > 0 else 0.0
    recall = float(n_match) / float(n_ref) if n_ref > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    matched_ref_summary = summarize_fight_set(matched_ref_fights, gold_impact_threshold=gold_impact_threshold)
    matched_cand_summary = summarize_fight_set(matched_cand_fights, gold_impact_threshold=gold_impact_threshold)
    ref_only_summary = summarize_fight_set(ref_only_fights, gold_impact_threshold=gold_impact_threshold)
    cand_only_summary = summarize_fight_set(cand_only_fights, gold_impact_threshold=gold_impact_threshold)

    return {
        "counts": {
            "n_ref": int(n_ref),
            "n_candidate": int(n_cand),
            "n_matched": int(n_match),
            "n_ref_only": int(len(ref_only_fights)),
            "n_candidate_only": int(len(cand_only_fights)),
        },
        "detection_quality": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mean_temporal_iou": float(np.mean(iou_vals)) if iou_vals else 0.0,
            "mean_center_distance": float(np.mean(center_vals)) if center_vals else 0.0,
            "mean_engage_shift_sec": float(np.mean(engage_shift_vals) / 1000.0) if engage_shift_vals else 0.0,
        },
        "meaningfulness": {
            "matched_reference": matched_ref_summary,
            "matched_candidate": matched_cand_summary,
            "reference_only": ref_only_summary,
            "candidate_only": cand_only_summary,
            "delta_candidate_only_minus_reference_only_impact_rate": float(
                cand_only_summary["impact_rate"] - ref_only_summary["impact_rate"]
            ),
        },
    }


def _resolve_variant_name(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        raise ValueError("Empty variant token.")
    t = VARIANT_ALIASES.get(t, t)
    if t not in VARIANT_PRESETS:
        raise ValueError(f"Unknown variant '{token}'. Available: {sorted(VARIANT_PRESETS.keys())}")
    return t


def parse_variant_list(raw: str) -> List[str]:
    toks = [x.strip() for x in str(raw or "").replace(";", ",").split(",") if x.strip()]
    if not toks:
        return ["full_interp", "no_kill_traj_interp", "snapshot_60s_no_interp"]

    out: List[str] = []
    seen = set()
    for tok in toks:
        name = _resolve_variant_name(tok)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _list_match_ids_from_cache(cache_dir: Path) -> List[str]:
    ids: List[str] = []
    for p in sorted(cache_dir.glob("*.meta.json")):
        ids.append(p.stem.replace(".meta", ""))
    return ids


def _load_match_ids_from_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]
            out.extend(parts)
    seen = set()
    uniq: List[str] = []
    for mid in out:
        if mid in seen:
            continue
        seen.add(mid)
        uniq.append(mid)
    return uniq


def _sample_match_ids(ids: Sequence[str], max_matches: int, seed: int, mode: str) -> List[str]:
    out = list(ids)
    if max_matches <= 0 or len(out) <= max_matches:
        return out
    mode_l = str(mode or "first").strip().lower()
    if mode_l == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(out), size=max_matches, replace=False)
        sampled = [out[int(i)] for i in idx.tolist()]
        sampled.sort()
        return sampled
    return out[:max_matches]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_variant_csv(path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "matches",
        "n_fights",
        "avg_fights_per_match",
        "mean_total_kills",
        "mean_abs_post_gold_diff",
        "objective_impact_rate",
        "tower_impact_rate",
        "gold_impact_rate",
        "impact_rate",
        "cluster_accept_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for variant, block in summaries.items():
            fight_s = block.get("fight_summary", {})
            diag_s = block.get("detector_diag", {})
            row = {
                "variant": variant,
                "matches": int(block.get("n_matches", 0)),
                "n_fights": int(fight_s.get("n_fights", 0)),
                "avg_fights_per_match": float(block.get("avg_fights_per_match", 0.0)),
                "mean_total_kills": float(fight_s.get("mean_total_kills", 0.0)),
                "mean_abs_post_gold_diff": float(fight_s.get("mean_abs_post_gold_diff", 0.0)),
                "objective_impact_rate": float(fight_s.get("objective_impact_rate", 0.0)),
                "tower_impact_rate": float(fight_s.get("tower_impact_rate", 0.0)),
                "gold_impact_rate": float(fight_s.get("gold_impact_rate", 0.0)),
                "impact_rate": float(fight_s.get("impact_rate", 0.0)),
                "cluster_accept_rate": float(diag_s.get("cluster_accept_rate", 0.0)),
            }
            w.writerow(row)


def _write_pairwise_csv(path: Path, pairwise: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_variant",
        "n_ref",
        "n_candidate",
        "n_matched",
        "precision",
        "recall",
        "f1",
        "mean_temporal_iou",
        "mean_center_distance",
        "mean_engage_shift_sec",
        "delta_candidate_only_minus_reference_only_impact_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for variant, block in pairwise.items():
            cnt = block.get("counts", {})
            dq = block.get("detection_quality", {})
            mf = block.get("meaningfulness", {})
            row = {
                "candidate_variant": variant,
                "n_ref": int(cnt.get("n_ref", 0)),
                "n_candidate": int(cnt.get("n_candidate", 0)),
                "n_matched": int(cnt.get("n_matched", 0)),
                "precision": float(dq.get("precision", 0.0)),
                "recall": float(dq.get("recall", 0.0)),
                "f1": float(dq.get("f1", 0.0)),
                "mean_temporal_iou": float(dq.get("mean_temporal_iou", 0.0)),
                "mean_center_distance": float(dq.get("mean_center_distance", 0.0)),
                "mean_engage_shift_sec": float(dq.get("mean_engage_shift_sec", 0.0)),
                "delta_candidate_only_minus_reference_only_impact_rate": float(
                    mf.get("delta_candidate_only_minus_reference_only_impact_rate", 0.0)
                ),
            }
            w.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpolation ablation report for teamfight detection quality and meaningfulness.",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="full_interp,no_kill_traj_interp,no_frame_interp_5s,snapshot_60s_no_interp",
        help="Comma-separated variants. Supports aliases: baseline, t8, t9, t10, snapshot_60s.",
    )
    p.add_argument("--reference", type=str, default="full_interp", help="Reference variant for pairwise comparison.")
    p.add_argument("--cache-dir", type=str, default=str(CACHE_DIR), help="Cache directory containing *.meta.json files.")
    p.add_argument("--match-ids", type=str, default="", help="Optional comma-separated explicit match IDs.")
    p.add_argument("--match-ids-file", type=str, default="", help="Optional file with match IDs (line or csv).")
    p.add_argument("--max-matches", type=int, default=200, help="0 keeps all matches.")
    p.add_argument("--sample-mode", type=str, default="first", choices=["first", "random"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--engage-tol-ms", type=int, default=20000)
    p.add_argument("--center-tol", type=float, default=3000.0)
    p.add_argument("--iou-min", type=float, default=0.10)
    p.add_argument("--gold-impact-threshold", type=float, default=1500.0)
    p.add_argument("--output-dir", type=str, default="ablation_results")
    p.add_argument("--output-prefix", type=str, default="detection_quality_report")
    p.add_argument("--progress-every", type=int, default=20)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    variants = parse_variant_list(args.variants)
    reference = _resolve_variant_name(args.reference)
    if reference not in variants:
        variants = [reference] + variants

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    if str(args.match_ids).strip():
        all_ids = [x.strip() for x in str(args.match_ids).replace(";", ",").split(",") if x.strip()]
    elif str(args.match_ids_file).strip():
        all_ids = _load_match_ids_from_file(Path(args.match_ids_file).expanduser())
    else:
        all_ids = _list_match_ids_from_cache(cache_dir)

    if not all_ids:
        raise RuntimeError("No match IDs available for report.")

    selected_ids = _sample_match_ids(
        all_ids,
        max_matches=max(0, int(args.max_matches)),
        seed=int(args.seed),
        mode=str(args.sample_mode),
    )

    default_horizon_ms = int(getattr(cfg, "FIGHT_HORIZON_SEC", 60)) * 1000

    variant_fights: Dict[str, Dict[str, List[FightDigest]]] = {v: {} for v in variants}
    variant_diags: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    processed = 0
    skipped = 0
    start_ts = time.time()

    for i, match_id in enumerate(selected_ids, start=1):
        pack = load_match_cache(str(match_id))
        if not pack:
            skipped += 1
            continue

        for variant in variants:
            digests, diag = _digest_fights_for_variant(
                match_id=str(match_id),
                cache_pack=pack,
                overlay=VARIANT_PRESETS[variant],
                default_horizon_ms=default_horizon_ms,
            )
            variant_fights[variant][str(match_id)] = digests
            variant_diags[variant].append(diag)

        processed += 1
        if int(args.progress_every) > 0 and (i % int(args.progress_every) == 0):
            elapsed = time.time() - start_ts
            print(f"[{i}/{len(selected_ids)}] processed={processed} skipped={skipped} elapsed={elapsed:.1f}s")

    if processed == 0:
        raise RuntimeError("No valid cache packs were loaded for selected matches.")

    variant_summary: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        all_fights = [f for rows in variant_fights[variant].values() for f in rows]
        fs = summarize_fight_set(all_fights, gold_impact_threshold=float(args.gold_impact_threshold))
        n_matches = len(variant_fights[variant])
        variant_summary[variant] = {
            "overlay": dict(VARIANT_PRESETS[variant]),
            "n_matches": int(n_matches),
            "avg_fights_per_match": float(fs["n_fights"]) / float(max(1, n_matches)),
            "fight_summary": fs,
            "detector_diag": _summarize_diags(variant_diags[variant]),
        }

    pairwise: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        if variant == reference:
            continue
        pairwise[variant] = compare_variants_against_reference(
            reference_fights=variant_fights[reference],
            candidate_fights=variant_fights[variant],
            engage_tol_ms=int(args.engage_tol_ms),
            center_tol=float(args.center_tol),
            iou_min=float(args.iou_min),
            gold_impact_threshold=float(args.gold_impact_threshold),
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_name = f"{str(args.output_prefix)}_{ts_str}"
    json_path = out_dir / f"{base_name}.json"
    variant_csv = out_dir / f"{base_name}_variants.csv"
    pairwise_csv = out_dir / f"{base_name}_pairwise.csv"

    payload = {
        "meta": {
            "created_at_local": ts_str,
            "processed_matches": int(processed),
            "skipped_matches": int(skipped),
            "selected_match_ids": int(len(selected_ids)),
            "cache_dir": str(cache_dir),
            "variants": variants,
            "reference": reference,
            "thresholds": {
                "engage_tol_ms": int(args.engage_tol_ms),
                "center_tol": float(args.center_tol),
                "iou_min": float(args.iou_min),
                "gold_impact_threshold": float(args.gold_impact_threshold),
            },
        },
        "variant_summary": variant_summary,
        "pairwise_vs_reference": pairwise,
    }

    _write_json(json_path, payload)
    _write_variant_csv(variant_csv, variant_summary)
    _write_pairwise_csv(pairwise_csv, pairwise)

    print(f"\nSaved report JSON: {json_path}")
    print(f"Saved variant CSV: {variant_csv}")
    print(f"Saved pairwise CSV: {pairwise_csv}")

    for variant, block in pairwise.items():
        dq = block.get("detection_quality", {})
        mf = block.get("meaningfulness", {})
        print(
            f"[{variant}] F1={float(dq.get('f1', 0.0)):.4f}, "
            f"IoU={float(dq.get('mean_temporal_iou', 0.0)):.4f}, "
            f"delta_impact(cand_only-ref_only)="
            f"{float(mf.get('delta_candidate_only_minus_reference_only_impact_rate', 0.0)):+.4f}"
        )


if __name__ == "__main__":
    main()
