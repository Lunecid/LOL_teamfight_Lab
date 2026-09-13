"""What kill-less encounters look like: damage, hp, map events, and whether the market can name a winner.

Meta-reviewer R4 (CoG 2026, submission 118): "a teamfight can be won even without any kills being
scored (i.e., by depleting the opponent's cooldowns and taking control of the map)".
``run_killless_v33.py`` answers how often; this script answers the winner half.  Every proximity
encounter found by the unchanged scanner under a preset gate is compared with a matched
no-encounter window and with the encounters that do contain a kill.

Groups
  killless / with_kill      encounters split by the scanner's kill test (+-grace, map-wide);
  baseline_of_<group>       one window per encounter: same match, same game-minute band, same number
                            of 5 s frames, the SAME champions (the encounter's participants), no frame
                            within +-grace that is active under the scanner's test at
                            min(--baseline-exclude-min-per-team, gate) champions per side (default 2:
                            no proximity run of any size), no CHAMPION_KILL within +-grace,
                            after-window inside the match, nearest start in time (earlier on ties).

Measured per window, stratified by game-minute band of the window start (2-10, 10-20, 20-30, 30+;
windows before minute 2 are excluded from the bands and counted as lt2; kill-less encounters skew
early, so the pooled row is flagged and a band-standardised with-kill mean is given):
  1. per-minute change of cumulative damage dealt to champions and of hp percentage over the minute
     frames bracketing the window (last frame <= start, first frame >= end), restricted to the
     encounter's champions (mean per champion, and blue-member mean minus red-member mean); hp only
     for champions alive at both frames.  Node columns are looked up by NAME
     (ds_totalDamageDoneToChampions x DS_DENOM, hp_pct, alive), checked against NODE_IDX at start-up
     and against the data on every match (physical + magic + true = total damage to champions
     within Riot's per-component integer rounding of <= 3 units, damage to champions <= all damage,
     cumulative damage never decreases, hp_pct in [0, 1], alive = 1 exactly when hp_pct > 0);
  2. millisecond events WARD_PLACED, WARD_KILL, TURRET_PLATE_DESTROYED, BUILDING_KILL,
     ELITE_MONSTER_KILL inside [start, end] and in the after-window (end, end + after_ms], credited
     to the encounter's champions (creatorId / killerId / assistingParticipantIds), map-wide, and as a
     side balance (blue-credited minus red-credited, the label's team-sign convention; ward placements
     by the creator's team) -- the map-control trace R4 points to;
  3. the market_event verdict over [start, end + after_ms] under engagement attribution (the label's
     own) and window attribution: whether any event the price table values is present (a non-zero
     price and a credited side: a market winner is computable) or none (the outcome is unmeasurable
     by the label, not merely undetected), the priced swing, the tier that decides (market gold beyond
     the dead zone / cluster kills / survivors / structures / draw), and which side the market favours
     (blue; the side with more champions in the encounter; the side ahead in team gold at the last
     minute frame at or before the window start; the side whose champions lost less hp; the side whose
     champions dealt more damage to champions per champion).

Output blocks: ``headline_killless_market`` (item 3 per band), ``headline_physio_events`` (items 1-2 per
band: encounter, baseline, paired difference, kill-less minus with-kill), the full ``summary``,
``paired_differences``, ``killless_minus_with_kill``, ``with_kill_standardized_to_killless_bands``,
``band_distribution``, the frequency ``denominator`` / ``corpus_comparison`` / ``grid_reproduction`` of the
same scan, ``checks`` and ``deviations``.

Match-V5 has no SUMMONER_SPELL_USED or ability-cast event, so cooldown depletion is not observable;
the damage and hp deltas are the closest measurable trace.

5 s track: detection and membership use the scanner's track, which by default moves kill participants
toward each kill position (see scripts/run_killless_v33.py).  ``--kill-trajectory-interp off`` repeats
the characterization on the frame-only track (gate suffix ``_ktoff``).  Every encounter row carries
``kt_touch`` (some frame of the encounter holds a champion whose position differs between the two
tracks) and ``kt_member_share`` (share of the encounter's frames where one of its own members is moved).

"No priced event" is not "no verdict": the label can still name a winner from survivors at the window
end (which, in a kill-less window, can reflect champions still dead from a kill more than grace
earlier) or from structure counts, and a table-type event can carry a zero price (e.g. dragons in the
pooled table).  ``headline_killless_market`` therefore reports ``unmeasurable_share`` (no priced event),
``no_verdict_share`` (the label gives no verdict) and ``verdict_unpriced`` (a verdict without any
priced event) side by side, per attribution.

Every mean carries a 95 % match-clustered percentile bootstrap CI.  One resampling matrix serves
every cell and the frequency denominators, so encounter-minus-baseline (paired, same row) and
kill-less-minus-with-kill differences resample matches jointly.

Reference, followed exactly (deviations are listed in the output JSON):
  * scanner and gate: scripts/run_killless_v33.py -> scripts/run_killless_encounters.py
    ``encounters_for_match`` (imported, unchanged; participants re-derived by ``frame_membership``,
    whose runs are checked against the scanner's on every match);
  * label: gameplay/labels.py ``_compute_label_market_event`` (verdict, called directly),
    ``_lex_refine`` (tier order), ``attribute_events``, ``_priced_event_gold``, ``_event_price_table``
    with ``_apply_price_table_variant``, ``_label_event_team_sign``; v3.3 dead zone 300 g, engagement
    attribution within CLUSTER_MAX_DIAMETER = 4,264 u, tie policy "drop" as for y_market_event in
    corpus_shards_v33;
  * node features: core/config.py NODE_FEATURE_NAMES / DS_DENOM and gameplay/pipeline_cache.py
    (hp_pct = health / healthMax; ds_* = damageStats / DS_DENOM);
  * bootstrap: match-resampling percentile bootstrap as scripts/run_scale_decomposition.py
    ``cluster_bootstrap`` (weights instead of index concatenation; same estimator).

    LOL_OUTPUT_ROOT=D:/LOL_Project LOL_CFG_PRESET=v3.3 python scripts/characterize_killless_v33.py \
        --n-matches 20000 --seed 7 --min-per-team 4 \
        --output D:/LOL_Project/fusion_2615/features/tog_revision/A5-killless/char_t4_dG_g15_20k.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_wrapper():
    path = Path(__file__).resolve().parent / "run_killless_v33.py"
    spec = importlib.util.spec_from_file_location("run_killless_v33", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KV = _load_wrapper()

BANDS = KV.BANDS
band_index = KV.band_index
band_indices = KV.band_indices
POOLED = "pooled_ge2"
DEFAULT_BASELINE_EXCLUDE_MIN_PER_TEAM = 2
EVENT_TYPES = ("WARD_PLACED", "WARD_KILL", "TURRET_PLATE_DESTROYED", "BUILDING_KILL", "ELITE_MONSTER_KILL")
EVENT_SHORT = {"WARD_PLACED": "ward_placed", "WARD_KILL": "ward_kill", "TURRET_PLATE_DESTROYED": "plate",
               "BUILDING_KILL": "building", "ELITE_MONSTER_KILL": "elite"}
TABLE_EVENT_TYPES = ("TURRET_PLATE_DESTROYED", "BUILDING_KILL", "ELITE_MONSTER_KILL", "WARD_KILL")
STRUCTURE_TYPES = ("ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED")
TIERS = ("market_gold", "refine_kills", "refine_alive", "refine_structures", "draw")
ATTRIBUTIONS = (("eng", "engagement"), ("win", "window"))
NODE_FEATURES = {"damage": "ds_totalDamageDoneToChampions", "hp": "hp_pct", "alive": "alive"}
DAMAGE_COMPONENTS = ("ds_physicalDamageDoneToChampions", "ds_magicDamageDoneToChampions", "ds_trueDamageDoneToChampions")
DAMAGE_ALL = "ds_totalDamageDone"

PHYSIO_COLS = ["dmg_pm", "dmg_pm_sum", "hp_pm", "n_hp", "bracket_min", "dmg_pm_bmr", "hp_pm_bmr"]
EVENT_COLS = [f"ev_{w}_{EVENT_SHORT[t]}_{s}" for w in ("in", "after") for t in EVENT_TYPES for s in ("part", "map", "bmr")]
MARKET_FIELDS = ["n_events", "kills", "n_priced_nonkill", "priced_nonkill_any", "priced_nonward_any", "table_type_any",
                 "priced_any", "gd", "abs_gd", "decided", "verdict_defined", "verdict_unpriced", "verdict_blue",
                 *[f"tier_{t}" for t in TIERS], "fav_blue", "fav_numbers", "fav_gold_leader", "fav_hp_leader",
                 "fav_dmg_leader"]
MARKET_COLS = [f"mk_{a}_{f}" for a, _ in ATTRIBUTIONS for f in MARKET_FIELDS]
WINDOW_COLS = PHYSIO_COLS + EVENT_COLS + MARKET_COLS
ENC_ONLY_COLS = ["dur_s", "n_members", "n_blue", "n_red", "peak_blue", "peak_red", "after_truncated", "base_found",
                 "kt_touch", "kt_member_share"]
META_COLS = ["match_idx", "start_ms", "end_ms", "dur_s", "minute", "band", "has_kill", "n_blue", "n_red", "n_members",
             "peak_blue", "peak_red", "numbers_side", "gold_side", "after_truncated", "no_anchor", "window_kills_map",
             "kt_touch", "kt_member_share",
             "base_found", "base_start_ms", "base_offset_s", "base_gold_side", "base_no_anchor"]
COLUMNS = META_COLS + WINDOW_COLS + [f"base_{c}" for c in WINDOW_COLS]
COL = {c: i for i, c in enumerate(COLUMNS)}

DEVIATIONS = [
    KV.TRACK_DEVIATION,
    "the track asymmetry also reaches membership: participants are read from the scanned 5 s track, so on the "
    "kill-adjusted track a with-kill encounter's members (and hence its damage / hp / event columns and the "
    "kill-less minus with-kill contrasts) include champions moved toward the kill; kt_touch and kt_member_share "
    "measure this per encounter, and --kill-trajectory-interp off gives the frame-only characterization",
    KV.ALIVE_DEVIATION,
    "'unmeasurable' (priced_any = 0: no event the price table values with a credited side) is not 'no verdict': the "
    "label can still decide from survivors at the window end or from structure counts (verdict_unpriced), and a "
    "table-type event can carry a zero price; quote unmeasurable_share and no_verdict_share separately and name the "
    "attribution",
    "a bootstrap replicate with an empty cell (no window of that group and band in the resample) is dropped from that "
    "CI; n_valid_reps sits next to every ci95.  fav_* shares are conditional on a priced window with a non-zero swing "
    "and must be quoted with their n",
    "Match-V5 has no SUMMONER_SPELL_USED / ability-cast event: cooldown depletion is unobservable; minute-frame damage "
    "and hp deltas are the proxy",
    "damage / hp deltas use the minute frames bracketing the window (last frame <= start, first frame >= end), not the "
    "exact encounter span; the matched baseline has the same number of 5 s frames, but the bracket length varies and "
    "is reported as bracket_min",
    "hp change is averaged over the window's champions alive at BOTH bracket frames (a death or respawn between frames "
    "would otherwise dominate); *_bmr = blue-member mean minus red-member mean (NaN without a member on each side)",
    "participants = union over the encounter's active frames of alive champions within R of any qualifying anchor; the "
    "scanner itself returns no members.  frame_membership reproduces the scanner's active test (runs compared on every "
    "match: checks.scanner_disagreement_matches)",
    "engagement-attribution centre: the corpus uses the first kill position, which kill-less encounters lack; here it is "
    "the centroid of the champions within R of the best-supported qualifying anchor at its peak frame (earliest on "
    "ties); for a baseline window, the centroid of the encounter's champions alive at the window's middle frame.  A "
    "centre is formed only from raw game-unit positions (xy_raw_minute, not normalised); otherwise the engagement "
    "fields are missing (checks.engagement_attribution_without_centre)",
    "market window = [start, end + after_ms] with after_ms defaulting to the grace window, so a kill-less window holds "
    "no CHAMPION_KILL by construction (checked: checks.killless_window_kills); the corpus label window is "
    "[cutoff, max(last kill + 1, cutoff + 35 s))",
    "for windows with kills, first/last kill for the refinement tier are the attributed kills inside the window (the "
    "corpus passes the detector's cluster kills)",
    "tie policy 'drop' (draws undefined), as y_market_event in corpus_shards_v33; the price table is the pooled "
    "15.14-15.16 fit, which includes the 15.16 test patch -- used here descriptively only, no model is fitted or "
    "selected; its pooled values are checked against the corpus build commit (checks.price_table)",
    "a priced event = an event with a credited side and a non-zero table price (kills always priced); a single ward "
    "kill (25 g) makes a market winner computable but cannot pass the 300 g dead zone, hence priced_nonward_any and decided",
    "n_events counts every Match-V5 event of the window after attribution (items, skills and level-ups included under "
    "window attribution); the market ignores unpriced types",
    "baseline: same match and band only, no cross-band or cross-match fallback (encounters without a valid window get "
    "none: baseline_found_share); exclusion uses the scanner's active test at min(baseline_exclude_min_per_team, gate) "
    "per side, so a baseline for the 4-per-side gate also holds no 2-per-side proximity run within +-grace; nearest "
    "start in time may lie after the encounter; one baseline window can serve several encounters; baselines must have "
    "their after-window inside the match, encounters are kept and flagged (after_truncated)",
    "kill-less minus with-kill differences are not duration-matched (with-kill runs last longer; dur_s is reported); "
    "encounter-minus-baseline pairs are",
    "the kill test is match-wide (scanner): kill-less = no CHAMPION_KILL anywhere on the map within +-grace",
    "per-match denominators are per scanned cache match; see corpus_comparison.same_sample_corpus",
    "WARD_KILL events carry no position, so engagement attribution drops them (attribute_events); they count only under "
    "window attribution and in the participant/map event counts",
    "a match whose characterization raises is kept in the frequency denominators and contributes no rows "
    "(checks.characterize_failed_matches)",
    "event side balance (ev_*_bmr): WARD_PLACED is sided by the creator's team; WARD_KILL, ELITE_MONSTER_KILL, "
    "BUILDING_KILL and TURRET_PLATE_DESTROYED by gameplay/labels.py _label_event_team_sign (structures credited to the "
    "side opposite the destroyed structure's teamId, so minion-taken plates are sided too); an event without a "
    "resolvable side adds 0",
    "for a baseline window fav_numbers uses the encounter's side composition (a baseline holds no proximity cluster of "
    "its own); fav_hp_leader and fav_dmg_leader use the baseline window's own member deltas",
    "with_kill_standardized_to_killless_bands weights the with-kill band means by the kill-less band shares of the "
    "whole sample; the weights are held fixed across bootstrap replicates (the band means are resampled)",
]


def _bump(stats: Dict[str, Any], key: str, n: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + int(n)


# --------------------------------------------------------------------------------------------
# geometry: who is in the encounter
# --------------------------------------------------------------------------------------------
def frame_membership(xy_dense: np.ndarray, alive: Optional[np.ndarray], blue: np.ndarray, red: np.ndarray,
                     radius: float, min_per_team: int) -> Dict[str, np.ndarray]:
    """The scanner's per-frame test, vectorised, keeping who stands near which anchor.

    ``near[t, a, p]``: alive champion p within ``radius`` of alive anchor a (float64, same arithmetic
    as ``encounters_for_match``); ``qual[t, a]``: a has >= min_per_team alive champions of each side
    near it; ``active[t]``: some anchor qualifies.
    """
    pts = np.asarray(xy_dense, dtype=float)[:, :, :2]
    n_frames, n_slots = pts.shape[:2]
    al = np.ones((n_frames, n_slots), dtype=bool) if alive is None else np.asarray(alive, dtype=bool)
    is_blue = np.zeros(n_slots, dtype=bool)
    is_blue[np.asarray(blue, dtype=int)] = True
    is_red = np.zeros(n_slots, dtype=bool)
    is_red[np.asarray(red, dtype=int)] = True
    d2 = ((pts[:, :, None, :] - pts[:, None, :, :]) ** 2).sum(axis=3)
    near = (d2 <= radius * radius) & al[:, :, None] & al[:, None, :]
    blue_near = (near & is_blue[None, None, :]).sum(axis=2)
    red_near = (near & is_red[None, None, :]).sum(axis=2)
    qual = al & (is_blue | is_red)[None, :] & (blue_near >= min_per_team) & (red_near >= min_per_team)
    return {"active": qual.any(axis=1), "qual": qual, "near": near, "blue_near": blue_near, "red_near": red_near,
            "pts": pts, "alive": al, "is_blue": is_blue, "is_red": is_red}


def active_at(mem: Dict[str, np.ndarray], min_per_team: int) -> np.ndarray:
    """The scanner's active test at another per-side threshold, from the same distance matrix."""
    team = mem["is_blue"] | mem["is_red"]
    return (mem["alive"] & team[None, :] & (mem["blue_near"] >= int(min_per_team))
            & (mem["red_near"] >= int(min_per_team))).any(axis=1)


def runs_from_active(active: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
    """(first, last) frame index of every run the scanner reports (same loop as encounters_for_match)."""
    out: List[Tuple[int, int]] = []
    t, n = 0, len(active)
    while t < n:
        if not active[t]:
            t += 1
            continue
        start = t
        while t < n and active[t]:
            t += 1
        if t - start < min_frames:
            continue
        out.append((start, t - 1))
    return out


def encounter_members(mem: Dict[str, np.ndarray], s: int, e: int) -> Optional[Dict[str, Any]]:
    q = mem["qual"][s:e + 1]
    if not q.any():
        return None
    members = (q[:, :, None] & mem["near"][s:e + 1]).any(axis=(0, 1))
    score = np.where(q, mem["blue_near"][s:e + 1] + mem["red_near"][s:e + 1], -1)
    t_rel, a = divmod(int(np.argmax(score)), score.shape[1])
    t = s + t_rel
    at_peak = mem["near"][t, a]
    return {
        "members": members,
        "n_blue": int((members & mem["is_blue"]).sum()), "n_red": int((members & mem["is_red"]).sum()),
        "peak_blue": int(mem["blue_near"][t, a]), "peak_red": int(mem["red_near"][t, a]),
        "anchor": mem["pts"][t, at_peak].mean(axis=0),
    }


def find_baseline(active: np.ndarray, dense_ts: np.ndarray, kill_ts: np.ndarray, n_frames: int, band: int, t0: int,
                  grace_ms: int, step_ms: int, after_ms: int, last_ts: int) -> Optional[int]:
    """First frame of the matched no-encounter window, or None.

    Candidate [b, b + n_frames - 1]: start in ``band``; no ``active`` frame within +-grace of the window
    (``active`` = the exclusion test, see ``active_at``); no CHAMPION_KILL in [ts_b - grace, ts_end + grace]
    (the scanner's kill test); ts_end + after_ms inside the match.  The candidate starting nearest to
    ``t0`` wins, the earlier one on ties.
    """
    n = len(active)
    if n_frames <= 0 or n < n_frames or band < 0:
        return None
    g = int(grace_ms // step_ms)
    csum = np.concatenate([[0], np.cumsum(np.asarray(active, dtype=np.int64))])
    b = np.arange(0, n - n_frames + 1)
    lo = np.clip(b - g, 0, n)
    hi = np.clip(b + n_frames + g, 0, n)
    act = csum[hi] - csum[lo]
    ts0 = np.asarray(dense_ts, dtype=np.int64)[b]
    ts1 = np.asarray(dense_ts, dtype=np.int64)[b + n_frames - 1]
    kills = np.sort(np.asarray(kill_ts, dtype=np.int64))
    k = np.searchsorted(kills, ts1 + grace_ms, side="right") - np.searchsorted(kills, ts0 - grace_ms, side="left")
    ok = (act == 0) & (k == 0) & (band_indices(ts0) == band) & (ts1 + after_ms <= last_ts)
    if not ok.any():
        return None
    cand = b[ok]
    return int(cand[int(np.argmin(np.abs(ts0[ok] - int(t0))))])


# --------------------------------------------------------------------------------------------
# node columns
# --------------------------------------------------------------------------------------------
def verify_node_columns(node_idx: Dict[str, int], feature_names: Sequence[str], ds_denom: Dict[str, float]) -> Dict[str, Any]:
    """Resolve the node columns by name and check NODE_IDX agrees with NODE_FEATURE_NAMES."""
    names = list(feature_names)
    out: Dict[str, Any] = {"n_node_features": len(names)}
    for role, name in NODE_FEATURES.items():
        if name not in names:
            raise ValueError(f"node feature {name!r} is not in NODE_FEATURE_NAMES")
        pos = names.index(name)
        idx = node_idx.get(name)
        if idx is None or int(idx) != pos:
            raise ValueError(f"NODE_IDX[{name!r}]={idx} but NODE_FEATURE_NAMES puts it at {pos}")
        out[role] = {"name": name, "index": pos}
    key = NODE_FEATURES["damage"][len("ds_"):]
    out["damage"]["denominator"] = float(ds_denom[key])
    out["damage"]["raw"] = f"damageStats.{key} (cumulative) = column value x denominator (gameplay/pipeline_cache.py)"
    out["hp"]["raw"] = "championStats.health / healthMax, clipped to [0, 1] (gameplay/pipeline_cache.py)"
    out["alive"]["raw"] = "1 when championStats.health > 0 (gameplay/pipeline_cache.py)"
    checks: Dict[str, Any] = {}
    for name in DAMAGE_COMPONENTS + (DAMAGE_ALL,):
        if name in names and node_idx.get(name) is not None and int(node_idx[name]) == names.index(name):
            checks[name] = {"index": names.index(name), "denominator": float(ds_denom[name[len("ds_"):]])}
    out["damage_check_columns"] = checks
    return out


def node_value_checks(node_minute: np.ndarray, cols: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Data-level evidence that the resolved columns hold what their names say (counts accumulate in ``stats``)."""
    nm = np.asarray(node_minute)
    d, h, a = cols["damage"]["index"], cols["hp"]["index"], cols["alive"]["index"]
    hp = nm[:, :, h].astype(float)
    al = nm[:, :, a].astype(float)
    _bump(stats, "node_cells_checked", hp.size)
    _bump(stats, "hp_pct_out_of_range_cells", int(((hp < -1e-6) | (hp > 1 + 1e-6)).sum()))
    _bump(stats, "alive_nonbinary_cells", int(((al != 0.0) & (al != 1.0)).sum()))
    _bump(stats, "alive_hp_inconsistent_cells", int((((al > 0.5) & (hp <= 0)) | ((al < 0.5) & (hp > 0))).sum()))
    dmg = nm[:, :, d].astype(float) * cols["damage"]["denominator"]
    if len(nm) > 1:
        dd = np.diff(dmg, axis=0)
        _bump(stats, "damage_frame_deltas", dd.size)
        _bump(stats, "damage_frame_deltas_negative", int((dd < -1.0).sum()))
    extra = cols.get("damage_check_columns") or {}
    if all(c in extra for c in DAMAGE_COMPONENTS):
        # Riot reports each component as a separately truncated integer, so physical + magic + true falls short of the
        # total by 0, 1 or 2 units (measured on 50 sampled matches: 45 % / 49 % / 6 % of cells, never more); a gap
        # within 3 units is rounding (damage_component_rounding_gap_cells counts gaps > 0.5), beyond it a mismatch.
        parts = sum(nm[:, :, extra[c]["index"]].astype(float) * extra[c]["denominator"] for c in DAMAGE_COMPONENTS)
        gap = np.abs(dmg - parts)
        tol = np.maximum(3.0, 1e-5 * np.abs(dmg))
        _bump(stats, "damage_component_cells", dmg.size)
        _bump(stats, "damage_component_sum_mismatch_cells", int((gap > tol).sum()))
        _bump(stats, "damage_component_rounding_gap_cells", int(((gap > 0.5) & (gap <= tol)).sum()))
    if DAMAGE_ALL in extra:
        total = nm[:, :, extra[DAMAGE_ALL]["index"]].astype(float) * extra[DAMAGE_ALL]["denominator"]
        _bump(stats, "damage_to_champions_exceeds_total_cells", int((dmg > total + np.maximum(2.0, 1e-5 * total)).sum()))


# --------------------------------------------------------------------------------------------
# per-window measurements
# --------------------------------------------------------------------------------------------
def window_physio(node_minute: np.ndarray, minute_ts: np.ndarray, t0: int, t1: int, members: np.ndarray,
                  cols: Dict[str, Any], is_blue: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {c: np.nan for c in PHYSIO_COLS}
    f0 = int(np.searchsorted(minute_ts, t0, side="right")) - 1
    f1 = int(np.searchsorted(minute_ts, t1, side="left"))
    if f0 < 0 or f1 >= len(minute_ts) or f1 <= f0 or not np.any(members):
        return out
    minutes = float(minute_ts[f1] - minute_ts[f0]) / 60000.0
    pids = np.flatnonzero(members)
    d, h, a = cols["damage"]["index"], cols["hp"]["index"], cols["alive"]["index"]
    dmg = (node_minute[f1, pids, d].astype(float) - node_minute[f0, pids, d].astype(float)) * cols["damage"]["denominator"] / minutes
    both = (node_minute[f0, pids, a] > 0.5) & (node_minute[f1, pids, a] > 0.5)
    hp = (node_minute[f1, pids, h].astype(float) - node_minute[f0, pids, h].astype(float)) * 100.0 / minutes
    out.update(dmg_pm=float(dmg.mean()), dmg_pm_sum=float(dmg.sum()),
               hp_pm=float(hp[both].mean()) if both.any() else np.nan, n_hp=float(both.sum()), bracket_min=minutes)
    if is_blue is not None:
        blue = np.asarray(is_blue, dtype=bool)[pids]
        if blue.any() and (~blue).any():
            out["dmg_pm_bmr"] = float(dmg[blue].mean() - dmg[~blue].mean())
        hb, hr = both & blue, both & ~blue
        if hb.any() and hr.any():
            out["hp_pm_bmr"] = float(hp[hb].mean() - hp[hr].mean())
    return out


def event_side(e: dict, tm: Optional[Dict[int, int]]) -> int:
    """+1 when the event is credited to blue (100), -1 to red (200), 0 when unknown.

    WARD_PLACED: the creator's team (``creatorId`` through the team map).  Every other type follows the
    label's own convention, gameplay/labels.py ``_label_event_team_sign``: WARD_KILL by the killer's team,
    ELITE_MONSTER_KILL by ``killerTeamId``, BUILDING_KILL / TURRET_PLATE_DESTROYED to the side opposite
    the destroyed structure's ``teamId``.
    """
    from gameplay import labels as L

    et = str(e.get("type", "")).upper()
    if et == "WARD_PLACED":
        try:
            team = int((tm or {}).get(int(e.get("creatorId", 0) or 0), 0) or 0)
        except (TypeError, ValueError):
            team = 0
        return 1 if team == 100 else (-1 if team == 200 else 0)
    return int(L._label_event_team_sign(e, tm or {}))


class EventIndex:
    """Sorted timestamps, actor bitmasks (bit p-1 for participant p) and credited side of the five event types."""

    def __init__(self, events: Sequence[dict], tm: Optional[Dict[int, int]] = None):
        buckets: Dict[str, Tuple[List[int], List[int], List[int]]] = {t: ([], [], []) for t in EVENT_TYPES}
        for e in events or []:
            if not isinstance(e, dict):
                continue
            et = str(e.get("type", "")).upper()
            if et not in buckets:
                continue
            try:
                ts = int(e.get("timestamp"))
            except (TypeError, ValueError):
                continue
            if et == "WARD_PLACED":
                actors = [e.get("creatorId")]
            else:
                assists = e.get("assistingParticipantIds") or []
                actors = [e.get("killerId")] + (list(assists) if isinstance(assists, list) else [])
            bits = 0
            for pid in actors:
                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    continue
                if 1 <= pid <= 10:
                    bits |= 1 << (pid - 1)
            buckets[et][0].append(ts)
            buckets[et][1].append(bits)
            buckets[et][2].append(event_side(e, tm))
        self._arr: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for et, (ts_list, bit_list, side_list) in buckets.items():
            ts_arr = np.asarray(ts_list, dtype=np.int64)
            order = np.argsort(ts_arr, kind="stable")
            self._arr[et] = (ts_arr[order], np.asarray(bit_list, dtype=np.int64)[order],
                             np.asarray(side_list, dtype=np.int64)[order])

    def _span(self, et: str, lo_ms: int, hi_excl_ms: int) -> Tuple[int, int]:
        ts = self._arr[et][0]
        return int(np.searchsorted(ts, lo_ms, side="left")), int(np.searchsorted(ts, hi_excl_ms, side="left"))

    def count(self, et: str, lo_ms: int, hi_excl_ms: int, member_bits: int) -> Tuple[int, int]:
        """(credited to a member, map-wide) events of type ``et`` with lo_ms <= t < hi_excl_ms."""
        _, bits, _ = self._arr[et]
        i0, i1 = self._span(et, lo_ms, hi_excl_ms)
        if i1 <= i0:
            return 0, 0
        return int(((bits[i0:i1] & int(member_bits)) != 0).sum()), i1 - i0

    def side_balance(self, et: str, lo_ms: int, hi_excl_ms: int) -> int:
        """Map-wide events of type ``et`` credited to blue minus those credited to red, lo_ms <= t < hi_excl_ms."""
        i0, i1 = self._span(et, lo_ms, hi_excl_ms)
        return int(self._arr[et][2][i0:i1].sum()) if i1 > i0 else 0


def member_bits(members: np.ndarray) -> int:
    return int(sum(1 << int(p) for p in np.flatnonzero(members) if 0 <= int(p) < 10))


def window_events(index: EventIndex, t0: int, t1: int, after_ms: int, bits: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for w, lo, hi in (("in", t0, t1 + 1), ("after", t1 + 1, t1 + after_ms + 1)):
        for et in EVENT_TYPES:
            part, total = index.count(et, lo, hi, bits)
            out[f"ev_{w}_{EVENT_SHORT[et]}_part"] = float(part)
            out[f"ev_{w}_{EVENT_SHORT[et]}_map"] = float(total)
            out[f"ev_{w}_{EVENT_SHORT[et]}_bmr"] = float(index.side_balance(et, lo, hi))
    return out


def gold_side(gold_team_minute: np.ndarray, minute_ts: np.ndarray, t0: int) -> int:
    """Side ahead in team total gold at the last minute frame at or before t0 (causal)."""
    f0 = int(np.searchsorted(minute_ts, t0, side="right")) - 1
    if f0 < 0:
        return 0
    return int(np.sign(float(gold_team_minute[f0, 0]) - float(gold_team_minute[f0, 1])))


def label_price_table(L) -> Dict[str, float]:
    """The table ``_compute_label_market_event`` prices with (its price-table variant applied when present)."""
    table = L._event_price_table()
    variant = getattr(L, "_apply_price_table_variant", None)
    return variant(table) if callable(variant) else table


def market_outcome(pack: Dict[str, Any], tm: Dict[int, int], s_ms: int, e_excl: int, anchor_xy, attribution: str, *,
                   interp, numbers_side: int, gold_leader: int, node_idx: Dict[str, int],
                   hp_leader: int = 0, dmg_leader: int = 0) -> Tuple[Dict[str, float], bool]:
    """market_event over [s_ms, e_excl): priced presence, swing, verdict, deciding tier, favoured side.

    The verdict is the label's own (``_compute_label_market_event``); the swing and the tier repeat its
    arithmetic and ``_lex_refine``'s order so the tier can be reported.  Returns (fields, consistent),
    where consistent is False when the reproduced tier or its direction disagrees with the label's verdict.
    """
    from core.common import safe_float
    from core.config import cfg
    from data.events_index import _events_in_window
    from gameplay import labels as L

    out = {f: np.nan for f in MARKET_FIELDS}
    if str(attribution) == "engagement" and anchor_xy is None:
        return out, True
    table = label_price_table(L)
    first_tower = L._first_tower_ts(pack) if table else None
    deadzone = float(getattr(cfg, "LABEL_GOLD_DEADZONE", 300.0))
    evs = L.attribute_events(_events_in_window(pack, int(s_ms), int(e_excl)), anchor_xy, attribution)

    gd = 0.0
    kd = 0
    struct = 0
    n_priced = 0
    n_priced_nonward = 0
    table_any = False
    kill_priced = False
    kill_ts: List[int] = []
    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            kill_ts.append(int(e.get("timestamp", 0) or 0))
            killer_team = tm.get(int(e.get("killerId", 0) or 0), 0)
            kd += 1 if killer_team == 100 else (-1 if killer_team == 200 else 0)
        elif et in STRUCTURE_TYPES:
            struct += L._label_event_team_sign(e, tm)
        sign = L._label_event_team_sign(e, tm)
        if sign == 0:
            continue
        if et == "CHAMPION_KILL":
            g = max(0.0, safe_float(e.get("bounty", 0.0))) + max(0.0, safe_float(e.get("shutdownBounty", 0.0)))
            assists = e.get("assistingParticipantIds", [])
            g += float(table.get("kills", 0.0)) + float(table.get("assists", 0.0)) * (len(assists) if isinstance(assists, list) else 0)
            gd += float(sign) * g
            kill_priced = kill_priced or g != 0.0
        else:
            g = L._priced_event_gold(e, table, first_tower)
            gd += float(sign) * g
            table_any = table_any or et in TABLE_EVENT_TYPES
            if g != 0.0:
                n_priced += 1
                if et != "WARD_KILL":
                    n_priced_nonward += 1

    first_kill = min(kill_ts) if kill_ts else None
    last_kill = max(kill_ts) if kill_ts else None
    verdict = L._compute_label_market_event(evs, tm, pack, int(s_ms), int(e_excl), interp_node_global=interp,
                                            first_kill_ts=first_kill, last_kill_ts=last_kill,
                                            tie_key=f"{int(s_ms)}:{int(e_excl)}")
    expected: Optional[int] = None
    if gd > deadzone or gd < -deadzone:
        tier, expected = 0, (1 if gd > 0 else 0)
    elif kd != 0:
        tier, expected = 1, (1 if kd > 0 else 0)
    else:
        tier = 4
        alive_idx = node_idx.get("alive", None)
        if alive_idx is not None:
            node_end, _ = interp(pack, int(e_excl) if not (last_kill and last_kill > 0) else int(last_kill))
            tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
            blue_alive = float(node_end[np.where(tids == 100)[0], alive_idx].sum())
            red_alive = float(node_end[np.where(tids == 200)[0], alive_idx].sum())
            if blue_alive != red_alive:
                tier, expected = 2, (1 if blue_alive > red_alive else 0)
        if tier == 4 and struct != 0:
            tier, expected = 3, (1 if struct > 0 else 0)
    consistent = (verdict is None) if expected is None else (verdict is not None and int(verdict) == expected)

    priced_any = n_priced > 0 or kill_priced
    side = 1 if gd > 0 else (-1 if gd < 0 else 0)
    out.update(
        n_events=float(len(evs)), kills=float(len(kill_ts)), n_priced_nonkill=float(n_priced),
        priced_nonkill_any=float(n_priced > 0), priced_nonward_any=float(n_priced_nonward > 0 or kill_priced),
        table_type_any=float(table_any), priced_any=float(priced_any),
        gd=float(gd), abs_gd=float(abs(gd)), decided=float(gd > deadzone or gd < -deadzone),
        verdict_defined=float(verdict is not None), verdict_unpriced=float(verdict is not None and not priced_any),
        verdict_blue=(float(verdict) if verdict is not None else np.nan),
    )
    for i, t in enumerate(TIERS):
        out[f"tier_{t}"] = float(tier == i)
    if priced_any and side != 0:
        out["fav_blue"] = float(side > 0)
        if numbers_side != 0:
            out["fav_numbers"] = float(side == numbers_side)
        if gold_leader != 0:
            out["fav_gold_leader"] = float(side == gold_leader)
        if hp_leader != 0:
            out["fav_hp_leader"] = float(side == hp_leader)
        if dmg_leader != 0:
            out["fav_dmg_leader"] = float(side == dmg_leader)
    return out, consistent


def window_metrics(pack, tm, prep, index: EventIndex, t0: int, t1: int, members: np.ndarray, anchor_xy,
                   numbers_side: int, gold_leader: int, after_ms: int, cols, interp, node_idx, stats,
                   is_blue: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = window_physio(pack["node_minute"], prep["minute_ts"], t0, t1, members, cols, is_blue=is_blue)
    hp_bmr = out.get("hp_pm_bmr", np.nan)
    hp_leader = int(np.sign(hp_bmr)) if np.isfinite(hp_bmr) else 0
    dmg_bmr = out.get("dmg_pm_bmr", np.nan)
    dmg_leader = int(np.sign(dmg_bmr)) if np.isfinite(dmg_bmr) else 0
    out.update(window_events(index, t0, t1, after_ms, member_bits(members)))
    for short, attribution in ATTRIBUTIONS:
        fields, consistent = market_outcome(pack, tm, t0, t1 + after_ms + 1, anchor_xy, attribution, interp=interp,
                                            numbers_side=numbers_side, gold_leader=gold_leader, node_idx=node_idx,
                                            hp_leader=hp_leader, dmg_leader=dmg_leader)
        if not consistent:
            _bump(stats, "tier_verdict_mismatch")
        for k, v in fields.items():
            out[f"mk_{short}_{k}"] = v
    return out


def new_stats() -> Dict[str, Any]:
    keys = ("scanner_disagreement_matches", "tier_verdict_mismatch", "killless_window_kills",
            "encounters_without_members", "node_layout_mismatch_matches", "feature_version_mismatch_matches",
            "xy_not_raw_matches", "track_diff_missing_matches", "characterize_failed_matches", "node_cells_checked",
            "hp_pct_out_of_range_cells",
            "alive_nonbinary_cells", "alive_hp_inconsistent_cells", "damage_frame_deltas", "damage_frame_deltas_negative",
            "damage_component_cells", "damage_component_sum_mismatch_cells", "damage_component_rounding_gap_cells",
            "damage_to_champions_exceeds_total_cells")
    return {k: 0 for k in keys}


def characterize_match(pack: Dict[str, Any], prep: Dict[str, Any], encounters: List[dict], gate: Dict[str, Any],
                       after_ms: int, cols: Dict[str, Any], interp, node_idx: Dict[str, int], match_idx: int,
                       stats: Dict[str, Any], baseline_exclude_min_per_team: Optional[int] = None,
                       feature_version: Optional[str] = None) -> np.ndarray:
    """One row per encounter (COLUMNS): the encounter window and its matched baseline."""
    node_min = pack["node_minute"]
    if np.ndim(node_min) != 3 or int(np.shape(node_min)[2]) != int(cols["n_node_features"]):
        _bump(stats, "node_layout_mismatch_matches")
        return np.empty((0, len(COLUMNS)))
    if feature_version is not None and str(pack.get("meta", {}).get("feature_version", "")) != str(feature_version):
        _bump(stats, "feature_version_mismatch_matches")
    node_value_checks(node_min, cols, stats)

    mem = frame_membership(prep["xy_dense"], prep["alive"], prep["blue"], prep["red"], prep["radius"], gate["min_per_team"])
    dense_ts = np.asarray(prep["dense_ts"], dtype=np.int64)
    mine = [(int(dense_ts[s]), int(dense_ts[e])) for s, e in runs_from_active(mem["active"], gate["min_active_frames"])]
    if mine != [(int(e["start_ms"]), int(e["end_ms"])) for e in encounters]:
        _bump(stats, "scanner_disagreement_matches")
    excl_m = int(gate["min_per_team"]) if baseline_exclude_min_per_team is None else min(
        int(baseline_exclude_min_per_team), int(gate["min_per_team"]))
    excl_active = mem["active"] if excl_m == int(gate["min_per_team"]) else active_at(mem, excl_m)
    tm = {int(k): int(v) for k, v in prep["team_map"].items()}
    minute_ts = prep["minute_ts"]
    gold = np.asarray(pack["gold_team_minute"], dtype=float)
    last_ts = int(minute_ts[-1])
    kill_ts = np.sort(np.asarray(prep["kill_ts"], dtype=np.int64))
    raw_xy = prep["xy_source"] == "xy_raw_minute" and not prep.get("is_norm", False)
    if not raw_xy:
        _bump(stats, "xy_not_raw_matches")
    kt_cells = prep.get("kt_cells")
    if kt_cells is not None and np.shape(kt_cells) != np.shape(mem["alive"]):
        kt_cells = None
    if kt_cells is None:
        _bump(stats, "track_diff_missing_matches")
    index = EventIndex(pack.get("events", []), tm)

    rows: List[List[float]] = []
    for enc in encounters:
        t0, t1 = int(enc["start_ms"]), int(enc["end_ms"])
        s = int(np.searchsorted(dense_ts, t0))
        e = int(np.searchsorted(dense_ts, t1))
        info = encounter_members(mem, s, e)
        if info is None:
            _bump(stats, "encounters_without_members")
            continue
        members = info["members"]
        band = band_index(t0)
        numbers = int(np.sign(info["n_blue"] - info["n_red"]))
        leader = gold_side(gold, minute_ts, t0)
        anchor = tuple(float(v) for v in info["anchor"]) if raw_xy and np.all(np.isfinite(info["anchor"])) else None
        window_kills = int(np.searchsorted(kill_ts, t1 + after_ms, side="right") - np.searchsorted(kill_ts, t0, side="left"))
        if not enc["has_kill"] and window_kills:
            _bump(stats, "killless_window_kills")
        row: Dict[str, float] = {
            "match_idx": match_idx, "start_ms": t0, "end_ms": t1, "dur_s": (t1 - t0) / 1000.0, "minute": t0 / 60000.0,
            "band": band, "has_kill": float(bool(enc["has_kill"])), "n_blue": info["n_blue"], "n_red": info["n_red"],
            "n_members": int(members.sum()), "peak_blue": info["peak_blue"], "peak_red": info["peak_red"],
            "numbers_side": numbers, "gold_side": leader, "after_truncated": float(t1 + after_ms > last_ts),
            "no_anchor": float(anchor is None), "window_kills_map": window_kills, "base_found": 0.0,
        }
        if kt_cells is not None:
            span = np.asarray(kt_cells[s:e + 1], dtype=bool)
            row["kt_touch"] = float(span.any())
            row["kt_member_share"] = float((span & members[None, :]).any(axis=1).mean())
        row.update(window_metrics(pack, tm, prep, index, t0, t1, members, anchor, numbers, leader, after_ms,
                                  cols, interp, node_idx, stats, is_blue=mem["is_blue"]))
        b = find_baseline(excl_active, dense_ts, kill_ts, e - s + 1, band, t0, gate["grace_ms"],
                          gate["grid_step_ms"], after_ms, last_ts) if band >= 0 else None
        if b is not None:
            tb0, tb1 = int(dense_ts[b]), int(dense_ts[b + e - s])
            mid = b + (e - s) // 2
            alive_mid = mem["alive"][mid] & members
            banchor = tuple(float(v) for v in mem["pts"][mid, alive_mid].mean(axis=0)) if (raw_xy and alive_mid.any()) else None
            if banchor is not None and not np.all(np.isfinite(banchor)):
                banchor = None
            bleader = gold_side(gold, minute_ts, tb0)
            base = window_metrics(pack, tm, prep, index, tb0, tb1, members, banchor, numbers, bleader, after_ms,
                                  cols, interp, node_idx, stats, is_blue=mem["is_blue"])
            row.update({f"base_{k}": v for k, v in base.items()})
            row.update(base_found=1.0, base_start_ms=tb0, base_offset_s=(tb0 - t0) / 1000.0, base_gold_side=bleader,
                       base_no_anchor=float(banchor is None))
        rows.append([float(row.get(c, np.nan)) for c in COLUMNS])
    return np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))


# --------------------------------------------------------------------------------------------
# aggregation with match-clustered bootstrap
# --------------------------------------------------------------------------------------------
def cell_stats(weights: np.ndarray, match_idx: np.ndarray, values: np.ndarray, mask: np.ndarray, n_matches: int):
    """Pooled mean per column over masked rows, its bootstrap replicates, n and matches with data (NaN = missing)."""
    v = np.asarray(values, dtype=float)[mask]
    mi = np.asarray(match_idx, dtype=int)[mask]
    k = v.shape[1]
    s = np.zeros((n_matches, k))
    c = np.zeros((n_matches, k))
    finite = np.isfinite(v)
    for j in range(k):
        f = finite[:, j]
        if f.any():
            s[:, j] = np.bincount(mi[f], weights=v[f, j], minlength=n_matches)
            c[:, j] = np.bincount(mi[f], minlength=n_matches)
    n = c.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.where(n > 0, s.sum(axis=0) / np.maximum(n, 1), np.nan)
        reps = (weights @ s) / (weights @ c) if weights.size else np.empty((0, k))
    return point, reps, n, (c > 0).sum(axis=0)


def _num(x) -> Optional[float]:
    return float(x) if x is not None and np.isfinite(x) else None


def _fmt(names: Sequence[str], point, reps, n, nm) -> Dict[str, Any]:
    return {name: {"mean": _num(point[j]), "ci95": KV._ci(reps[:, j]) if reps.size else [None, None],
                   "n_valid_reps": KV.n_valid_reps(reps[:, j]) if reps.size else 0,
                   "n": int(n[j]), "n_matches": int(nm[j])} for j, name in enumerate(names)}


def _quantiles(x: np.ndarray) -> Optional[Dict[str, float]]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
            "n": int(x.size)}


def summarize(table: np.ndarray, n_matches: int, n_boot: int, seed: int,
              weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if weights is None:
        weights = KV.cluster_weights(n_matches, n_boot, seed) if n_boot > 0 and n_matches > 0 else np.empty((0, n_matches))
    mi = table[:, COL["match_idx"]].astype(int)
    band = table[:, COL["band"]].astype(int)
    has_kill = table[:, COL["has_kill"]] > 0.5
    found = table[:, COL["base_found"]] > 0.5
    band_masks = {name: band == i for i, (name, _, _) in enumerate(BANDS)}
    band_masks[POOLED] = band >= 0
    enc_names = ENC_ONLY_COLS + WINDOW_COLS
    enc_vals = table[:, [COL[c] for c in enc_names]]
    base_vals = table[:, [COL[f"base_{c}"] for c in WINDOW_COLS]]
    diff_vals = table[:, [COL[c] for c in WINDOW_COLS]] - base_vals

    groups = {"killless": ~has_kill, "with_kill": has_kill}
    summary: Dict[str, Any] = {}
    paired: Dict[str, Any] = {}
    reps_by: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for gname, gmask in groups.items():
        summary[gname], summary[f"baseline_of_{gname}"], paired[f"{gname}_minus_baseline"] = {}, {}, {}
        for bname, bmask in band_masks.items():
            m = gmask & bmask
            point, reps, n, nm = cell_stats(weights, mi, enc_vals, m, n_matches)
            reps_by[(gname, bname)] = (point, reps)
            summary[gname][bname] = {"n_windows": int(m.sum()), "n_matches": int(len(np.unique(mi[m]))),
                                     "metrics": _fmt(enc_names, point, reps, n, nm)}
            mb = m & found
            point, reps, n, nm = cell_stats(weights, mi, base_vals, mb, n_matches)
            summary[f"baseline_of_{gname}"][bname] = {"n_windows": int(mb.sum()), "n_matches": int(len(np.unique(mi[mb]))),
                                                      "metrics": _fmt(WINDOW_COLS, point, reps, n, nm)}
            point, reps, n, nm = cell_stats(weights, mi, diff_vals, mb, n_matches)
            paired[f"{gname}_minus_baseline"][bname] = {"n_pairs": int(mb.sum()), "metrics": _fmt(WINDOW_COLS, point, reps, n, nm)}

    unpaired: Dict[str, Any] = {}
    for bname in band_masks:
        pk, rk = reps_by[("killless", bname)]
        pw, rw = reps_by[("with_kill", bname)]
        unpaired[bname] = {name: {"diff": _num(pk[j] - pw[j]),
                                  "ci95": KV._ci(rk[:, j] - rw[:, j]) if rk.size else [None, None],
                                  "n_valid_reps": KV.n_valid_reps(rk[:, j] - rw[:, j]) if rk.size else 0}
                           for j, name in enumerate(enc_names)}

    kl_counts = np.array([summary["killless"][name]["n_windows"] for name, _, _ in BANDS], dtype=float)
    standardized: Dict[str, Any] = {"weights_killless_band_shares": None, "metrics": {}}
    if kl_counts.sum() > 0:
        w = kl_counts / kl_counts.sum()
        standardized["weights_killless_band_shares"] = {name: float(x) for (name, _, _), x in zip(BANDS, w)}
        pts = np.stack([reps_by[("with_kill", name)][0] for name, _, _ in BANDS])
        present = w > 0
        std_point = (w[present, None] * pts[present]).sum(axis=0)
        pk, rk = reps_by[("killless", POOLED)]
        std_reps = None
        if rk.size:
            reps_bands = np.stack([reps_by[("with_kill", name)][1] for name, _, _ in BANDS])
            std_reps = (w[present, None, None] * reps_bands[present]).sum(axis=0)
        for j, name in enumerate(enc_names):
            standardized["metrics"][name] = {
                "with_kill_standardized": _num(std_point[j]),
                "ci95": KV._ci(std_reps[:, j]) if std_reps is not None else [None, None],
                "n_valid_reps": KV.n_valid_reps(std_reps[:, j]) if std_reps is not None else 0,
                "killless_pooled_minus_standardized": _num(pk[j] - std_point[j]),
                "ci95_diff": KV._ci(rk[:, j] - std_reps[:, j]) if std_reps is not None else [None, None],
                "n_valid_reps_diff": KV.n_valid_reps(rk[:, j] - std_reps[:, j]) if std_reps is not None else 0,
            }

    distribution = {}
    for gname, gmask in groups.items():
        counts = {name: int((gmask & (band == i)).sum()) for i, (name, _, _) in enumerate(BANDS)}
        counts["lt2"] = int((gmask & (band < 0)).sum())
        total = max(1, int(gmask.sum()))
        distribution[gname] = {
            "counts": counts, "shares": {k: v / total for k, v in counts.items()},
            "baseline_found_share": {name: (float((gmask & bmask & found).sum() / max(1, (gmask & bmask).sum())))
                                     for name, bmask in band_masks.items()},
            "baseline_offset_s": {name: _quantiles(table[gmask & bmask & found, COL["base_offset_s"]])
                                  for name, bmask in band_masks.items()},
            "duration_s": {name: _quantiles(table[gmask & bmask, COL["dur_s"]]) for name, bmask in band_masks.items()},
        }
    return {"summary": summary, "paired_differences": paired, "killless_minus_with_kill": unpaired,
            "with_kill_standardized_to_killless_bands": standardized, "band_distribution": distribution,
            "n_boot": int(weights.shape[0]),
            "pooled_warning": f"{POOLED} pools bands; kill-less encounters skew early, so compare within bands or "
                              "against with_kill_standardized_to_killless_bands"}


def headline(result: Dict[str, Any]) -> Dict[str, Any]:
    """The kill-less 'winner half' numbers, per band, with their matched-baseline and with-kill context."""
    keys = ["priced_any", "priced_nonward_any", "priced_nonkill_any", "table_type_any", "decided", "verdict_defined",
            "verdict_unpriced", "tier_market_gold", "tier_refine_alive", "tier_refine_structures", "tier_draw",
            "fav_blue", "fav_numbers", "fav_gold_leader", "fav_hp_leader", "fav_dmg_leader"]
    context = ["priced_any", "priced_nonward_any", "decided", "verdict_defined"]
    out: Dict[str, Any] = {"note": "shares of kill-less encounter windows [start, end + after_ms]; 'eng' = the label's "
                                   "engagement attribution, 'win' = every event in the window -- always name the "
                                   "attribution.  unmeasurable_share = no event the price table values with a credited "
                                   "side (1 - priced_any); no_verdict_share = the label gives no verdict "
                                   "(1 - verdict_defined); they differ because the label can decide without a priced "
                                   "event from survivors at the window end (tier_refine_alive; in a kill-less window this "
                                   "can reflect champions still dead from a kill more than grace earlier) or from "
                                   "structure counts (tier_refine_structures), counted in verdict_unpriced; table_type_any "
                                   "can exceed priced_any because a table-type event can carry a zero price.  fav_* are "
                                   "shares among priced windows with a non-zero swing: quote each with its n"}
    for short, _ in ATTRIBUTIONS:
        block: Dict[str, Any] = {}
        for bname in [b[0] for b in BANDS] + [POOLED]:
            enc = result["summary"]["killless"][bname]
            base = result["summary"]["baseline_of_killless"][bname]
            wk = result["summary"]["with_kill"][bname]
            row = {"n_windows": enc["n_windows"], "n_baseline": base["n_windows"], "n_with_kill": wk["n_windows"]}
            for k in keys:
                col = f"mk_{short}_{k}"
                row[k] = enc["metrics"][col]
                row[f"baseline_{k}"] = base["metrics"][col]
            for k in context:
                row[f"with_kill_{k}"] = wk["metrics"][f"mk_{short}_{k}"]
            for out_key, col in (("unmeasurable_share", "priced_any"), ("no_verdict_share", "verdict_defined")):
                pm = enc["metrics"][f"mk_{short}_{col}"]
                row[out_key] = (1.0 - pm["mean"]) if pm["mean"] is not None else None
                row[f"{out_key}_ci95"] = ([1.0 - pm["ci95"][1], 1.0 - pm["ci95"][0]]
                                          if pm["ci95"][0] is not None else [None, None])
                row[f"{out_key}_n_valid_reps"] = pm.get("n_valid_reps")
            block[bname] = row
        out[short] = block
    return out


HEADLINE_PHYSIO_EVENT_COLS = (["dmg_pm", "hp_pm", "dmg_pm_bmr", "hp_pm_bmr"]
                              + [f"ev_{w}_{EVENT_SHORT[t]}_{s}" for w in ("in", "after") for t in EVENT_TYPES
                                 for s in ("part", "map", "bmr")])


def headline_physio_events(result: Dict[str, Any]) -> Dict[str, Any]:
    """Damage, hp and map-event columns per band: encounter mean, matched-baseline mean, the paired difference
    (encounter minus its own baseline) and, for kill-less encounters, the within-band kill-less minus with-kill
    difference.  Every entry is {mean | diff, ci95, ...} from the shared match-resampling matrix."""
    out: Dict[str, Any] = {
        "note": "dmg_pm = damage dealt to champions per minute per encounter champion; hp_pm = hp percentage points per "
                "minute (champions alive at both bracket frames); *_bmr (physio) = blue-member mean minus red-member mean; "
                "ev_<in|after>_<type>_<part|map|bmr> = events inside [start, end] / in (end, end + after_ms], credited "
                "to an encounter champion / map-wide / blue-credited minus red-credited; paired = encounter minus its "
                "matched baseline (rows with a baseline only); kt_touch / kt_member_share = share of encounters holding "
                "a frame moved by the kill-trajectory adjustment / mean share of an encounter's frames where one of its "
                "members is moved (the track asymmetry behind the kill-less minus with-kill contrasts)",
        "columns": list(HEADLINE_PHYSIO_EVENT_COLS),
    }
    for gname in ("killless", "with_kill"):
        block: Dict[str, Any] = {}
        for bname in [b[0] for b in BANDS] + [POOLED]:
            enc = result["summary"][gname][bname]
            base = result["summary"][f"baseline_of_{gname}"][bname]
            pair = result["paired_differences"][f"{gname}_minus_baseline"][bname]
            row: Dict[str, Any] = {"n_windows": enc["n_windows"], "n_pairs": pair["n_pairs"]}
            for col in ("kt_touch", "kt_member_share"):
                cell = {"encounter": enc["metrics"][col]}
                if gname == "killless":
                    cell["killless_minus_with_kill"] = result["killless_minus_with_kill"][bname][col]
                row[col] = cell
            for col in HEADLINE_PHYSIO_EVENT_COLS:
                cell = {"encounter": enc["metrics"][col], "baseline": base["metrics"][col],
                        "paired_diff": pair["metrics"][col]}
                if gname == "killless":
                    cell["killless_minus_with_kill"] = result["killless_minus_with_kill"][bname][col]
                row[col] = cell
            block[bname] = row
        out[gname] = block
    return out


def price_table_provenance(cfg_path: str, manifest: Dict[str, Any], pooled_used: Dict[str, float]) -> Dict[str, Any]:
    """Is the table pricing these windows the one the corpus label used?  Bytes differ across checkouts (CRLF),
    so the pooled values are compared with the file at the corpus build commit."""
    rel = str(cfg_path or "")
    path = Path(rel) if Path(rel).is_absolute() else PROJECT_ROOT / rel
    out: Dict[str, Any] = {"path": str(path), "bytes_sha1": KV.sha1_of(path)}
    try:
        pooled_file = {str(k): float(v) for k, v in (json.loads(path.read_text(encoding="utf-8")).get("pooled") or {}).items()}
    except (OSError, ValueError):
        pooled_file = None
    out["label_table_equals_file_pooled"] = (pooled_file == {str(k): float(v) for k, v in pooled_used.items()}) \
        if pooled_file is not None else None
    lab = (manifest or {}).get("label") or {}
    out["manifest_price_table_sha1"] = lab.get("price_table_sha1")
    out["bytes_sha1_matches_manifest"] = (out["bytes_sha1"] == lab.get("price_table_sha1")) if lab else None
    commit = (manifest or {}).get("git_commit")
    out["corpus_build_commit"] = commit
    out["pooled_matches_corpus_build_commit"] = None
    if commit and not Path(rel).is_absolute():
        try:
            blob = subprocess.run(["git", "show", f"{commit}:{Path(rel).as_posix()}"], cwd=str(PROJECT_ROOT),
                                  capture_output=True)
            if blob.returncode == 0:
                lf = blob.stdout.replace(b"\r\n", b"\n")
                crlf = lf.replace(b"\n", b"\r\n")
                out["manifest_sha1_is_build_commit_file"] = lab.get("price_table_sha1") in (
                    hashlib.sha1(lf).hexdigest(), hashlib.sha1(crlf).hexdigest())
                then = {str(k): float(v) for k, v in (json.loads(lf.decode("utf-8")).get("pooled") or {}).items()}
                out["pooled_matches_corpus_build_commit"] = (then == {str(k): float(v) for k, v in pooled_used.items()})
        except Exception as exc:  # git missing or unreadable blob: leave None and say why
            out["git_error"] = repr(exc)[:200]
    out["pooled_used"] = {str(k): float(v) for k, v in pooled_used.items()}
    return out


# --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="v3.3")
    ap.add_argument("--radius", type=float, default=None, help="override R (default: preset TF2_VALIDITY_RADIUS)")
    ap.add_argument("--grace-ms", type=int, default=None, help="override grace (default: preset TF2_ENGAGE_PRE_KILL_MS)")
    ap.add_argument("--min-per-team", type=int, default=4)
    ap.add_argument("--min-duration", type=float, default=None, help="seconds (default: preset G)")
    ap.add_argument("--after-ms", type=int, default=None, help="after-window length (default: the grace window)")
    ap.add_argument("--baseline-exclude-min-per-team", type=int, default=DEFAULT_BASELINE_EXCLUDE_MIN_PER_TEAM,
                    help="a baseline window holds no frame, within +-grace, active at min(this, gate) champions per side")
    ap.add_argument("--n-matches", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scan-first", type=int, default=None, help="characterize only the first K matches of the sorted sample")
    ap.add_argument("--match-source", choices=("cache", "corpus"), default="cache")
    ap.add_argument("--patches", default="", help="comma-separated patches to keep (default: all)")
    ap.add_argument("--scale-json", type=Path, default=KV.DEFAULT_SCALE_JSON)
    ap.add_argument("--shards", type=Path, default=KV.DEFAULT_SHARDS)
    ap.add_argument("--grid-dir", type=Path, default=KV.DEFAULT_GRID_DIR,
                    help="finished v3.3 grid to check the frequency block against ('' to skip)")
    KV.add_scan_args(ap)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--rows-out", default="auto", help="per-encounter rows (.npz); 'auto' = <output>.rows.npz, 'none' = skip")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --output (refused by default)")
    args = ap.parse_args(argv)
    if Path(args.output).exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    if args.scan_slice is not None and args.scan_first:
        raise SystemExit("--scan-first and --scan-slice are mutually exclusive")

    started = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    kt_on = args.kill_trajectory_interp == "on"
    gate = KV.resolve_gate(args.preset, args.radius, args.grace_ms, args.min_per_team, args.min_duration,
                           kill_trajectory_interp=kt_on)
    after_ms = int(gate["grace_ms"] if args.after_ms is None else args.after_ms)
    excl_m = min(int(args.baseline_exclude_min_per_team), int(gate["min_per_team"]))

    from core.config import CACHE_DIR, DS_DENOM, NODE_FEATURE_NAMES, NODE_IDX, cfg
    from gameplay import labels as L
    from gameplay.pipeline_interp import interpolate_node_global

    preset_env, applied_in_process = KV.apply_preset_if_needed(cfg, args.preset)
    track_setting = KV.set_track(cfg, kt_on)
    tie_before = getattr(cfg, "LABEL_TIE_POLICY", None)
    cfg.LABEL_TIE_POLICY = "drop"
    scanner = KV.load_scanner()
    if int(getattr(cfg, "TF2_GRID_STEP_MS", 5000)) != int(scanner.GRID_STEP_MS):
        raise SystemExit("cfg.TF2_GRID_STEP_MS differs from the scanner's GRID_STEP_MS")
    cols = verify_node_columns(NODE_IDX, NODE_FEATURE_NAMES, DS_DENOM)
    feature_version = str(getattr(cfg, "FEATURE_VERSION", ""))

    shard_counts = KV.corpus_counts_by_match(args.shards) if str(args.shards) and Path(args.shards).exists() else None
    published = KV.published_corpus_rates(args.scale_json)
    mids, pool = KV.sample_match_ids(CACHE_DIR, args.n_matches, args.seed, args.match_source,
                                     sorted(shard_counts) if shard_counts else None)
    lo, hi = KV.parse_scan_slice(args.scan_slice, len(mids)) if args.scan_slice is not None else (
        0, len(mids) if (not args.scan_first or args.scan_first >= len(mids)) else int(args.scan_first))
    to_scan = mids[lo:hi]
    keep_patches = frozenset(p.strip() for p in str(args.patches).split(",") if p.strip())
    print(f"gate {gate['name']} (grid {gate['grid_setting']}; {gate['track']} track) after_ms={after_ms} "
          f"baseline_exclusion={excl_m}/side | pool={pool} sampled={len(mids)} scanning=[{lo}:{hi}] ({len(to_scan)})",
          flush=True)

    stats = new_stats()
    failures: List[Dict[str, str]] = []
    tables: List[np.ndarray] = []
    tally = KV.ScanTally()
    for done, match_idx, mid, pack, prep, found in KV.iter_scanned(to_scan, gate, cfg, NODE_IDX, tally, keep_patches,
                                                                    args.progress_every, started, start=lo):
        try:
            tables.append(characterize_match(pack, prep, found, gate, after_ms, cols, interpolate_node_global, NODE_IDX,
                                             match_idx, stats, baseline_exclude_min_per_team=excl_m,
                                             feature_version=feature_version))
        except Exception as exc:
            _bump(stats, "characterize_failed_matches")
            if len(failures) < 20:
                failures.append({"match_id": str(mid), "error": repr(exc)[:300]})
            print(f"characterize failed on {mid}: {exc!r}", file=sys.stderr, flush=True)

    table = np.vstack(tables) if tables else np.empty((0, len(COLUMNS)))
    scan_s = time.time() - started
    n_matches = len(tally.match_ids)
    weights = KV.cluster_weights(n_matches, args.n_boot, args.seed) if (args.n_boot > 0 and n_matches > 0) else None
    result = summarize(table, n_matches, args.n_boot, args.seed,
                       weights=weights if weights is not None else np.empty((0, n_matches)))
    grid_dir = Path(args.grid_dir) if str(args.grid_dir) else None
    blocks = KV.frequency_blocks(tally, gate, n_boot=args.n_boot, seed=args.seed, published=published,
                                 shard_counts=shard_counts, shards_dir=args.shards, grid_dir=grid_dir, n_sampled=len(mids),
                                 full_sample_scanned=(lo == 0 and hi == len(mids)), match_source=args.match_source,
                                 patches_filter=bool(keep_patches), weights=weights, scan_offset=lo)
    corpus_manifest: Dict[str, Any] = {}
    try:
        corpus_manifest = json.loads((Path(args.shards) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    price = price_table_provenance(str(getattr(cfg, "LABEL_EVENT_PRICE_TABLE", "")), corpus_manifest, label_price_table(L))
    checks = {
        **stats,
        "matches_scanned": n_matches,
        "encounter_rows": int(len(table)),
        "characterize_failures": failures,
        "negative_damage_delta_share": (stats["damage_frame_deltas_negative"] / stats["damage_frame_deltas"]
                                        if stats["damage_frame_deltas"] else None),
        "engagement_attribution_without_centre": int(np.nansum(table[:, COL["no_anchor"]])) if len(table) else 0,
        "baseline_attribution_without_centre": int(np.nansum(table[:, COL["base_no_anchor"]])) if len(table) else 0,
        "price_table": price,
        "all_zero": [k for k in ("scanner_disagreement_matches", "tier_verdict_mismatch", "killless_window_kills",
                                 "encounters_without_members", "node_layout_mismatch_matches", "track_diff_missing_matches",
                                 "characterize_failed_matches", "hp_pct_out_of_range_cells", "alive_nonbinary_cells",
                                 "alive_hp_inconsistent_cells", "damage_frame_deltas_negative",
                                 "damage_component_sum_mismatch_cells", "damage_to_champions_exceeds_total_cells")
                     if stats.get(k, 0) != 0],
    }
    checks["all_zero_ok"] = not checks["all_zero"]
    patches = tally.patch_counts()
    output = {
        "item": "A5-killless",
        "script": "scripts/characterize_killless_v33.py",
        "provenance": {
            **KV.git_state(), "code": KV.code_provenance(), "preset": args.preset, "cfg_preset_env": preset_env,
            "preset_applied_in_process": applied_in_process, "track": {"name": gate["track"], **track_setting},
            "scan_slice": [lo, hi],
            "label_key": "market_event (computed per window; tie policy drop; dead zone "
                         f"{float(getattr(cfg, 'LABEL_GOLD_DEADZONE', 300.0)):g} g)",
            "label_tie_policy_before": tie_before,
            "split": "none (descriptive, no model); patches in sample: " + json.dumps(patches, sort_keys=True),
            "seed": args.seed, "bootstrap_seed": args.seed, "n_boot": args.n_boot,
            "n_matches_requested": args.n_matches, "match_source": args.match_source, "match_pool": pool,
            "matches_sampled": len(mids), "scan_first": args.scan_first, "matches_attempted": len(to_scan),
            "matches_scanned": n_matches, "n_missing_cache": tally.n_missing, "n_failed": tally.n_failed,
            "n_patch_skipped": tally.n_patch_skipped, "n_rows": int(len(table)),
            "cache_dir": str(CACHE_DIR), "feature_version": feature_version,
            "attribution_radius_u": float(getattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", 0.0) or getattr(cfg, "CLUSTER_MAX_DIAMETER", 0.0)),
            "argv": list(sys.argv if argv is None else argv), "started_at": started_at,
            "scan_wall_clock_s": round(scan_s, 1), "wall_clock_s": round(time.time() - started, 1),
        },
        "gate": gate,
        "windows": {
            "after_ms": after_ms,
            "encounter_events_inside": "[start_ms, end_ms]",
            "after_window": "(end_ms, end_ms + after_ms]",
            "market_window": "[start_ms, end_ms + after_ms]",
            "physio_bracket": "minute frames: last <= start_ms, first >= end_ms",
            "bands_minutes": {name: [lo, hi] for name, lo, hi in BANDS},
            "baseline_exclusion": {
                "min_per_team": excl_m, "flag": int(args.baseline_exclude_min_per_team),
                "rule": "no frame within +-grace of the baseline window is active under the scanner's test at "
                        "min_per_team champions per side; no CHAMPION_KILL within +-grace; same band; after-window inside the match",
            },
        },
        "feature_index_verification": cols,
        "patch_distribution": patches,
        "checks": checks,
        **{k: blocks[k] for k in ("denominator", "corpus_comparison", "grid_reproduction", "quote")},
        "headline_killless_market": headline(result),
        "headline_physio_events": headline_physio_events(result),
        **result,
        "reference": "scanner scripts/run_killless_encounters.py via scripts/run_killless_v33.py; label "
                     "gameplay/labels.py _compute_label_market_event / _lex_refine / attribute_events; node features "
                     "core/config.py + gameplay/pipeline_cache.py; bootstrap scripts/run_scale_decomposition.py cluster_bootstrap",
        "deviations": DEVIATIONS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1), encoding="utf-8")
    rows_out = None if str(args.rows_out).lower() == "none" else (
        args.output.with_suffix(".rows.npz") if str(args.rows_out).lower() == "auto" else Path(args.rows_out))
    if rows_out is not None:
        np.savez_compressed(rows_out, table=table.astype(np.float32), columns=np.array(COLUMNS),
                            match_ids=np.array(tally.match_ids), match_patch=np.array(tally.patches))
    print(json.dumps({"quote": output["quote"], "grid_reproduction": blocks["grid_reproduction"].get("status"),
                      "checks": {k: v for k, v in checks.items() if k not in ("price_table", "characterize_failures")},
                      "price_pooled_matches_build": price.get("pooled_matches_corpus_build_commit"),
                      "killless_pooled_priced_any_eng": output["headline_killless_market"]["eng"][POOLED]["priced_any"]},
                     indent=1), flush=True)
    print("wrote", args.output, "rows" if rows_out else "", rows_out or "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
