"""Kill-less proximity encounters under a named preset, with the gate and the denominator written out.

Meta-reviewer R4 (CoG 2026, submission 118) asked about teamfights that end without a kill.
``scripts/run_killless_encounters.py`` measures that gap, but its argparse defaults are the v2
constants (1,800 u, 2 per team, 10 s, 10 s grace) and it never reads ``LOL_CFG_PRESET``; its v2
share (4.9 %) was later quoted as a share of *engagements* although the denominator is proximity
*encounters*.  This wrapper runs the same scanner (``encounters_for_match``, imported unchanged)
with R, the grace window and the minimum duration read from ``core.presets`` and writes, next to
every share, the gate that produced it, the denominator block (encounters, kill-less, share,
per-match rates, and the same split by game-minute band of the encounter start), the corpus rates
it may be set against, and a reproduction check against the finished v3.3 grid, so no number can
be quoted without its gate.

Grid reproduction (``grid_reproduction`` in the output; grid = tog_revision/killless_grid, which ran
the scanner with explicit flags, seed 7, 20,000 cache matches):
  * full sample (``--n-matches 20000 --seed 7``, cache pool): matches, encounters, with_kill,
    kill-less, per-match rates, share and median kill-less duration must equal the grid row of the
    same setting (the default gate is row r1600_t4_dG_g15);
  * prefix (``--scan-first K`` on the same 20,000-match sample): the cumulative encounter count and
    the one-decimal kill-less percentage printed by the scanner at every 200th sampled match must
    equal the progress lines of ``<setting>.log`` -- the same matches in the same order.

5 s track (``--kill-trajectory-interp``, default on = the scanner and the grid):
  gameplay.fight_clustering.build_5s_position_grid interpolates minute-frame positions linearly and,
  with cfg.TF2_USE_KILL_TRAJECTORY_INTERP (default True), then moves every kill participant (killer,
  victim, assistants) from its position at the minute frame before the kill straight to the kill
  position over [that frame, kill].  The track that decides whether an encounter exists therefore
  depends on the kills the kill test then splits on: a with-kill encounter is usually detected on
  kill-adjusted positions, a kill-less one mostly on minute-frame interpolation.  ``off`` scans the
  frame-only track (gate name suffix ``_ktoff``; grid reproduction is then not_comparable and the
  kill-adjusted grid row is reported next to the run).  ``--track-sensitivity`` also scans the other
  track on the same matches and writes a paired block (``track_sensitivity``: both shares, per-match
  rates, and their frame-only minus kill-adjusted differences with match-clustered CIs).  Every run
  reports ``denominator.kill_adjusted_track``: the share of with-kill and of kill-less encounters
  holding at least one 5 s frame whose position differs between the two tracks.

Reference, followed exactly:
  * scanner: scripts/run_killless_encounters.py -- ``encounters_for_match`` and the per-match
    preparation in its ``main`` (5 s grid from gameplay.fight_clustering.build_5s_position_grid,
    alive flag held from the last minute frame, coordinate-scale detection, a failed match skipped,
    sample = sorted(random.Random(seed).sample(sorted cache ids, n)));
  * constants: core/presets.py preset "v3.3" (docs/DEFINITION_EVIDENCE.md sections 17, 21, 22):
    R = TF2_VALIDITY_RADIUS = 1,600 u, grace = B = TF2_ENGAGE_PRE_KILL_MS = 15 s, and the default
    minimum duration G = TF2_KILL_CLUSTER_GAP_MS = 13.7 s;
  * corpus rates: fusion_2615/features/scale_decomposition_v33_market_event.json
    (engagements = n / n_matches = 532,547 / 191,940; teamfight class =
    by_participation_scale.teamfight.n / n_matches = 109,829 / 191,940, i.e.
    min(cluster_blue, cluster_red) >= 4 over rows carrying a market_event label), re-counted from
    corpus_shards_v33 when the shards are readable;
  * bootstrap: match-resampling percentile bootstrap as scripts/run_scale_decomposition.py
    ``cluster_bootstrap`` (resampling counts per match instead of index concatenation; same estimator).
Deviations are listed in the ``deviations`` field of the output JSON.

    LOL_OUTPUT_ROOT=D:/LOL_Project LOL_CFG_PRESET=v3.3 python scripts/run_killless_v33.py \
        --n-matches 20000 --seed 7 --min-per-team 4 --track-sensitivity \
        --output D:/LOL_Project/fusion_2615/features/tog_revision/A5-killless/freq_t4_dG_g15_20k.json
    # frame-only track (not comparable with the grid; the kill-adjusted grid row is set next to it):
    ... --min-per-team 4 --kill-trajectory-interp off --output .../freq_t4_dG_g15_ktoff_20k.json
    # a later-patch slice of the same sample (e.g. the last 200 matches, patch 15.16):
    ... --scan-slice=-200: --output .../freq_t4_last200.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_OUTPUT_ROOT = Path(os.environ.get("LOL_OUTPUT_ROOT", "D:/LOL_Project"))
DEFAULT_SCALE_JSON = _OUTPUT_ROOT / "fusion_2615" / "features" / "scale_decomposition_v33_market_event.json"
DEFAULT_SHARDS = _OUTPUT_ROOT / "fusion_2615" / "corpus_shards_v33"
DEFAULT_GRID_DIR = _OUTPUT_ROOT / "fusion_2615" / "features" / "tog_revision" / "killless_grid"
GRID_LOG_EVERY = 200  # the scanner prints cumulative totals after every 200th sampled match

# Game-minute bands of a window start; slot "lt2" (before minute 2) is kept apart, never pooled.
BANDS: Tuple[Tuple[str, float, Optional[float]], ...] = (
    ("2-10", 2.0, 10.0), ("10-20", 10.0, 20.0), ("20-30", 20.0, 30.0), ("30+", 30.0, None),
)
BAND_SLOTS: Tuple[str, ...] = ("lt2",) + tuple(b[0] for b in BANDS)

ANCHOR_RULE = ("a 5 s frame is active when some alive champion has >= min_per_team alive champions of EACH side "
               "within radius_u of its position (every alive champion is tried as the anchor)")
ALIVE_RULE = "alive flag of the last minute frame at or before the 5 s grid timestamp"
KILL_TEST = "has_kill = any CHAMPION_KILL anywhere on the map with start_ms - grace_ms <= t <= end_ms + grace_ms"
DURATION_RULE = "a run of consecutive active frames counts when it holds >= round(min_duration_s * 1000 / grid_step_ms) frames"
TRACK_RULES = {
    True: ("kill-adjusted 5 s track: minute-frame positions interpolated linearly, then every kill participant moved "
           "from its position at the minute frame before the kill straight to the kill position over [that frame, kill] "
           "(gameplay.fight_clustering.build_5s_position_grid, TF2_USE_KILL_TRAJECTORY_INTERP=True; scanner and grid)"),
    False: ("frame-only 5 s track: minute-frame positions interpolated linearly, no kill-position adjustment "
            "(TF2_USE_KILL_TRAJECTORY_INTERP=False)"),
}
TRACK_SHORT = {True: "kill-adjusted", False: "frame-only"}

TRACK_DEVIATION = (
    "detection is not kill-blind: on the default kill-adjusted 5 s track (TF2_USE_KILL_TRAJECTORY_INTERP=True, as in the "
    "scanner and the grid) every kill participant is moved from the minute frame before the kill to the kill position, "
    "so a with-kill encounter is mostly detected on kill-adjusted positions while a kill-less one (no kill within "
    "+-grace) is mostly detected on minute-frame interpolation only (a kill more than grace after it can still move "
    "its frames); the kill test then splits on the same kills.  Encounter counts, the kill-less share and, in the "
    "characterization, encounter membership are conditional on that track.  denominator.kill_adjusted_track gives "
    "the share of with-kill and kill-less encounters holding a kill-adjusted frame; --kill-trajectory-interp off "
    "scans the frame-only track and --track-sensitivity reports both tracks on the same matches with paired CIs, "
    "so a share should be quoted as a range over the two tracks"
)
ALIVE_DEVIATION = (
    "the alive flag of a 5 s frame is held from the last minute frame at or before it (up to 59 s stale, as in the "
    "scanner), so a champion who died or respawned since that frame can count as present in the proximity test; this "
    "applies to with-kill and kill-less encounters alike"
)

WRAPPER_DEVIATIONS = [
    TRACK_DEVIATION,
    ALIVE_DEVIATION,
    "grace window = B = TF2_ENGAGE_PRE_KILL_MS (15 s under v3.3), as in the v3.3 verifier probe; the v2 runs used 10 s",
    "minimum duration defaults to G = TF2_KILL_CLUSTER_GAP_MS / 1000 (13.7 s); on the 5 s grid it rounds to 3 active "
    "frames, i.e. a 10 s span from first to last active frame (13.72 s gives the same gate)",
    "per-match rates are per scanned CACHE match (the cache holds more matches than the corpus, which keeps only matches "
    "with >= 1 labelled engagement); the same-sample corpus block counts corpus engagements on exactly the scanned matches",
    "the kill test is match-wide, so 'kill-less' means no kill anywhere on the map within +-grace of the encounter, "
    "stricter than 'no kill inside the encounter'",
    "matches whose preparation or scan raises are skipped, as in the scanner; they are counted in n_failed (a match with "
    "< 3 minute frames or an empty team, which the scanner skips silently, is counted there too)",
    "by_start_band assigns an encounter to the band of its first active frame; encounters starting before minute 2 "
    "(slot lt2) are in every pooled count and listed separately",
    "bootstrap CIs resample scanned matches; the grid itself reported point values only.  A replicate whose "
    "denominator is empty (e.g. a band with no encounter in the resample) is dropped from that CI; *_n_valid_reps "
    "records how many replicates remain",
    "the corpus comparison reports corpus rates next to the encounter rates and never divides one by the other: the "
    "two units come from different detectors, and proximity is not commitment",
]

# Code whose bytes define a run: the three A5 scripts may be untracked when a run starts, and git_state ignores
# untracked files, so their sha1 (and that of the modules they call) is recorded in provenance.
CODE_FILES = (
    "scripts/run_killless_v33.py", "scripts/characterize_killless_v33.py", "scripts/run_killless_encounters.py",
    "gameplay/fight_clustering.py", "gameplay/fights.py", "gameplay/labels.py", "gameplay/pipeline_interp.py",
    "data/events_index.py", "core/presets.py", "core/config.py",
)

_SCANNER = None


def load_scanner():
    """scripts/run_killless_encounters.py as a module (scripts/ is not a package); never modified."""
    global _SCANNER
    if _SCANNER is None:
        path = Path(__file__).resolve().parent / "run_killless_encounters.py"
        spec = importlib.util.spec_from_file_location("run_killless_encounters", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SCANNER = mod
    return _SCANNER


def sha1_of(path: Path) -> Optional[str]:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_state() -> Dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(PROJECT_ROOT),
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "", None
    return {"git_commit": commit, "git_dirty": dirty}


def code_provenance(paths: Sequence[str] = CODE_FILES) -> Dict[str, Any]:
    """sha1 of the bytes of each file and its git status ('' clean, '??' untracked, ' M' modified, ...)."""
    status: Dict[str, str] = {}
    try:
        res = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
                             cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if len(line) > 3:
                status[line[3:].strip().strip('"').replace("\\", "/")] = line[:2]
        git_ok = res.returncode == 0
    except Exception:
        git_ok = False
    out: Dict[str, Any] = {}
    for rel in paths:
        out[rel] = {"sha1": sha1_of(PROJECT_ROOT / rel),
                    "git_status": (status.get(rel, "") if git_ok else None)}
    return out


# --------------------------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------------------------
def grid_setting_name(radius_u: float, min_per_team: int, min_duration_s: float, grace_ms: int, g_ms: float) -> str:
    """The grid's row name: r<R>_t<M>_d<G|seconds>_g<grace s> (e.g. r1600_t4_dG_g15)."""
    d = "G" if abs(float(min_duration_s) * 1000.0 - float(g_ms)) < 0.5 else f"{float(min_duration_s):g}"
    return f"r{float(radius_u):g}_t{int(min_per_team)}_d{d}_g{int(grace_ms) / 1000:g}"


def resolve_gate(
    preset: str,
    radius: Optional[float] = None,
    grace_ms: Optional[int] = None,
    min_per_team: int = 4,
    min_duration_s: Optional[float] = None,
    grid_step_ms: int = 5000,
    kill_trajectory_interp: bool = True,
) -> Dict[str, Any]:
    """The complete encounter gate: preset values unless a flag overrides them, each with its source.

    ``kill_trajectory_interp`` selects the 5 s track (see TRACK_RULES); off adds the suffix ``_ktoff`` to the gate
    name and to ``grid_setting`` (no grid row exists for it; ``grid_row_same_constants`` names the kill-adjusted row).
    """
    from core.presets import PRESETS

    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    values = PRESETS[preset]
    sources: Dict[str, str] = {}
    if radius is None:
        radius = float(values["TF2_VALIDITY_RADIUS"])
        sources["radius_u"] = f"preset {preset}: TF2_VALIDITY_RADIUS (R)"
    else:
        sources["radius_u"] = "flag --radius"
    if grace_ms is None:
        grace_ms = int(values["TF2_ENGAGE_PRE_KILL_MS"])
        sources["grace_ms"] = f"preset {preset}: TF2_ENGAGE_PRE_KILL_MS (B)"
    else:
        sources["grace_ms"] = "flag --grace-ms"
    if min_duration_s is None:
        min_duration_s = float(values["TF2_KILL_CLUSTER_GAP_MS"]) / 1000.0
        sources["min_duration_s"] = f"preset {preset}: TF2_KILL_CLUSTER_GAP_MS / 1000 (G)"
    else:
        sources["min_duration_s"] = "flag --min-duration"
    sources["min_per_team"] = "flag --min-per-team"
    sources["kill_trajectory_interp"] = ("flag --kill-trajectory-interp (default on = cfg default, scanner and grid)")
    kt = bool(kill_trajectory_interp)
    suffix = "" if kt else "_ktoff"
    min_frames = max(1, int(round(float(min_duration_s) * 1000 / grid_step_ms)))
    same_constants = grid_setting_name(radius, min_per_team, min_duration_s, grace_ms,
                                       float(values["TF2_KILL_CLUSTER_GAP_MS"]))
    return {
        "preset": preset,
        "radius_u": float(radius),
        "min_per_team": int(min_per_team),
        "min_duration_s": float(min_duration_s),
        "grace_ms": int(grace_ms),
        "grid_step_ms": int(grid_step_ms),
        "min_active_frames": int(min_frames),
        "min_span_s": float((min_frames - 1) * grid_step_ms / 1000.0),
        "kill_trajectory_interp": kt,
        "track": TRACK_SHORT[kt],
        "track_rule": TRACK_RULES[kt],
        "name": f"r{float(radius):g}_t{int(min_per_team)}_d{float(min_duration_s):g}_g{int(grace_ms) / 1000:g}{suffix}",
        "grid_setting": same_constants + suffix,
        "grid_row_same_constants": same_constants,
        "sources": sources,
        "anchor_rule": ANCHOR_RULE,
        "alive_rule": ALIVE_RULE,
        "duration_rule": DURATION_RULE,
        "kill_test": KILL_TEST,
        "unit": "proximity encounter (not a corpus engagement)",
    }


def band_index(start_ms: float) -> int:
    """Band of a window start: 0..3 for BANDS, -1 before minute 2."""
    minute = float(start_ms) / 60000.0
    for i, (_, lo, hi) in enumerate(BANDS):
        if minute >= lo and (hi is None or minute < hi):
            return i
    return -1


def band_indices(ts) -> np.ndarray:
    minute = np.asarray(ts, dtype=float).reshape(-1) / 60000.0
    out = np.full(len(minute), -1, dtype=int)
    for i, (_, lo, hi) in enumerate(BANDS):
        sel = minute >= lo
        if hi is not None:
            sel &= minute < hi
        out[sel] = i
    return out


# --------------------------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------------------------
class TrackCfg:
    """Attribute view of a cfg object with TF2_USE_KILL_TRAJECTORY_INTERP replaced.

    build_5s_position_grid reads its switches with getattr(cfg_obj, name, default); every other attribute (grid
    step, frame interpolation) comes from the wrapped object, or the function's default when it is None.
    """

    def __init__(self, base, kill_trajectory_interp: bool) -> None:
        self._base = base
        self.TF2_USE_KILL_TRAJECTORY_INTERP = bool(kill_trajectory_interp)

    def __getattr__(self, name):
        return getattr(self._base, name)


def track_uses_kill_trajectory(cfg_obj) -> bool:
    """The switch build_5s_position_grid will read from ``cfg_obj`` (same default as the function)."""
    return bool(getattr(cfg_obj, "TF2_USE_KILL_TRAJECTORY_INTERP", True)) if cfg_obj is not None else True


def cells_differ(xy_a: np.ndarray, xy_b: np.ndarray) -> np.ndarray:
    """(frames, slots) True where two 5 s tracks put a champion at different positions (NaN == NaN)."""
    a = np.asarray(xy_a, dtype=np.float64)[:, :, :2]
    b = np.asarray(xy_b, dtype=np.float64)[:, :, :2]
    both_nan = np.isnan(a) & np.isnan(b)
    return ((a != b) & ~both_nan).any(axis=2)


def encounter_frame_spans(found: Sequence[dict], dense_ts: np.ndarray) -> List[Tuple[int, int]]:
    """(first, last) 5 s frame index of each encounter (start_ms / end_ms are grid timestamps)."""
    ts = np.asarray(dense_ts, dtype=np.int64)
    return [(int(np.searchsorted(ts, int(e["start_ms"]))), int(np.searchsorted(ts, int(e["end_ms"])))) for e in found]


def prepare_match(pack: Dict[str, Any], radius: float, cfg_obj, node_idx,
                  track_diff: bool = False) -> Optional[Dict[str, Any]]:
    """Per-match inputs of ``encounters_for_match``, exactly as the scanner's ``main`` builds them.

    With ``track_diff`` the other 5 s track (kill-trajectory interpolation toggled) is built as well:
    ``xy_dense_alt`` and ``kt_cells`` = (frames, slots) cells where the two tracks differ, i.e. the
    positions the kill-trajectory adjustment moves.  ``xy_dense`` stays the scanner's own track.
    """
    from gameplay.fight_clustering import build_5s_position_grid
    from gameplay.fights import _extract_kill_events, detect_coordinate_scale

    xy = pack.get("xy_raw_minute")
    xy_source = "xy_raw_minute"
    if xy is None:
        xy = pack["node_minute"][:, :, [node_idx.get("x_norm", 0), node_idx.get("y_norm", 1)]]
        xy_source = "node_minute_xy_norm"
    minute_ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    if len(minute_ts) < 3:
        return None
    team_map = pack["meta"]["team_map"]
    blue = np.array([p - 1 for p, t in team_map.items() if int(t) == 100], dtype=int)
    red = np.array([p - 1 for p, t in team_map.items() if int(t) == 200], dtype=int)
    if len(blue) == 0 or len(red) == 0:
        return None
    events = pack.get("events", [])
    kill_events = _extract_kill_events(events)
    kill_ts = np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
    dense_ts, xy_dense = build_5s_position_grid(xy, minute_ts, kill_events, team_map, cfg_obj=cfg_obj)

    is_norm = bool(pack.get("meta", {}).get("anchor_is_norm", False))
    scale = 1.0
    if not is_norm:
        is_norm, scale = detect_coordinate_scale(xy)
    radius_eff = radius / scale if (is_norm and scale > 0) else radius

    alive_idx = node_idx.get("alive", None)
    alive = None
    if alive_idx is not None:
        try:
            alive_minute = np.asarray(pack["node_minute"][:, :, alive_idx]) > 0.5
            idx = np.clip(np.searchsorted(minute_ts, dense_ts, side="right") - 1, 0, len(minute_ts) - 1)
            alive = alive_minute[idx]
        except Exception:
            alive = None
    prep = {
        "xy_dense": xy_dense, "dense_ts": dense_ts, "alive": alive, "blue": blue, "red": red,
        "kill_ts": kill_ts, "radius": float(radius_eff), "minute_ts": minute_ts, "team_map": team_map,
        "xy_source": xy_source, "is_norm": bool(is_norm), "scale": float(scale),
        "kill_trajectory_interp": track_uses_kill_trajectory(cfg_obj),
    }
    if track_diff:
        try:  # the other track never decides whether the primary scan of this match succeeds
            alt_ts, xy_alt = build_5s_position_grid(xy, minute_ts, kill_events, team_map,
                                                    cfg_obj=TrackCfg(cfg_obj, not prep["kill_trajectory_interp"]))
        except Exception:
            alt_ts, xy_alt = None, None
        if (alt_ts is not None and np.array_equal(np.asarray(alt_ts), np.asarray(dense_ts))
                and np.shape(xy_alt) == np.shape(xy_dense)):
            prep["xy_dense_alt"] = xy_alt
            prep["kt_cells"] = cells_differ(xy_dense, xy_alt)
    return prep


def scan_prepared(prep: Dict[str, Any], gate: Dict[str, Any], alt: bool = False) -> List[dict]:
    """``encounters_for_match`` on the prepared track (``alt``: on the other 5 s track, see prepare_match)."""
    return load_scanner().encounters_for_match(
        prep["xy_dense_alt"] if alt else prep["xy_dense"], prep["dense_ts"], prep["alive"], prep["blue"], prep["red"],
        prep["kill_ts"], prep["radius"], gate["min_per_team"], gate["min_duration_s"], gate["grace_ms"],
    )


class ScanTally:
    """Per-match encounter accounting shared by this wrapper and scripts/characterize_killless_v33.py."""

    def __init__(self) -> None:
        self.match_ids: List[str] = []
        self.patches: List[str] = []
        self.n_enc: List[int] = []
        self.n_kl: List[int] = []
        self.enc_bands: List[np.ndarray] = []
        self.kl_bands: List[np.ndarray] = []
        self.durations_killless: List[float] = []
        self.checkpoints: Dict[int, Dict[str, int]] = {}
        self.kt_enc: List[int] = []
        self.kt_kl: List[int] = []
        self.kt_missing = 0
        self.n_missing = 0
        self.n_failed = 0
        self.n_alt_failed = 0
        self.n_patch_skipped = 0
        self._enc = 0
        self._kl = 0

    def add(self, done: int, mid: str, patch: str, found: Sequence[dict], kt_frames: Optional[np.ndarray] = None,
            dense_ts: Optional[np.ndarray] = None) -> int:
        """Record one scanned match; returns its row index.  ``done`` is the 1-based sample position.

        ``kt_frames`` (bool per 5 s frame: some champion's position differs between the two tracks) with
        ``dense_ts`` counts the encounters holding at least one such frame.
        """
        killless = [e for e in found if not e["has_kill"]]
        slots = len(BAND_SLOTS)
        self.match_ids.append(str(mid))
        self.patches.append(str(patch))
        self.n_enc.append(len(found))
        self.n_kl.append(len(killless))
        if kt_frames is not None and dense_ts is not None:
            kt = np.asarray(kt_frames, dtype=bool)
            touch = [bool(kt[s:e + 1].any()) for s, e in encounter_frame_spans(found, dense_ts)]
            self.kt_enc.append(int(sum(touch)))
            self.kt_kl.append(int(sum(t for t, enc in zip(touch, found) if not enc["has_kill"])))
        else:
            self.kt_enc.append(0)
            self.kt_kl.append(0)
            self.kt_missing += 1
        self.enc_bands.append(np.bincount(band_indices([e["start_ms"] for e in found]) + 1, minlength=slots))
        self.kl_bands.append(np.bincount(band_indices([e["start_ms"] for e in killless]) + 1, minlength=slots))
        self.durations_killless.extend((e["end_ms"] - e["start_ms"]) / 1000.0 for e in killless)
        self._enc += len(found)
        self._kl += len(killless)
        if int(done) % GRID_LOG_EVERY == 0:
            self.checkpoints[int(done)] = {"encounters": self._enc, "killless": self._kl}
        return len(self.match_ids) - 1

    def patch_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self.patches:
            out[p] = out.get(p, 0) + 1
        return dict(sorted(out.items()))

    def band_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        slots = len(BAND_SLOTS)
        return (np.asarray(self.enc_bands, dtype=float).reshape(-1, slots),
                np.asarray(self.kl_bands, dtype=float).reshape(-1, slots))


def iter_scanned(mids: Sequence[str], gate: Dict[str, Any], cfg_obj, node_idx, tally: ScanTally,
                 keep_patches=frozenset(), progress_every: int = 0, started: Optional[float] = None,
                 start: int = 0, alt_tally: Optional[ScanTally] = None, track_diff: bool = True,
                 ) -> Iterator[Tuple[int, int, str, Dict[str, Any], Dict[str, Any], List[dict]]]:
    """Scan ``mids`` in order as the scanner's ``main`` does; yields (done, match_idx, id, pack, prep, encounters).

    ``start``: sample position of ``mids[0]`` minus one (``done`` stays the 1-based position in the whole sample).
    ``alt_tally``: also scan the other 5 s track of every scanned match into it (row for row with ``tally``; a
    failure there records an empty match and counts in ``alt_tally.n_alt_failed``, never removing the primary row).
    ``cfg_obj`` must already carry the primary track's TF2_USE_KILL_TRAJECTORY_INTERP.
    """
    from data.cache_io import load_match_cache

    if bool(gate.get("kill_trajectory_interp", True)) != track_uses_kill_trajectory(cfg_obj):
        raise ValueError("gate kill_trajectory_interp differs from cfg_obj.TF2_USE_KILL_TRAJECTORY_INTERP")
    t0 = time.time() if started is None else float(started)
    for done, mid in enumerate(mids, int(start) + 1):
        pack = load_match_cache(mid)
        if not pack:
            tally.n_missing += 1
            if alt_tally is not None:
                alt_tally.n_missing += 1
        else:
            patch = str(pack.get("meta", {}).get("patch", ""))
            if keep_patches and patch not in keep_patches:
                tally.n_patch_skipped += 1
                if alt_tally is not None:
                    alt_tally.n_patch_skipped += 1
            else:
                try:
                    prep = prepare_match(pack, gate["radius_u"], cfg_obj, node_idx,
                                         track_diff=track_diff or alt_tally is not None)
                    found = None if prep is None else scan_prepared(prep, gate)
                except Exception:
                    found = None
                if found is None:
                    tally.n_failed += 1
                    if alt_tally is not None:
                        alt_tally.n_failed += 1
                else:
                    kt_frames = prep["kt_cells"].any(axis=1) if "kt_cells" in prep else None
                    match_idx = tally.add(done, mid, patch, found, kt_frames, prep["dense_ts"])
                    if alt_tally is not None:
                        try:
                            if "xy_dense_alt" not in prep:
                                raise ValueError("the other track could not be built")
                            found_alt = scan_prepared(prep, gate, alt=True)
                        except Exception:
                            found_alt = []
                            alt_tally.n_alt_failed += 1
                        alt_tally.add(done, mid, patch, found_alt, kt_frames, prep["dense_ts"])
                    yield done, match_idx, mid, pack, prep, found
        pos = done - int(start)
        if progress_every and pos % progress_every == 0:
            share = tally._kl / max(1, tally._enc)
            print(f"{pos}/{len(mids)} matches | encounters={tally._enc} killless={share * 100:.2f}% "
                  f"| {(time.time() - t0) / pos * 1000:.0f} ms/match", flush=True)


def sample_match_ids(cache_dir: Path, n_matches: Optional[int], seed: int, source: str,
                     corpus_ids: Optional[List[str]] = None) -> Tuple[List[str], int]:
    """Same sampling as the scanner (sorted ids, random.Random(seed).sample, sorted) over the chosen pool."""
    if source == "cache":
        mids = sorted(p.stem.replace(".meta", "") for p in Path(cache_dir).glob("*.meta.json"))
    elif source == "corpus":
        if not corpus_ids:
            raise SystemExit("--match-source corpus needs readable --shards")
        mids = sorted(corpus_ids)
    else:
        raise SystemExit(f"unknown --match-source {source!r}")
    pool = len(mids)
    if n_matches and n_matches < len(mids):
        mids = sorted(random.Random(seed).sample(mids, n_matches))
    return mids, pool


# --------------------------------------------------------------------------------------------
# corpus references
# --------------------------------------------------------------------------------------------
def corpus_counts_by_match(shards_dir: Path, y_key: str = "y_market_event", teamfight_min: int = 4) -> Dict[str, Tuple[int, int, str]]:
    """match id -> (labelled engagements, teamfight-class engagements, patch) from the corpus shards."""
    counts: Dict[str, List[Any]] = {}
    for f in sorted(Path(shards_dir).glob("shard_*.npz")):
        with np.load(f) as z:
            groups = z["groups"]
            y = z[y_key]
            smaller = np.minimum(z["cluster_blue"], z["cluster_red"])
            patch = z["patch"]
        keep = y >= 0
        uniq, first, inverse = np.unique(groups[keep], return_index=True, return_inverse=True)
        n_eng = np.bincount(inverse, minlength=len(uniq))
        n_tf = np.bincount(inverse, weights=(smaller[keep] >= teamfight_min).astype(float), minlength=len(uniq))
        patches = patch[keep][first]
        for mid, a, b, p in zip(uniq, n_eng, n_tf, patches):
            row = counts.setdefault(str(mid), [0, 0, str(p)])
            row[0] += int(a)
            row[1] += int(round(b))
    return {k: (v[0], v[1], v[2]) for k, v in counts.items()}


def corpus_totals(shard_counts: Dict[str, Tuple[int, int, str]]) -> Dict[str, int]:
    return {"n_matches": int(len(shard_counts)),
            "n_engagements": int(sum(v[0] for v in shard_counts.values())),
            "n_teamfight_class": int(sum(v[1] for v in shard_counts.values()))}


def published_corpus_rates(scale_json: Path) -> Optional[Dict[str, Any]]:
    """Corpus rates from the scale decomposition; teamfight class = by_participation_scale.teamfight.n."""
    path = Path(scale_json)
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    n, nm = int(d["n"]), int(d["n_matches"])
    by_scale = ((d.get("by_participation_scale") or {}).get("teamfight") or {}).get("n")
    min4 = (d.get("teamfight_min4") or {}).get("n")
    if by_scale is None and min4 is None:
        raise KeyError(f"{path}: neither by_participation_scale.teamfight.n nor teamfight_min4.n")
    tf = int(by_scale if by_scale is not None else min4)
    return {
        "source": str(path), "sha1": sha1_of(path), "y_key": d.get("y_key"),
        "n_engagements": n, "n_matches": nm, "engagements_per_match": n / nm,
        "teamfight_min": int(d.get("teamfight_min", 4)), "n_teamfight_class": tf,
        "teamfight_class_per_match": tf / nm,
        "teamfight_class_key": "by_participation_scale.teamfight.n" if by_scale is not None else "teamfight_min4.n",
        "teamfight_min4_n": None if min4 is None else int(min4),
        "teamfight_keys_agree": None if (by_scale is None or min4 is None) else int(by_scale) == int(min4),
        "unit": "kill-anchored corpus engagement with a market_event label (draws dropped), per corpus match",
        "teamfight_class_definition": "min(cluster_blue, cluster_red) >= 4",
        "corpus_manifest": d.get("corpus_manifest"),
    }


def same_sample_corpus(shard_counts: Optional[Dict[str, Tuple[int, int, str]]], scanned: Sequence[str], shards_dir,
                       published: Optional[Dict[str, Any]] = None, teamfight_min: int = 4) -> Optional[Dict[str, Any]]:
    """Corpus engagements on exactly the scanned matches, plus the whole-corpus totals re-counted from the shards."""
    if shard_counts is None or not scanned:
        return None
    eng = np.array([shard_counts.get(m, (0, 0, ""))[0] for m in scanned], dtype=float)
    tfc = np.array([shard_counts.get(m, (0, 0, ""))[1] for m in scanned], dtype=float)
    totals = corpus_totals(shard_counts)
    block: Dict[str, Any] = {
        "source": str(shards_dir), "y_key": "y_market_event", "teamfight_min": int(teamfight_min),
        "matches": int(len(scanned)), "matches_in_corpus": int(sum(1 for m in scanned if m in shard_counts)),
        "n_engagements": int(eng.sum()), "n_teamfight_class": int(tfc.sum()),
        "engagements_per_match": float(eng.mean()), "teamfight_class_per_match": float(tfc.mean()),
        "unit": "corpus engagements on exactly the scanned matches (0 for a scanned match outside the corpus)",
        "corpus_totals_from_shards": totals,
    }
    if published:
        block["corpus_totals_agree_with_published"] = bool(
            totals["n_engagements"] == published["n_engagements"]
            and totals["n_teamfight_class"] == published["n_teamfight_class"]
            and totals["n_matches"] == published["n_matches"])
    return block


# --------------------------------------------------------------------------------------------
# denominators with match-clustered bootstrap
# --------------------------------------------------------------------------------------------
def cluster_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_clusters) resampling counts: each row draws n_clusters matches with replacement.
    One matrix serves every statistic, so all comparisons resample matches jointly."""
    rng = np.random.default_rng(seed)
    w = np.empty((int(n_boot), int(n_clusters)), dtype=np.float64)
    for b in range(int(n_boot)):
        w[b] = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
    return w


def _ci(reps: np.ndarray) -> List[Optional[float]]:
    """95 % percentile interval over the finite replicates (see n_valid_reps for how many there are)."""
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return [None, None]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return [float(lo), float(hi)]


def n_valid_reps(reps: np.ndarray) -> int:
    """Replicates _ci keeps: a resample with an empty denominator gives a non-finite replicate and is dropped."""
    return int(np.isfinite(np.asarray(reps, dtype=float)).sum())


def band_block(enc_bands: np.ndarray, kl_bands: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Encounters and kill-less encounters by band of the encounter start (column 0 = before minute 2)."""
    slots = len(BAND_SLOTS)
    eb = np.asarray(enc_bands, dtype=float).reshape(-1, slots)
    kb = np.asarray(kl_bands, dtype=float).reshape(-1, slots)
    m = int(eb.shape[0])
    tot_k = float(kb.sum())

    def cell(e_col: np.ndarray, k_col: np.ndarray) -> Dict[str, Any]:
        e, k = float(e_col.sum()), float(k_col.sum())
        c: Dict[str, Any] = {
            "encounters": int(e), "killless": int(k), "with_kill": int(e - k),
            "killless_share_of_encounters": (k / e) if e > 0 else None,
            "encounters_per_match": (e / m) if m else None, "killless_per_match": (k / m) if m else None,
            "share_of_all_killless": (k / tot_k) if tot_k > 0 else None,
        }
        if weights is not None and m:
            we, wk = weights @ e_col, weights @ k_col
            with np.errstate(invalid="ignore", divide="ignore"):
                share_reps = wk / we
            c["killless_share_ci"] = _ci(share_reps)
            c["killless_share_ci_n_valid_reps"] = n_valid_reps(share_reps)
            c["killless_per_match_ci"] = _ci(wk / m)
        return c

    out: Dict[str, Any] = {"slots_minutes": {"lt2": [None, 2.0], **{n: [lo, hi] for n, lo, hi in BANDS}}}
    for j, name in enumerate(BAND_SLOTS):
        out[name] = cell(eb[:, j], kb[:, j])
    out["from_minute_2"] = cell(eb[:, 1:].sum(axis=1), kb[:, 1:].sum(axis=1))
    return out


KT_TOUCH_DEFINITION = ("an encounter touches the kill-adjusted track when at least one of its 5 s frames holds a "
                       "champion whose position differs between the kill-adjusted and the frame-only track (any "
                       "champion on the map; the characterization also reports the encounter's own members)")


def kill_adjusted_track_block(n_enc: np.ndarray, n_kl: np.ndarray, kt_enc: np.ndarray, kt_kl: np.ndarray,
                              weights: Optional[np.ndarray], kt_missing: int = 0) -> Dict[str, Any]:
    """How many with-kill and kill-less encounters hold a frame the kill-trajectory adjustment moves."""
    n_enc, n_kl = np.asarray(n_enc, dtype=float), np.asarray(n_kl, dtype=float)
    kt_enc, kt_kl = np.asarray(kt_enc, dtype=float), np.asarray(kt_kl, dtype=float)
    n_wk, kt_wk = n_enc - n_kl, kt_enc - kt_kl
    out: Dict[str, Any] = {
        "definition": KT_TOUCH_DEFINITION, "matches_without_track_diff": int(kt_missing),
        "encounters_touching": int(kt_enc.sum()), "with_kill_touching": int(kt_wk.sum()),
        "killless_touching": int(kt_kl.sum()),
        "with_kill_touching_share": float(kt_wk.sum() / n_wk.sum()) if n_wk.sum() > 0 else None,
        "killless_touching_share": float(kt_kl.sum() / n_kl.sum()) if n_kl.sum() > 0 else None,
    }
    if weights is not None and weights.size and len(n_enc):
        with np.errstate(invalid="ignore", divide="ignore"):
            r_wk = (weights @ kt_wk) / (weights @ n_wk)
            r_kl = (weights @ kt_kl) / (weights @ n_kl)
        out.update(with_kill_touching_share_ci=_ci(r_wk), with_kill_touching_share_ci_n_valid_reps=n_valid_reps(r_wk),
                   killless_touching_share_ci=_ci(r_kl), killless_touching_share_ci_n_valid_reps=n_valid_reps(r_kl))
    return out


def denominator_block(n_enc: np.ndarray, n_killless: np.ndarray, durations_killless: List[float],
                      n_boot: int = 1000, seed: int = 7, enc_bands: Optional[np.ndarray] = None,
                      kl_bands: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None,
                      kt_enc: Optional[np.ndarray] = None, kt_kl: Optional[np.ndarray] = None,
                      kt_missing: int = 0) -> Dict[str, Any]:
    """Encounters, kill-less, share and per-match rates over the scanned matches, with match-clustered CIs."""
    n_enc = np.asarray(n_enc, dtype=float)
    n_kl = np.asarray(n_killless, dtype=float)
    m = int(len(n_enc))
    w = weights if weights is not None else (cluster_weights(m, n_boot, seed) if (m and n_boot > 0) else None)
    out: Dict[str, Any] = {
        "matches_scanned": m,
        "encounters": int(n_enc.sum()),
        "with_kill": int((n_enc - n_kl).sum()),
        "killless": int(n_kl.sum()),
        "killless_share_of_encounters": float(n_kl.sum() / n_enc.sum()) if n_enc.sum() > 0 else None,
        "encounters_per_match": float(n_enc.mean()) if m else None,
        "killless_per_match": float(n_kl.mean()) if m else None,
        "with_kill_per_match": float((n_enc - n_kl).mean()) if m else None,
        "share_of_matches_with_killless": float((n_kl > 0).mean()) if m else None,
        "killless_median_duration_s": float(np.median(durations_killless)) if durations_killless else None,
        "n_boot": 0 if w is None else int(w.shape[0]),
    }
    if w is not None and m:
        we, wk = w @ n_enc, w @ n_kl
        with np.errstate(invalid="ignore", divide="ignore"):
            share_reps = wk / we
        out["killless_share_ci"] = _ci(share_reps)
        out["killless_share_ci_n_valid_reps"] = n_valid_reps(share_reps)
        out["encounters_per_match_ci"] = _ci(we / m)
        out["killless_per_match_ci"] = _ci(wk / m)
    if enc_bands is not None and kl_bands is not None:
        out["by_start_band"] = band_block(enc_bands, kl_bands, w if m else None)
    if kt_enc is not None and kt_kl is not None:
        out["kill_adjusted_track"] = kill_adjusted_track_block(n_enc, n_kl, kt_enc, kt_kl, w if m else None, kt_missing)
    return out


def track_sensitivity_block(primary: ScanTally, alt: ScanTally, primary_kill_trajectory: bool,
                            weights: Optional[np.ndarray]) -> Dict[str, Any]:
    """Both 5 s tracks on the same matches: shares, per-match rates and frame-only minus kill-adjusted differences.

    The two tallies hold the same matches row for row, so one match-resampling matrix pairs them (matches
    resampled jointly)."""
    if primary.match_ids != alt.match_ids:
        raise ValueError("track tallies do not hold the same matches in the same order")
    by_track = {TRACK_SHORT[bool(primary_kill_trajectory)]: primary, TRACK_SHORT[not bool(primary_kill_trajectory)]: alt}
    m = len(primary.match_ids)
    use_w = weights is not None and weights.size and m
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    out: Dict[str, Any] = {
        "primary_track": TRACK_SHORT[bool(primary_kill_trajectory)], "matches": m,
        "alt_track_failed_matches": int(alt.n_alt_failed),
        "note": "same matches scanned on both 5 s tracks; differences are frame-only minus kill-adjusted with "
                "match-clustered paired CIs",
    }
    for name, t in by_track.items():
        enc = np.asarray(t.n_enc, dtype=float)
        kl = np.asarray(t.n_kl, dtype=float)
        eb, kb = t.band_arrays()
        enc2, kl2 = eb[:, 1:].sum(axis=1), kb[:, 1:].sum(axis=1)
        arrays[name] = {"enc": enc, "kl": kl, "wk": enc - kl, "enc2": enc2, "kl2": kl2}
        block = {
            "encounters": int(enc.sum()), "killless": int(kl.sum()), "with_kill": int((enc - kl).sum()),
            "killless_share_of_encounters": float(kl.sum() / enc.sum()) if enc.sum() > 0 else None,
            "encounters_per_match": float(enc.mean()) if m else None,
            "killless_per_match": float(kl.mean()) if m else None,
            "with_kill_per_match": float((enc - kl).mean()) if m else None,
            "from_minute_2_killless_share": float(kl2.sum() / enc2.sum()) if enc2.sum() > 0 else None,
            "killless_median_duration_s": float(np.median(t.durations_killless)) if t.durations_killless else None,
        }
        if use_w:
            with np.errstate(invalid="ignore", divide="ignore"):
                share = (weights @ kl) / (weights @ enc)
            block.update(killless_share_ci=_ci(share), killless_share_ci_n_valid_reps=n_valid_reps(share),
                         killless_per_match_ci=_ci((weights @ kl) / m), encounters_per_match_ci=_ci((weights @ enc) / m))
        out[name] = block
    a, b = arrays["frame-only"], arrays["kill-adjusted"]
    diff: Dict[str, Any] = {}
    for key, num, den in (("killless_share_of_encounters", "kl", "enc"), ("from_minute_2_killless_share", "kl2", "enc2")):
        pa = a[num].sum() / a[den].sum() if a[den].sum() > 0 else np.nan
        pb = b[num].sum() / b[den].sum() if b[den].sum() > 0 else np.nan
        cell: Dict[str, Any] = {"diff": float(pa - pb) if np.isfinite(pa - pb) else None,
                                "ratio": float(pa / pb) if (np.isfinite(pa) and np.isfinite(pb) and pb > 0) else None}
        if use_w:
            with np.errstate(invalid="ignore", divide="ignore"):
                reps = (weights @ a[num]) / (weights @ a[den]) - (weights @ b[num]) / (weights @ b[den])
            cell.update(ci95=_ci(reps), n_valid_reps=n_valid_reps(reps))
        diff[key] = cell
    for key, col in (("encounters_per_match", "enc"), ("killless_per_match", "kl"), ("with_kill_per_match", "wk")):
        cell = {"diff": float((a[col] - b[col]).mean()) if m else None}
        if use_w:
            cell["ci95"] = _ci((weights @ (a[col] - b[col])) / m)
        diff[key] = cell
    out["frame_only_minus_kill_adjusted"] = diff
    return out


def corpus_comparison(denom: Dict[str, Any], gate: Dict[str, Any], published: Optional[Dict[str, Any]],
                      same_sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    teamfight_gate = published is not None and gate["min_per_team"] >= int(published.get("teamfight_min", 4))
    kt = bool(gate.get("kill_trajectory_interp", True))
    out: Dict[str, Any] = {
        "published_corpus": published,
        "same_sample_corpus": same_sample,
        "comparable_corpus_class": ("teamfight_class (min side >= 4)" if teamfight_gate else "all engagements"),
        "caveats": [
            "encounters are proximity runs found without a kill; corpus engagements are kill-anchored clusters -- "
            "the two units are counted by different detectors, so their per-match rates are set side by side "
            "(side_by_side_per_match), never divided into each other",
            "published rates are per corpus match; encounter rates are per scanned cache match (see same_sample_corpus)",
            "killless_share_of_encounters is a share of proximity ENCOUNTERS, never of the 532,547 corpus engagements",
            "proximity is not commitment: do not divide per-match encounter rates into engagement counts to claim missed fights",
            ("the encounter counts come from the " + TRACK_SHORT[kt] + " 5 s track: on the kill-adjusted track (default, "
             "scanner and grid) with-kill encounters are detected on positions moved toward their kills while kill-less "
             "ones are detected on minute-frame interpolation, so the kill-less share is conditional on the track; "
             "quote it together with the other track (--track-sensitivity or a --kill-trajectory-interp off run)"),
        ],
    }
    klpm = denom.get("killless_per_match")
    side: Dict[str, Any] = {"killless_encounters_per_scanned_match": klpm,
                            "encounters_per_scanned_match": denom.get("encounters_per_match"),
                            "track": TRACK_SHORT[kt]}
    for label, ref in (("same_sample_corpus", same_sample), ("published_corpus", published)):
        if ref:
            side[f"{label}.engagements_per_match"] = ref.get("engagements_per_match")
            side[f"{label}.teamfight_class_per_match"] = ref.get("teamfight_class_per_match")
    out["side_by_side_per_match"] = side
    return out


def quote(gate: Dict[str, Any], denom: Dict[str, Any], comparison: Dict[str, Any],
          sensitivity: Optional[Dict[str, Any]] = None) -> str:
    share = denom.get("killless_share_of_encounters")
    same = comparison.get("same_sample_corpus") or {}
    pub = comparison.get("published_corpus") or {}
    ref = same or pub
    tf = gate["min_per_team"] >= int(ref.get("teamfight_min", 4)) if ref else False
    rate_key = "teamfight_class_per_match" if tf else "engagements_per_match"
    ci = denom.get("killless_share_ci") or [None, None]
    ci_txt = f" (95% match-clustered CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f} %)" if ci[0] is not None else ""
    track = TRACK_SHORT[bool(gate.get("kill_trajectory_interp", True))]
    text = (
        f"{(share or 0) * 100:.2f} %{ci_txt} of proximity encounters contain no kill "
        f"[gate {gate['name']} = grid {gate.get('grid_setting', gate['name'])}: >= {gate['min_per_team']} alive per side "
        f"within {gate['radius_u']:.0f} u of one champion for >= {gate['min_active_frames']} consecutive 5 s frames "
        f"of the {track} track; no CHAMPION_KILL within +-{gate['grace_ms'] / 1000:g} s; preset {gate['preset']}; "
        f"{denom['matches_scanned']:,} matches]: {denom['killless']:,} of {denom['encounters']:,} encounters, "
        f"{(denom['killless_per_match'] or 0):.4f} kill-less encounters per match"
    )
    other = None
    sens_txt = ""
    if sensitivity:
        other = next((k for k in ("kill-adjusted", "frame-only") if k != track and k in sensitivity), None)
    if other:
        o = sensitivity[other]
        oci = o.get("killless_share_ci") or [None, None]
        oci_txt = f" (CI {oci[0] * 100:.2f}-{oci[1] * 100:.2f} %)" if oci[0] is not None else ""
        d = (sensitivity.get("frame_only_minus_kill_adjusted") or {}).get("killless_share_of_encounters") or {}
        dci = d.get("ci95") or [None, None]
        d_txt = (f"; frame-only minus kill-adjusted share {d['diff'] * 100:+.2f} points (paired CI {dci[0] * 100:+.2f} "
                 f"to {dci[1] * 100:+.2f})" if (d.get("diff") is not None and dci[0] is not None) else "")
        sens_txt = (f".  On the {other} track of the same matches: {(o.get('killless_share_of_encounters') or 0) * 100:.2f} %"
                    f"{oci_txt}, {o['killless']:,} of {o['encounters']:,} encounters, "
                    f"{(o.get('killless_per_match') or 0):.4f} kill-less encounters per match{d_txt}")
    cls = "teamfight-class " if tf else ""
    against: List[str] = []
    if same.get(rate_key) is not None:
        against.append(f"{same[rate_key]:.4f} corpus {cls}engagements per scanned match")
    if pub.get(rate_key) is not None:
        n_key = "n_teamfight_class" if tf else "n_engagements"
        against.append(f"{pub[rate_key]:.4f} corpus {cls}engagements per corpus match "
                       f"({int(pub[n_key]):,} / {int(pub['n_matches']):,})")
    if against:
        text += " against " + "; ".join(against)
    m2 = (denom.get("by_start_band") or {}).get("from_minute_2") or {}
    if m2.get("killless_share_of_encounters") is not None:
        text += (f"; from minute 2 on: {m2['killless_share_of_encounters'] * 100:.2f} % "
                 f"({m2['killless']:,} of {m2['encounters']:,})")
    return text + sens_txt + "."


# --------------------------------------------------------------------------------------------
# reproduction of the finished v3.3 grid
# --------------------------------------------------------------------------------------------
_PROGRESS_RE = re.compile(r"^\s*(\d+)/(\d+) matches \| encounters=(\d+) killless=([0-9.]+)%\s*$")


def _track_contrast(row: Optional[Dict[str, Any]], log: Dict[int, Dict[str, Any]], denom: Dict[str, Any],
                    checkpoints: Dict[int, Dict[str, int]], full_sample_scanned: bool) -> Dict[str, Any]:
    """This frame-only run next to the kill-adjusted grid on the same matches (unpaired point values)."""
    out: Dict[str, Any] = {"note": "kill-adjusted = the grid (scanner default); frame-only = this run; same sample, "
                                   "point values only"}
    if full_sample_scanned and row is not None:
        keys = ("matches", "encounters", "with_kill", "killless", "killless_share_of_encounters",
                "encounters_per_match", "killless_per_match", "killless_median_duration_s")
        out["mode"] = "full_sample"
        out["kill_adjusted_grid_row"] = {k: row.get(k) for k in keys}
        out["frame_only_this_run"] = {"matches": denom["matches_scanned"], **{k: denom.get(k) for k in keys[1:]}}
    else:
        out["mode"] = "prefix"
        out["checkpoints"] = {
            str(done): {"kill_adjusted_grid_log": {"encounters": log[done]["encounters"],
                                                   "killless_pct": log[done]["killless_pct"]},
                        "frame_only_this_run": {"encounters": cp["encounters"], "killless": cp["killless"],
                                                "killless_pct": progress_pct(cp["killless"], cp["encounters"])}}
            for done, cp in sorted(checkpoints.items()) if done in log}
    return out


def parse_grid_log(path: Path) -> Dict[int, Dict[str, Any]]:
    """Progress lines of a scanner log: done -> {n_sampled, encounters, killless_pct (one-decimal string)}."""
    out: Dict[int, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PROGRESS_RE.match(line)
        if m:
            out[int(m.group(1))] = {"n_sampled": int(m.group(2)), "encounters": int(m.group(3)),
                                    "killless_pct": m.group(4)}
    return out


def progress_pct(killless: int, encounters: int) -> str:
    """The scanner's progress formatting of the kill-less share."""
    return f"{killless / max(1, encounters) * 100:.1f}"


def _same_number(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


KT_OFF_REASON = ("kill-trajectory interpolation is off (frame-only 5 s track); the grid ran the scanner with the default "
                 "TF2_USE_KILL_TRAJECTORY_INTERP=True")


def grid_reproduction(gate: Dict[str, Any], denom: Dict[str, Any], checkpoints: Dict[int, Dict[str, int]], *,
                      n_sampled: int, seed: int, match_source: str, patches_filter: bool,
                      full_sample_scanned: bool, grid_dir: Optional[Path], scan_offset: int = 0) -> Dict[str, Any]:
    """Compare this run with the grid row of the same setting (full sample) or with its log (prefix).

    A frame-only track (gate kill_trajectory_interp False) or a scan that does not start at the first sampled
    match is not_comparable.  For the frame-only track the kill-adjusted grid row of the same constants (and,
    in prefix mode, its log checkpoints) is set next to this run under ``track_contrast`` when nothing else
    differs; that contrast is unpaired (the grid kept no per-match counts) -- use --track-sensitivity for CIs.
    """
    setting = gate.get("grid_setting") or gate["name"]
    row_setting = gate.get("grid_row_same_constants") or setting
    kt_on = bool(gate.get("kill_trajectory_interp", True))
    out: Dict[str, Any] = {"setting": setting, "grid_dir": str(grid_dir) if grid_dir else None}
    if not grid_dir or not (Path(grid_dir) / "summary.json").exists():
        out["status"] = "no_grid"
        return out
    grid_dir = Path(grid_dir)
    summary = json.loads((grid_dir / "summary.json").read_text(encoding="utf-8"))
    out["grid_git_commit"] = summary.get("git_commit")
    seed_match = re.search(r"seed\s+(\d+)", str(summary.get("what", "")))
    grid_seed = int(seed_match.group(1)) if seed_match else None
    row = next((r for r in summary.get("rows", []) if r.get("setting") == row_setting), None)
    log_path = grid_dir / f"{row_setting}.log"
    log = parse_grid_log(log_path)
    reasons: List[str] = []
    if not kt_on:
        reasons.append(KT_OFF_REASON)
    if int(scan_offset) > 0:
        reasons.append(f"the scan starts at sample position {int(scan_offset) + 1}; the grid log's cumulative counts "
                       "start at the first sampled match")
    if row is None:
        reasons.append(f"no grid row named {row_setting}")
    else:
        for key, mine in (("radius_u", gate["radius_u"]), ("min_per_team", gate["min_per_team"]),
                          ("min_duration_s", gate["min_duration_s"]), ("grace_ms", gate["grace_ms"])):
            if not _same_number(row.get(key), mine):
                reasons.append(f"grid row {key}={row.get(key)} but gate {key}={mine}")
    if match_source != "cache":
        reasons.append("match source is not the cache pool the grid sampled")
    if patches_filter:
        reasons.append("a patch filter is applied")
    if grid_seed is not None and int(seed) != grid_seed:
        reasons.append(f"seed {seed} but grid seed {grid_seed}")
    grid_n_sampled = next(iter(log.values()))["n_sampled"] if log else (int(row["matches"]) if row else None)
    if grid_n_sampled is not None and int(n_sampled) != int(grid_n_sampled):
        reasons.append(f"sample of {n_sampled} matches but the grid sampled {grid_n_sampled}")
    out["grid_seed"] = grid_seed
    out["grid_n_sampled"] = grid_n_sampled
    if reasons:
        out.update(status="not_comparable", reasons=reasons)
        if reasons == [KT_OFF_REASON]:
            out["track_contrast"] = _track_contrast(row, log, denom, checkpoints, full_sample_scanned)
        return out
    if full_sample_scanned:
        fields = {
            "matches": denom["matches_scanned"], "encounters": denom["encounters"], "with_kill": denom["with_kill"],
            "killless": denom["killless"], "killless_share_of_encounters": denom["killless_share_of_encounters"],
            "encounters_per_match": denom["encounters_per_match"], "killless_per_match": denom["killless_per_match"],
            "killless_median_duration_s": denom["killless_median_duration_s"],
        }
        mism = {k: {"grid": row.get(k), "this_run": v} for k, v in fields.items() if not _same_number(row.get(k), v)}
        out.update(mode="full_sample", compared_fields=sorted(fields), grid_row=row, mismatches=mism,
                   status="reproduced" if not mism else "MISMATCH")
        return out
    compared: List[int] = []
    mism: Dict[str, Any] = {}
    last_done = max(checkpoints) if checkpoints else 0
    for done, cp in sorted(checkpoints.items()):
        ref = log.get(done)
        if ref is None:
            continue
        mine_pct = progress_pct(cp["killless"], cp["encounters"])
        compared.append(done)
        if ref["encounters"] != cp["encounters"] or ref["killless_pct"] != mine_pct:
            mism[str(done)] = {"grid": {"encounters": ref["encounters"], "killless_pct": ref["killless_pct"]},
                               "this_run": {"encounters": cp["encounters"], "killless": cp["killless"],
                                            "killless_pct": mine_pct}}
    only_run = sorted(d for d in checkpoints if d not in log)
    only_log = sorted(d for d in log if d <= last_done and d not in checkpoints)
    if only_run or only_log:
        mism["checkpoints_missing"] = {"only_in_this_run": only_run, "only_in_grid_log": only_log}
    status = "prefix_reproduced" if (compared and not mism) else ("MISMATCH" if mism else "no_overlapping_checkpoints")
    out.update(mode="prefix", log=str(log_path), checkpoints_compared=compared, mismatches=mism, status=status,
               note="the log gives cumulative encounters and a one-decimal kill-less percentage per 200 sampled matches; "
                    "the kill-less count is pinned exactly while cumulative encounters stay below 1,000")
    return out


def frequency_blocks(tally: ScanTally, gate: Dict[str, Any], *, n_boot: int, seed: int,
                     published: Optional[Dict[str, Any]], shard_counts, shards_dir, grid_dir: Optional[Path],
                     n_sampled: int, full_sample_scanned: bool, match_source: str, patches_filter: bool,
                     weights: Optional[np.ndarray] = None, scan_offset: int = 0,
                     alt_tally: Optional[ScanTally] = None) -> Dict[str, Any]:
    """denominator, corpus_comparison, grid_reproduction, track_sensitivity and quote for one scan (shared with
    the characterization)."""
    enc_b, kl_b = tally.band_arrays()
    m = len(tally.match_ids)
    if weights is None and n_boot > 0 and m:
        weights = cluster_weights(m, n_boot, seed)
    denom = denominator_block(np.asarray(tally.n_enc, dtype=float), np.asarray(tally.n_kl, dtype=float),
                              tally.durations_killless, n_boot=n_boot, seed=seed, enc_bands=enc_b, kl_bands=kl_b,
                              weights=weights, kt_enc=np.asarray(tally.kt_enc, dtype=float),
                              kt_kl=np.asarray(tally.kt_kl, dtype=float), kt_missing=tally.kt_missing)
    same = same_sample_corpus(shard_counts, tally.match_ids, shards_dir, published)
    comparison = corpus_comparison(denom, gate, published, same)
    grid = grid_reproduction(gate, denom, tally.checkpoints, n_sampled=n_sampled, seed=seed, match_source=match_source,
                             patches_filter=patches_filter, full_sample_scanned=full_sample_scanned, grid_dir=grid_dir,
                             scan_offset=scan_offset)
    sensitivity = None
    if alt_tally is not None:
        sensitivity = track_sensitivity_block(tally, alt_tally, bool(gate.get("kill_trajectory_interp", True)), weights)
    return {"denominator": denom, "corpus_comparison": comparison, "grid_reproduction": grid,
            "track_sensitivity": sensitivity, "quote": quote(gate, denom, comparison, sensitivity)}


def set_track(cfg_obj, kill_trajectory_interp: bool) -> Dict[str, Any]:
    """Set cfg.TF2_USE_KILL_TRAJECTORY_INTERP before any scan; returns the before / after values for provenance."""
    before = getattr(cfg_obj, "TF2_USE_KILL_TRAJECTORY_INTERP", None)
    setattr(cfg_obj, "TF2_USE_KILL_TRAJECTORY_INTERP", bool(kill_trajectory_interp))
    return {"TF2_USE_KILL_TRAJECTORY_INTERP_before": before,
            "TF2_USE_KILL_TRAJECTORY_INTERP": bool(getattr(cfg_obj, "TF2_USE_KILL_TRAJECTORY_INTERP"))}


def parse_scan_slice(text: str, n: int) -> Tuple[int, int]:
    """'START:END' (python slice bounds over the sorted sample; negative START counts from the end)."""
    parts = str(text).split(":")
    if len(parts) != 2:
        raise SystemExit(f"--scan-slice expects START:END, got {text!r}")
    lo = int(parts[0]) if parts[0].strip() else 0
    hi = int(parts[1]) if parts[1].strip() else n
    lo, hi, _ = slice(lo, hi).indices(n)
    if hi <= lo:
        raise SystemExit(f"--scan-slice {text!r} selects no match of {n}")
    return lo, hi


def add_scan_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared by this wrapper and the characterization (track, slice)."""
    ap.add_argument("--kill-trajectory-interp", choices=("on", "off"), default="on",
                    help="5 s track: 'on' = kill-adjusted (cfg default, scanner and grid); 'off' = frame-only "
                         "(gate suffix _ktoff; grid reproduction not_comparable)")
    ap.add_argument("--scan-slice", default=None,
                    help="scan only sample[START:END] of the sorted sample (e.g. 10000:10150, or --scan-slice=-200: for the last 200); grid "
                         "reproduction is then not_comparable unless START is 0")


def apply_preset_if_needed(cfg_obj, preset: str) -> Tuple[str, bool]:
    preset_env = str(os.environ.get("LOL_CFG_PRESET", "")).strip()
    if preset_env != preset:
        from core.presets import apply_preset
        apply_preset(cfg_obj, preset)
        return preset_env, True
    return preset_env, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="v3.3", help="core.presets name supplying R, the grace window and G")
    ap.add_argument("--radius", type=float, default=None, help="override R (default: preset TF2_VALIDITY_RADIUS)")
    ap.add_argument("--grace-ms", type=int, default=None, help="override the grace window (default: preset TF2_ENGAGE_PRE_KILL_MS)")
    ap.add_argument("--min-per-team", type=int, default=4, help="alive champions per side (default 4 = v3.3 teamfight class)")
    ap.add_argument("--min-duration", type=float, default=None, help="seconds (default: preset G)")
    ap.add_argument("--n-matches", type=int, default=20000,
                    help="matches sampled from the pool (default 20,000 = the v3.3 grid's sample; the v2 scanner defaults to 2,000)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scan-first", type=int, default=None,
                    help="scan only the first K matches of the sorted sample (prefix check against the grid log)")
    ap.add_argument("--match-source", choices=("cache", "corpus"), default="cache",
                    help="sample from every cached match (scanner behaviour) or only from corpus matches")
    ap.add_argument("--patches", default="", help="comma-separated patches to keep (default: all)")
    ap.add_argument("--scale-json", type=Path, default=DEFAULT_SCALE_JSON)
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS, help="corpus shards for same-sample counts ('' to skip)")
    ap.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR,
                    help="finished v3.3 grid (summary.json and <setting>.log) to check reproduction against ('' to skip)")
    ap.add_argument("--track-sensitivity", action="store_true",
                    help="also scan the other 5 s track on the same matches and write the paired track_sensitivity block")
    add_scan_args(ap)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--progress-every", type=int, default=500)
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
    gate = resolve_gate(args.preset, args.radius, args.grace_ms, args.min_per_team, args.min_duration,
                        kill_trajectory_interp=kt_on)

    from core.config import CACHE_DIR, NODE_IDX, cfg

    preset_env, applied_in_process = apply_preset_if_needed(cfg, args.preset)
    track_setting = set_track(cfg, kt_on)
    scanner = load_scanner()
    if int(getattr(cfg, "TF2_GRID_STEP_MS", 5000)) != int(scanner.GRID_STEP_MS):
        raise SystemExit(f"cfg.TF2_GRID_STEP_MS={cfg.TF2_GRID_STEP_MS} differs from the scanner's GRID_STEP_MS={scanner.GRID_STEP_MS}")

    shard_counts = None
    if str(args.shards) and Path(args.shards).exists():
        shard_counts = corpus_counts_by_match(args.shards)
    published = published_corpus_rates(args.scale_json)
    mids, pool = sample_match_ids(CACHE_DIR, args.n_matches, args.seed, args.match_source,
                                  sorted(shard_counts) if shard_counts else None)
    lo, hi = parse_scan_slice(args.scan_slice, len(mids)) if args.scan_slice is not None else (
        0, len(mids) if (not args.scan_first or args.scan_first >= len(mids)) else int(args.scan_first))
    to_scan = mids[lo:hi]
    keep_patches = frozenset(p.strip() for p in str(args.patches).split(",") if p.strip())
    print(f"gate {gate['name']} (grid {gate['grid_setting']}; {gate['track']} track) | pool={pool} sampled={len(mids)} "
          f"scanning=[{lo}:{hi}] ({len(to_scan)}) | source={args.match_source} | track_sensitivity={args.track_sensitivity}",
          flush=True)

    tally = ScanTally()
    alt_tally = ScanTally() if args.track_sensitivity else None
    for _ in iter_scanned(to_scan, gate, cfg, NODE_IDX, tally, keep_patches, args.progress_every, started,
                          start=lo, alt_tally=alt_tally):
        pass
    grid_dir = Path(args.grid_dir) if str(args.grid_dir) else None
    blocks = frequency_blocks(tally, gate, n_boot=args.n_boot, seed=args.seed, published=published,
                              shard_counts=shard_counts, shards_dir=args.shards, grid_dir=grid_dir, n_sampled=len(mids),
                              full_sample_scanned=(lo == 0 and hi == len(mids)), match_source=args.match_source,
                              patches_filter=bool(keep_patches), scan_offset=lo, alt_tally=alt_tally)
    patches = tally.patch_counts()
    wall = time.time() - started
    results = {
        "item": "A5-killless",
        "script": "scripts/run_killless_v33.py",
        "provenance": {
            **git_state(), "code": code_provenance(), "preset": args.preset, "cfg_preset_env": preset_env,
            "preset_applied_in_process": applied_in_process,
            "label_key": "none (frequency only; no label is computed)",
            "split": "none (descriptive, no model); patches in sample: " + json.dumps(patches, sort_keys=True),
            "seed": args.seed, "bootstrap_seed": args.seed, "n_boot": args.n_boot,
            "n_matches_requested": args.n_matches, "match_source": args.match_source,
            "match_pool": pool, "matches_sampled": len(mids), "scan_first": args.scan_first,
            "scan_slice": [lo, hi], "track_sensitivity": bool(args.track_sensitivity),
            "matches_attempted": len(to_scan), "matches_scanned": len(tally.match_ids), "n_rows": int(sum(tally.n_enc)),
            "n_missing_cache": tally.n_missing, "n_failed": tally.n_failed, "n_patch_skipped": tally.n_patch_skipped,
            "n_alt_track_failed": (alt_tally.n_alt_failed if alt_tally is not None else None),
            "cache_dir": str(CACHE_DIR), "feature_version": str(getattr(cfg, "FEATURE_VERSION", "")),
            "grid": {"TF2_GRID_STEP_MS": int(getattr(cfg, "TF2_GRID_STEP_MS", 5000)),
                     "TF2_USE_FRAME_INTERP": bool(getattr(cfg, "TF2_USE_FRAME_INTERP", True)),
                     **track_setting},
            "argv": list(sys.argv if argv is None else argv), "started_at": started_at, "wall_clock_s": round(wall, 1),
        },
        "gate": gate,
        **blocks,
        "patch_distribution": patches,
        "reference": "scripts/run_killless_encounters.py encounters_for_match + main() preparation; core/presets.py "
                     f"{args.preset}; scale_decomposition_v33_market_event.json; tog_revision/killless_grid",
        "deviations": WRAPPER_DEVIATIONS,
    }
    print(json.dumps({"gate": gate["grid_setting"], "denominator": {k: v for k, v in blocks["denominator"].items()
                                                                     if k != "by_start_band"},
                      "grid_reproduction": {k: blocks["grid_reproduction"].get(k) for k in
                                            ("status", "mode", "checkpoints_compared", "mismatches", "reasons",
                                             "track_contrast")},
                      "track_sensitivity": blocks["track_sensitivity"],
                      "quote": blocks["quote"]}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
