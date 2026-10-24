"""T5.2 E5 case figures: three real Q_DEV engagements on a time axis and a minimap.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E5_KO.md (sources section 2, selection section 3,
figures and map check section 4 with SDD Rulings 26 and 26b, BLOCKED section 6, outputs section 9); PLAN_E1_E5_KO.md
Task 6; SDD Rulings 20, 25, 32, 33 and 34 (fix round 1).  Q_DEV (15.15) only: arrays that hold every q role are
restricted to the 107,970 Q_DEV rows as soon as they are read, and no TEST value enters a computation or an output.
The figures are illustrations, not statistical evidence (PRESPEC_E5 section 1).

One process, run once, at below-normal priority, CPU only (main() hides CUDA before torch loads; the final V on
"cpu").  Every output is written once under D:/LOL_Training_20261004/tog_e5_2 (tog_e5_1 holds the superseded first
run, Ruling 34).  An error after the map check is recorded as E5_FAILED_<stage>_<n>.json and raised again.

  map     The minimap transform X = (29 + (g_x + 120) 532/14990) 512/586, Y = (561 - (g_y + 120) 532/15100) 512/586
          (pixels, origin top left; the plan's 586-pixel deck formula scaled to the 512 x 512 map image) is checked on
          reports/thesis_manuscript_20261010/latex/figures/summoners_rift_1514.png before anything else (Rulings 26,
          26b): lane colour = median RGB of the 9 x 9 block at (256, 256); a lane-colour pixel is within RGB distance
          40 of it; (i) the border (first lane-colour pixel scanning inward along row 256 and column 256) lies within
          10.24 px (2 % of 512) of the formula extent 25.34 / 490.16 px; (ii) each of the eight turrets on the drawn
          lanes (six outer, two mid inner; game coordinates from the old-lineage map_anchors.json, equal to the
          PRESPEC_E5 table) lies within 10.24 px of a lane-colour pixel.  Pixels are integer (column, row) indices.
          The top and bottom inner turrets are drawn and measured but not judged (Ruling 26b).  Writes MAP_CHECK.json
          and MAP_CHECK.png; any failure is BLOCKED (E5_BLOCKED_map_<n>.json) and the run stops.
  gate    PRESPEC_E5 section 5: the time-axis states rest on the Task 2 equality gate.  tog_e1_1/EQUALITY_GATE_TOG.json
          must have passed and every code file it recorded must be unchanged, except common.py (extended for E2 in
          3423d6f with helpers that build no state).  Otherwise BLOCKED.
  table   The Q_DEV table of Ruling 32 from the PRESPEC_E5 section 2 sources, aligned by engagement key and checked
          (assemble_table); the KA_LABELS key order must equal the q rows (r10_step1b.py:69-70), else BLOCKED.  Both
          prediction files must reproduce their search job's recorded Q_DEV match-weighted Brier (RESULT.json), file
          hash and key hash; rows and matches must be 107,970 and 28,566.
  select  e5_cases.select_cases (rules (a), (b), (c), ties by key hash, distinct cases).
  cases   Per case: states every 5 s from tau - 60 s to the h90 endpoint, plus the endpoint itself (time_grid), built
          with controls_states.build_match + semantic_rows (= rows_at_times, the path the equality gate checked) and
          scored with the calibrated final V (score.LockedV "final" through queries.transform, as the labels); V at
          tau and at the endpoint must reproduce the labels' p_pre and p_post.  Kills from the adopted population
          code on the reopened match; F(tau) positions by e1_gate.xy_at_times (spatial.py's frame rules).  A player
          whose F(tau) position equals its own last death position at or before the frame is marked likely dead and
          drawn hollow grey (Ruling 34: the data carry no alive flag).  The h90 endpoint's reasons come from the label's
          truncation bits and must equal the population row's.  Labels are placed so that they do not overlap.
          case_<r>.pdf.
  facts   E5_CASES.json: the keys, rules and threshold, per case p_q, p_pt, y, svi, kill counts, kill times,
          positions with champion names (no player identifier: teams and champions only), the closer prediction,
          the time axis values and its y-axis span, the endpoint reasons, the likely-dead marks, the selection audit
          (per rule: candidates, chosen and runner-up keys and scores, margin; rule (c)'s R10 population), library
          versions, input and code SHA-256, UTC times, stage wall times and peak memory.

Usage: python -B -m studies.tog_experiments_20261012.e5_figures [--out DIR]   (DIR for a dry run outside tog_e5_2)
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Iterable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005.engagements import TRUNCATION  # noqa: E402
from studies.q_production_20261005.qtest import read_json  # noqa: E402
from studies.q_production_20261005.qtrain import match_brier  # noqa: E402
from studies.tog_experiments_20261012 import e5_cases  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TOG_E1, TRAINING, document_record, git_head, peak_memory_bytes, run_record, sha256_file, utc,
    write_once)

VERSION = "tog_e5_figures_v2_20261012"
TOG_E5 = TRAINING / "tog_e5_2"                   # fix round 1 (Ruling 34); tog_e5_1 is the superseded first run
SUPERSEDES = {"folder": str(TRAINING / "tog_e5_1"),
              "reason": "Ruling 34 (Task 6 fix round 1): the first run drew players who were most likely dead at F(tau) "
                        "as present, at their own last death positions; this run marks them hollow grey and adds the "
                        "review minors (label placement, endpoint reasons, selection audit, y-axis span, kill_side_blue "
                        "only for single-kill cases, library versions). Selection and every number are unchanged."}
PRESPEC = ROOT / "reports/tog_experiments_20261012/PRESPEC_E5_KO.md"
MAP_IMAGE = ROOT / "reports/thesis_manuscript_20261010/latex/figures/summoners_rift_1514.png"
MAP_ANCHORS = Path(r"C:\Users\todtj\PycharmProjects\LOL_teamfight\config\game_rules\map_anchors.json")
GATE = TOG_E1 / "EQUALITY_GATE_TOG.json"
GATE_IGNORE = {"studies/tog_experiments_20261012/common.py":
               "extended for E2 in 3423d6f (TOG_E2, write_npz_once, peak_memory_bytes, git helpers, document_record); "
               "it builds no state"}
Q_ROWS = Q_RUN / "Q_ROWS_h90.npz"
LABEL_TABLE = LABELS / "LABELS_h90.npz"
KA_LABELS = TRAINING / "q_ka_1" / "KA_LABELS.npz"
OBJECTIVE_COUNTS = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1/OBJECTIVE_COUNTS.npy"
RUN_3 = TRAINING / "run_3"
JOBS = {"P1": "search__auxiliary__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__lgbm_l15_m200",     # PRESPEC_E5 line 13
        "PT_flex": "search__reference__PT_FLEX__PT_flex__ptflex_k8_C0.001"}
Q_DEV, PATCH = 3, "15.15"
EXPECTED_ROWS, EXPECTED_MATCHES = 107_970, 28_566                     # Q_ROWS_h90.json counts.Q_DEV
BRIER_ATOL = 1e-12            # recomputed vs recorded Q_DEV Brier (same arrays, same order)
V_REPRO_ATOL = 1e-5           # rescored V at tau / endpoint vs the labels' p_pre / p_post (labels were scored on CUDA)
MAP_SIDE, DECK_SIDE = 512, 586
SCALE = MAP_SIDE / DECK_SIDE
TOLERANCE_PX = 0.02 * MAP_SIDE                                        # 10.24 px
LANE_RGB_DISTANCE = 40
CENTRE, BLOCK = 256, 9
LEAD_MS, STEP_MS = 60_000, 5_000
MAP_JSON, MAP_PNG, CASES_JSON = "MAP_CHECK.json", "MAP_CHECK.png", "E5_CASES.json"
RULES = ("a", "b", "c")
RULE_NAMES = {"a": "confidently wrong", "b": "apart from the state baseline", "c": "monotonicity violation"}
RULE_TEXT = {  # PRESPEC_E5 section 3
    "a": "|p_q - y| >= 0.8 and eval_n_kills >= 2, largest |svi|; with none, the threshold drops to 0.7, then 0.6 "
         "(the first non-empty set is used and its threshold reported)",
    "b": "|p_q - y| < |p_pt - y|, largest |p_q - p_pt|",
    "c": "R10 primary definition: F_pre_ms == F_post_ms, KA blue_kills + red_kills == 1, OBJECTIVE_COUNTS equal at "
         "query_pre and query_post, s * svi < 0 (s = +1 for a blue kill, -1 otherwise; so svi != 0), largest |svi|",
}
TIE_TEXT = ("ties: ascending hashlib.sha256(engagement_key.decode('ascii').encode('utf-8')).hexdigest(); the cases are "
            "distinct engagements chosen in the order (a), (b), (c), a later rule dropping those already chosen "
            "(Ruling 25)")
SIGN_TEXT = "svi = V(end) - V(start), V = blue win probability: positive is a move toward blue"
TEAM_NAMES = {100: "blue", 200: "red"}
COLOURS = {"blue": "#1f6fd1", "red": "#d6332f", "environment": "#7a7a7a", None: "#7a7a7a"}
DEAD_COLOUR = "#b4b4b4"         # hollow marker and label of a player marked likely dead at F(tau) (Ruling 34)
LIKELY_DEAD_RULE = ("a player is marked likely dead at F(tau) when its F(tau) position equals, exactly, the recorded "
                    "position of its own most recent death (CHAMPION_KILL victim) at or before the frame time; the data "
                    "carry no alive flag, so this is a position-equals-death-position heuristic, not a recorded status "
                    "(Ruling 34)")
Q_FIELDS = ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "query_pre", "pre_ms", "y",
            "label_svi_delta", "eval_n_kills", "weight_raw", "truncation_bits")
LABEL_FIELDS = ("role", "canonical_match_key", "component_key", "valid", "direction_positive", "svi_delta", "F_pre_ms",
                "F_post_ms", "query_pre", "query_post", "pre_ms", "post_ms", "first_kill_ms", "last_kill_ms", "p_pre",
                "p_post", "match_index", "truncation_bits")
CODE_FILES = (
    "studies/tog_experiments_20261012/e5_figures.py", "studies/tog_experiments_20261012/e5_cases.py",
    "studies/tog_experiments_20261012/e1_gate.py", "studies/tog_experiments_20261012/common.py",
    "studies/tog_experiments_20261012/__init__.py", "studies/label_production_20261005/score.py",
    "studies/observable_state_20261002/population.py", "studies/v_production_20261004/train.py",
    "studies/v_training_gate_20261003/guard.py", "studies/v_full_preprocess_20261003/artifacts.py",
    "studies/q_production_20261005/qtrain.py", "studies/q_production_20261005/qtest.py",
    "studies/q_production_20261005/spatial.py", "studies/cache_source_audit_20261002/config.py",
)


def _turret(team: int, lane: str, tower: str, gx: int, gy: int, px: float, py: float) -> dict[str, Any]:
    return {"team": team, "lane": lane, "tower": tower, "game_xy": (gx, gy), "pixel_xy": (px, py),
            "checked": tower == "OUTER_TURRET" or lane == "MID_LANE"}


PRESPEC_TURRETS = (  # PRESPEC_E5 line 38: game coordinates and transformed pixels; checked = six outer + two mid inner
    _turret(100, "BOT_LANE", "OUTER_TURRET", 10504, 1029, 354.77, 454.79),
    _turret(100, "BOT_LANE", "INNER_TURRET", 6919, 1483, 243.61, 440.81),
    _turret(100, "MID_LANE", "OUTER_TURRET", 5846, 6396, 210.34, 289.58),
    _turret(100, "MID_LANE", "INNER_TURRET", 5048, 4812, 185.59, 338.34),
    _turret(100, "TOP_LANE", "OUTER_TURRET", 981, 10441, 59.48, 165.06),
    _turret(100, "TOP_LANE", "INNER_TURRET", 1512, 6699, 75.94, 280.25),
    _turret(200, "BOT_LANE", "OUTER_TURRET", 13866, 4505, 459.02, 347.79),
    _turret(200, "BOT_LANE", "INNER_TURRET", 13327, 8226, 442.31, 233.24),
    _turret(200, "MID_LANE", "OUTER_TURRET", 8955, 8510, 306.74, 224.50),
    _turret(200, "MID_LANE", "INNER_TURRET", 9767, 10113, 331.92, 175.16),
    _turret(200, "TOP_LANE", "OUTER_TURRET", 4318, 13875, 162.95, 59.35),
    _turret(200, "TOP_LANE", "INNER_TURRET", 7943, 13411, 275.36, 73.64),
)


class Blocked(RuntimeError):
    """A PRESPEC_E5 section 6 condition (or a map-check precondition): the run writes it and stops."""

    def __init__(self, stage: str, reasons: Mapping[str, Any]) -> None:
        super().__init__(f"BLOCKED at {stage}: {sorted(reasons)}")
        self.stage, self.reasons = stage, dict(reasons)


# ---------------------------------------------------------------- minimap transform and map check

def minimap_xy(gx: Any, gy: Any) -> tuple[np.ndarray, np.ndarray]:
    """Game coordinates -> pixels on the 512 x 512 map image (origin top left), PRESPEC_E5 line 35."""

    gx, gy = np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)
    return (29.0 + (gx + 120.0) * 532.0 / 14990.0) * SCALE, (561.0 - (gy + 120.0) * 532.0 / 15100.0) * SCALE


def lane_colour(image: np.ndarray) -> tuple[int, int, int]:
    """Median RGB of the 9 x 9 block centred on pixel (256, 256) (81 values, so each median is a pixel value)."""

    half = BLOCK // 2
    block = np.asarray(image)[CENTRE - half:CENTRE + half + 1, CENTRE - half:CENTRE + half + 1, :3].reshape(-1, 3)
    return tuple(int(round(float(v))) for v in np.median(block, axis=0))


def lane_mask(image: np.ndarray, colour: Sequence[int], max_distance: float = LANE_RGB_DISTANCE) -> np.ndarray:
    """True where a pixel's RGB distance to ``colour`` is at most ``max_distance``."""

    diff = np.asarray(image)[..., :3].astype(np.int64) - np.asarray(colour, dtype=np.int64)
    return (diff * diff).sum(axis=-1) <= max_distance * max_distance


def measure_border(mask: np.ndarray, centre: int = CENTRE) -> dict[str, int | None]:
    """First lane pixel scanning inward along row ``centre`` (left, right) and column ``centre`` (top, bottom)."""

    row, col = np.flatnonzero(mask[centre, :]), np.flatnonzero(mask[:, centre])
    return {"left": int(row[0]) if row.size else None, "right": int(row[-1]) if row.size else None,
            "top": int(col[0]) if col.size else None, "bottom": int(col[-1]) if col.size else None}


def nearest_lane_pixel(mask: np.ndarray, x: float, y: float) -> tuple[float, tuple[int, int] | None]:
    """(distance, (column, row)) of the lane pixel nearest to point (x, y); pixels sit at integer indices."""

    rows, cols = np.nonzero(mask)
    if not rows.size:
        return float("inf"), None
    d2 = (cols - float(x)) ** 2 + (rows - float(y)) ** 2
    i = int(np.argmin(d2))
    return float(math.sqrt(float(d2[i]))), (int(cols[i]), int(rows[i]))


def lane_turrets(anchors: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The twelve lane turrets (outer and inner, three lanes) of map_anchors.json in PRESPEC_E5 order, with their
    exact transformed pixels.  Coordinates or pixels that differ from the PRESPEC_E5 table are BLOCKED."""

    found: dict[tuple[int, str, str], tuple[int, int]] = {}
    problems: dict[str, Any] = {}
    for row in anchors["towers"]:
        if row.get("building") == "TOWER_BUILDING" and row.get("tower") in ("OUTER_TURRET", "INNER_TURRET"):
            key = (int(row["team"]), str(row["lane"]), str(row["tower"]))
            if key in found:
                problems[f"repeated:{key}"] = True
            found[key] = (int(row["x"]), int(row["y"]))
    turrets = []
    for spec in PRESPEC_TURRETS:
        key = (spec["team"], spec["lane"], spec["tower"])
        xy = found.pop(key, None)
        x, y = minimap_xy(*spec["game_xy"])
        pixel = (round(float(x), 2), round(float(y), 2))
        if xy != spec["game_xy"] or pixel != spec["pixel_xy"]:
            problems[":".join(map(str, key))] = {"anchors": xy, "prespec": spec["game_xy"], "pixel": pixel,
                                                 "prespec_pixel": spec["pixel_xy"]}
        turrets.append({**spec, "game_xy": list(spec["game_xy"]), "pixel_xy": list(spec["pixel_xy"]),
                        "pixel_xy_exact": [float(x), float(y)]})
    if found:
        problems["not_in_prespec"] = sorted(":".join(map(str, key)) for key in found)
    if problems:
        raise Blocked("map", {"map_anchors_differ_from_prespec": problems})
    return turrets


def map_check(image: np.ndarray, turrets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rulings 26 and 26b on the map image: (i) border within 10.24 px of the formula extent, (ii) each checked turret
    within 10.24 px of a lane-colour pixel.  ``passed`` is False with the failing parts in ``failures``."""

    image = np.asarray(image)
    if image.shape[:2] != (MAP_SIDE, MAP_SIDE):
        raise Blocked("map", {"image_size": list(image.shape[:2])})
    colour = lane_colour(image)
    half = BLOCK // 2
    block = image[CENTRE - half:CENTRE + half + 1, CENTRE - half:CENTRE + half + 1, :3].reshape(-1, 3)
    mask = lane_mask(image, colour)
    measured = measure_border(mask)
    left, bottom = minimap_xy(-120.0, -120.0)
    right, top = minimap_xy(14_870.0, 14_980.0)
    formula = {"left": float(left), "right": float(right), "top": float(top), "bottom": float(bottom)}
    failures, border, rows = [], {}, []
    for side, value in formula.items():
        got = measured[side]
        diff = None if got is None else abs(got - value)
        ok = diff is not None and diff <= TOLERANCE_PX
        border[side] = {"measured": got, "formula": value, "abs_diff": diff, "passed": ok}
        if not ok:
            failures.append(f"border:{side}")
    for turret in turrets:
        x, y = turret["pixel_xy_exact"] if "pixel_xy_exact" in turret else map(float, minimap_xy(*turret["game_xy"]))
        distance, pixel = nearest_lane_pixel(mask, x, y)
        ok = bool(distance <= TOLERANCE_PX) if turret["checked"] else None
        rows.append({"team": turret["team"], "lane": turret["lane"], "tower": turret["tower"],
                     "game_xy": list(turret["game_xy"]), "pixel_xy": [float(x), float(y)], "checked": turret["checked"],
                     "nearest_lane_pixel": list(pixel) if pixel else None, "distance_px": distance, "passed": ok})
        if turret["checked"] and not ok:
            failures.append(f"turret:{turret['team']}:{turret['lane']}:{turret['tower']}")
    checked = [r["distance_px"] for r in rows if r["checked"]]
    return {"image_size": [int(image.shape[1]), int(image.shape[0])], "tolerance_px": TOLERANCE_PX,
            "pixel_convention": "integer (column, row) pixel indices, origin top left (PRESPEC_E5 line 39 values)",
            "lane_colour": list(colour), "lane_colour_block": {"centre": [CENTRE, CENTRE], "size": BLOCK,
                                                               "uniform": bool((block == block[0]).all())},
            "lane_rgb_distance_max": LANE_RGB_DISTANCE, "lane_pixels": int(mask.sum()), "formula_extent_px": formula,
            "border": border, "turrets": rows,
            "max_checked_turret_distance_px": max(checked) if checked else None,
            "max_border_abs_diff_px": max((b["abs_diff"] for b in border.values() if b["abs_diff"] is not None),
                                          default=None),
            "failures": failures, "passed": not failures}


# ---------------------------------------------------------------- time grid and the Q_DEV table

def time_grid(tau_ms: int, end_ms: int, lead_ms: int = LEAD_MS, step_ms: int = STEP_MS) -> list[int]:
    """tau - 60 s, tau - 55 s, ... up to the endpoint (times before 0 left out), then the endpoint if not on the grid."""

    tau, end = int(tau_ms), int(end_ms)
    if end <= tau:
        raise ValueError("the endpoint must lie after tau")
    times = [t for t in range(tau - int(lead_ms), end + 1, int(step_ms)) if t >= 0]
    if times[-1] != end:
        times.append(end)
    return times


def assemble_table(q: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], ka: Mapping[str, np.ndarray],
                   objective_counts: Any, preds: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """The Ruling 32 table on the Q_DEV q rows (Q_ROWS order) plus the row facts the figures need.

    ``q``: q row columns (every role); ``labels``: LABELS_h90 columns (any rows; only role Q_DEV is used);
    ``ka``: KA_LABELS columns in q row order (key order checked as r10_step1b.py:69-70, else Blocked);
    ``objective_counts``: OBJECTIVE_COUNTS indexed by query; ``preds``: {"P1", "PT_flex"} -> {"row", "p"}.
    A misaligned or inconsistent source raises ValueError.
    """

    keys = np.asarray(q["engagement_key"])
    if not np.array_equal(np.asarray(ka["engagement_key"]), keys):
        raise Blocked("table", {"ka_key_order": "KA_LABELS engagement_key differs from the q rows "
                                                "(r10_step1b.py:69-70)"})
    dev = np.flatnonzero(np.asarray(q["role"]) == Q_DEV)
    if not dev.size:
        raise ValueError("no Q_DEV q rows")
    p = {}
    for cell in ("P1", "PT_flex"):
        rows, values = np.asarray(preds[cell]["row"], dtype=np.int64), np.asarray(preds[cell]["p"], dtype=np.float64)
        if not np.array_equal(rows, dev):
            raise ValueError(f"{cell}: prediction rows are not the Q_DEV q rows in order")
        if not (np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()):
            raise ValueError(f"{cell}: predictions outside [0, 1]")
        p[cell] = values
    own = np.flatnonzero(np.asarray(labels["role"]) == Q_DEV)
    lab = {name: np.asarray(labels[name])[own] for name in LABEL_FIELDS}
    label_keys = np.char.add(np.char.add(lab["canonical_match_key"], b":"), lab["component_key"])
    if len(np.unique(label_keys)) != len(label_keys):
        raise ValueError("repeated Q_DEV label keys")
    if len(label_keys) != dev.size:
        raise ValueError(f"{len(label_keys)} Q_DEV label rows for {dev.size} Q_DEV q rows")
    order = np.argsort(label_keys, kind="stable")
    pos = np.clip(np.searchsorted(label_keys[order], keys[dev]), 0, len(order) - 1)
    if not np.array_equal(label_keys[order][pos], keys[dev]):
        raise ValueError("a Q_DEV q row has no Q_DEV label row")
    at = order[pos]
    lab = {name: values[at] for name, values in lab.items()}
    for name, q_name in (("query_pre", "query_pre"), ("pre_ms", "pre_ms"), ("match_index", "match_index"),
                         ("direction_positive", "y"), ("truncation_bits", "truncation_bits")):
        if not np.array_equal(lab[name].astype(np.int64), np.asarray(q[q_name])[dev].astype(np.int64)):
            raise ValueError(f"label {name} differs from the q row {q_name}")
    svi = lab["svi_delta"].astype(np.float64)
    if not np.array_equal(svi, np.asarray(q["label_svi_delta"], dtype=np.float64)[dev]):
        raise ValueError("label svi_delta differs from the q row label_svi_delta")
    y = lab["direction_positive"].astype(np.int64)
    if not (lab["valid"].all() and np.isin(y, (0, 1)).all() and np.array_equal(y == 1, svi > 0)
            and (lab["query_pre"] >= 0).all() and (lab["query_post"] >= 0).all()):
        raise ValueError("a Q_DEV label is invalid or y differs from 1[svi > 0]")
    blue = np.asarray(ka["blue_kills"])[dev].astype(np.int64)
    red = np.asarray(ka["red_kills"])[dev].astype(np.int64)
    pre_q, post_q = lab["query_pre"].astype(np.int64), lab["query_post"].astype(np.int64)
    objective = ~np.all(np.asarray(objective_counts[pre_q]) == np.asarray(objective_counts[post_q]), axis=1)
    table = {"engagement_key": keys[dev], "p_q": p["P1"], "p_pt": p["PT_flex"], "y": y, "svi": svi,
             "n_kills": np.asarray(q["eval_n_kills"])[dev].astype(np.int64), "ka_kills": blue + red,
             "kill_side_blue": (blue == 1).astype(np.int64),
             "same_frame": (lab["F_pre_ms"] == lab["F_post_ms"]).astype(np.int64), "objective": objective.astype(np.int64)}
    rows = {"q_row": dev, "blue_kills": blue, "red_kills": red,
            "weight_raw": np.asarray(q["weight_raw"], dtype=np.float64)[dev],
            **{name: lab[name] for name in ("canonical_match_key", "component_key", "match_index", "query_pre",
                                            "query_post", "pre_ms", "post_ms", "F_pre_ms", "F_post_ms", "first_kill_ms",
                                            "last_kill_ms", "p_pre", "p_post", "truncation_bits")}}
    match = rows["match_index"]
    checks = {"rows": int(dev.size), "matches": int(np.unique(match).size), "label_rows_q_dev": int(len(label_keys)),
              "ka_key_order_equal": True, "same_frame_rows": int(table["same_frame"].sum()),
              "objective_change_rows": int(table["objective"].sum()), "ka_single_kill_rows": int((table["ka_kills"] == 1).sum()),
              "ka_kills_equal_eval_n_kills_share": float(np.mean(table["ka_kills"] == table["n_kills"])),
              "aligned_by": "engagement_key = canonical_match_key:component_key; label query_pre, pre_ms, match_index, "
                            "y, truncation_bits and svi equal to the q row's"}
    return {"table": table, "rows": rows, "checks": checks}


def brier_check(p: np.ndarray, y: np.ndarray, w: np.ndarray, recorded: float, atol: float = BRIER_ATOL) -> dict[str, Any]:
    """The worker's raw match-weighted Brier (qtrain.match_brier) recomputed and compared with the recorded value."""

    value = match_brier(p, y, w)
    diff = abs(value - float(recorded))
    return {"recomputed": value, "recorded": float(recorded), "abs_diff": diff, "atol": atol, "passed": bool(diff <= atol)}


def closer_prediction(p_q: float, p_pt: float, y: float) -> str:
    """Which prediction lies closer to y (smaller |p - y|): "p_q", "p_pt" or "tie"."""

    dq, dp = abs(float(p_q) - float(y)), abs(float(p_pt) - float(y))
    return "p_q" if dq < dp else "p_pt" if dp < dq else "tie"


def gate_check(gate: Mapping[str, Any], current: Mapping[str, str], *, ignore: Iterable[str] = ()) -> dict[str, Any]:
    """The equality gate passed and every code file it recorded still has the recorded SHA-256 (``ignore`` aside)."""

    ignore = set(ignore)
    recorded = dict(gate.get("run", {}).get("code_sha256", {}))
    compared = sorted(name for name in recorded if name not in ignore)
    changed = sorted(name for name in compared if current.get(name) != recorded[name])
    ignored = sorted(name for name in recorded if name in ignore and current.get(name) != recorded[name])
    return {"gate_passed": bool(gate.get("passed")), "compared": compared, "changed": changed,
            "ignored_changed": ignored, "passed": bool(gate.get("passed")) and bool(compared) and not changed}


def kill_records(kills: Sequence[Mapping[str, Any]], members: Iterable[int], team_ids: Sequence[int],
                 champion_ids: Sequence[int], names: Mapping[int, str], window: tuple[int, int]
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(engagement kills, other kills in ``window``) from normalized CHAMPION_KILL records, in (time, ordinal) order.

    Killer and victim are given by team ("blue", "red"; "environment" for killerId 0) and champion name only; the
    participant slots are not kept.  Engagement kills are numbered 1, 2, ... in time order.
    """

    member_set = {int(o) for o in members}
    lo, hi = int(window[0]), int(window[1])

    def team(pid: Any) -> str | None:
        if pid is None:
            return None
        if int(pid) == 0:
            return "environment"
        return TEAM_NAMES.get(int(team_ids[int(pid) - 1])) if 1 <= int(pid) <= len(team_ids) else None

    def champion(pid: Any) -> str | None:
        if pid is None or not 1 <= int(pid) <= len(champion_ids):
            return None
        return names.get(int(champion_ids[int(pid) - 1]))

    ordered = sorted(kills, key=lambda k: (k["timestamp_ms"] is None, k["timestamp_ms"] or 0, int(k["source_ordinal"])))
    engagement, other = [], []
    for kill in ordered:
        ordinal, t = int(kill["source_ordinal"]), kill["timestamp_ms"]
        member = ordinal in member_set
        if not member and (t is None or not lo <= int(t) <= hi):
            continue
        record = {"order": None, "member": member, "source_ordinal": ordinal, "time_ms": None if t is None else int(t),
                  "killer_team": team(kill["killer_id"]), "killer_champion": champion(kill["killer_id"]),
                  "victim_team": team(kill["victim_id"]), "victim_champion": champion(kill["victim_id"]),
                  "x": None if kill["x"] is None else float(kill["x"]), "y": None if kill["y"] is None else float(kill["y"]),
                  "xy_valid": bool(kill["xy_valid"])}
        (engagement if member else other).append(record)
    if len(engagement) != len(member_set):
        raise ValueError("an engagement kill is missing from the normalized kills")
    for number, record in enumerate(engagement, start=1):
        record["order"] = number
    return engagement, other


def objective_records(history: Sequence[Mapping[str, Any]], window: tuple[int, int], *, tau: int) -> list[dict[str, Any]]:
    """Objective events (the kernel's objective history, ObservableMatch.query(...)["objectives"]["history"]) inside
    ``window``, duplicates left out, in time order: category, acquiring team and the recorded subtype.  Shown on the
    time axis next to the kills; they do not enter any selection or value."""

    lo, hi = int(window[0]), int(window[1])
    out = []
    for event in history:
        t = event.get("timestamp_ms")
        if t is None or event.get("duplicate_identity") or not lo <= int(t) <= hi:
            continue
        detail = event.get("monster_type") or " ".join(str(event[name]) for name in ("tower_type", "lane_type")
                                                         if event.get(name)) or event.get("building_type")
        out.append({"time_ms": int(t), "time_from_tau_s": (int(t) - int(tau)) / 1000.0,
                    "category": event.get("objective_category") or "unknown", "team": TEAM_NAMES.get(event.get("team_id")),
                    "event_type": event.get("event_type"), "detail": detail})
    return sorted(out, key=lambda record: record["time_ms"])


MAP_EXTENT = (-120.0, 14_870.0, -120.0, 14_980.0)    # game x and y at the formula's 25.34 / 490.16 px
ZOOM_HALF_CAP = 4000.0                                 # the zoom panel shows at most 8,000 x 8,000 map units


def zoom_window(points: Iterable[Sequence[float]], minimum_half: float = 1800.0, margin: float = 900.0
                ) -> tuple[float, float, float]:
    """(centre x, centre y, half side) in game units of the square zoom around ``points``, moved (not shrunk) to lie
    inside the map extent when it fits."""

    xy = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    cx, cy = (float(v) for v in xy.mean(axis=0))
    half = max(float(minimum_half), float(np.abs(xy - (cx, cy)).max()) + float(margin))
    x_lo, x_hi, y_lo, y_hi = MAP_EXTENT
    if 2 * half <= x_hi - x_lo:
        cx = min(max(cx, x_lo + half), x_hi - half)
    if 2 * half <= y_hi - y_lo:
        cy = min(max(cy, y_lo + half), y_hi - half)
    return cx, cy, half


def likely_dead(xy: np.ndarray, known: Sequence[bool], kills: Sequence[Mapping[str, Any]], frame_ms: int
                ) -> list[dict[str, Any]]:
    """Per participant slot: the time of its most recent death at or before the frame time, and whether its frame
    position equals that death's recorded position exactly (Ruling 34).  The data carry no alive flag, so a player
    standing on its own last death position is marked "likely dead": a position-equals-death-position heuristic, not
    a recorded status.  An unknown position is never marked."""

    xy = np.asarray(xy, dtype=np.float64)
    last: dict[int, Mapping[str, Any]] = {}
    deaths = (k for k in kills if k["timestamp_ms"] is not None and k["victim_id"] is not None)
    for kill in sorted(deaths, key=lambda k: (int(k["timestamp_ms"]), int(k["source_ordinal"]))):
        if int(kill["timestamp_ms"]) <= int(frame_ms):
            last[int(kill["victim_id"])] = kill
    out = []
    for slot in range(len(xy)):
        death = last.get(slot + 1)
        same = bool(known[slot] and death is not None and death["xy_valid"]
                    and float(death["x"]) == float(xy[slot, 0]) and float(death["y"]) == float(xy[slot, 1]))
        out.append({"last_death_ms": None if death is None else int(death["timestamp_ms"]), "likely_dead": same})
    return out


ENDPOINT_TEXT = {"horizon": "last kill + 90 s", "next_kill_minus_1": "next kill - 1 ms",
                 "next_anchor_minus_1": "next engagement anchor - 1 ms", "game_end_minus_1": "game end - 1 ms"}


def endpoint_reasons(bits: int) -> list[str]:
    """Why the h90 endpoint ended, from the label's truncation bits (engagements.TRUNCATION, bit i = 1 << i; the
    endpoint is the smallest of last kill + 90 s, next kill - 1, next anchor - 1 and game end - 1)."""

    bits = int(bits)
    if bits <= 0 or bits >= 1 << len(TRUNCATION):
        raise ValueError(f"truncation bits {bits} are not a valid h90 endpoint reason set")
    return [name for i, name in enumerate(TRUNCATION) if bits >> i & 1]


def endpoint_text(reasons: Sequence[str]) -> str:
    return " = ".join(ENDPOINT_TEXT[name] for name in reasons)


def selection_audit(table: Mapping[str, np.ndarray], selection: Mapping[str, Any]) -> dict[str, Any]:
    """Per rule of PRESPEC_E5 section 3, recomputed from the table: the candidates left after the earlier cases are
    dropped, the chosen and runner-up keys and scores in the rule's order (score descending, then key hash), the
    margin, and how many rows tie at the top score; rule (a) also per threshold; rule (c) also its population (the R10
    primary rows) and its violations before the exclusion.  Each chosen key must equal select_cases' choice."""

    keys = np.asarray(table["engagement_key"])
    texts = [key.decode("ascii") if isinstance(key, bytes) else str(key) for key in keys.tolist()]
    row_of = {text: i for i, text in enumerate(texts)}
    col = {name: np.asarray(table[name], dtype=np.float64) for name in e5_cases.NUMERIC}
    wrong, magnitude = np.abs(col["p_q"] - col["y"]), np.abs(col["svi"])
    chosen = {rule: None if selection[rule] is None else
              selection[rule].decode("ascii") if isinstance(selection[rule], bytes) else str(selection[rule]) for rule in RULES}

    def ranked(candidate: np.ndarray, score: np.ndarray, drop: Sequence[str]) -> dict[str, Any]:
        mask = np.asarray(candidate, dtype=bool).copy()
        for rule in drop:
            if chosen[rule] is not None:
                mask[row_of[chosen[rule]]] = False
        rows = np.flatnonzero(mask)
        order = sorted(rows.tolist(), key=lambda i: (-score[i], hashlib.sha256(texts[i].encode("utf-8")).hexdigest()))
        out = {"candidates": len(order), "chosen": None, "chosen_score": None, "runner_up": None,
               "runner_up_score": None, "margin": None, "tied_at_top": 0}
        if order:
            top = order[0]
            out.update(chosen=texts[top], chosen_score=float(score[top]), tied_at_top=int((score[rows] == score[top]).sum()))
            if len(order) > 1:
                second = order[1]
                out.update(runner_up=texts[second], runner_up_score=float(score[second]),
                           margin=float(score[top] - score[second]))
        return out

    enough_kills = col["n_kills"] >= e5_cases.A_MIN_KILLS
    threshold = selection["a_threshold"]
    audit: dict[str, Any] = {"a": {"threshold": threshold,
                                   "candidates_by_threshold": {str(t): int(((wrong >= t) & enough_kills).sum())
                                                               for t in e5_cases.A_THRESHOLDS},
                                   **ranked((wrong >= threshold) & enough_kills if threshold is not None
                                            else np.zeros(len(texts), dtype=bool), magnitude, ())}}
    audit["b"] = ranked(wrong < np.abs(col["p_pt"] - col["y"]), np.abs(col["p_q"] - col["p_pt"]), ("a",))
    sign = np.where(col["kill_side_blue"] == 1, 1.0, -1.0)
    population = (col["same_frame"] == 1) & (col["ka_kills"] == 1) & (col["objective"] == 0) & (col["svi"] != 0)
    violation = population & (sign * col["svi"] < 0)
    audit["c"] = {"population_rows": int(population.sum()), "violations_before_exclusion": int(violation.sum()),
                  **ranked(violation, magnitude, ("a", "b"))}
    for rule in RULES:
        if audit[rule]["chosen"] != chosen[rule]:
            raise RuntimeError(f"selection audit ({rule}) chose {audit[rule]['chosen']}, select_cases {chosen[rule]}")
    return audit


def time_axis_limits(v: Sequence[float]) -> tuple[float, float]:
    """The time axis's y limits: the data range (at least 0.02 wide) with a margin, inside [0, 1]."""

    values = np.asarray(v, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.02)
    return max(0.0, low - 0.16 * span), min(1.0, high + 0.10 * span)


LABEL_RINGS = (4.0, 9.0, 15.0, 22.0)                  # label offsets from the point, points
LABEL_DIRECTIONS = ((1, 0.6), (-1, 0.6), (1, -0.6), (-1, -0.6), (0, 1), (0, -1), (1, 0), (-1, 0))


def _overlap_area(a: Any, b: Any) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def boxes_overlap(a: Any, b: Any) -> bool:
    return _overlap_area(a, b) > 0.0


def place_labels(ax: Any, labels: Sequence[tuple], markers: Sequence[tuple[float, float, float]], *, fontsize: float,
                 color: str = "white", outline: bool = True, zorder: float = 7.0, weight: str = "normal",
                 avoid: Sequence[Any] = ()) -> list[tuple[Any, Any]]:
    """Name labels that avoid each other, the markers and the axes edge.  For each label (x, y, text[, colour]) in
    data coordinates the offsets of LABEL_RINGS x LABEL_DIRECTIONS are tried in order; the first whose text box (display
    coordinates, the figure's canvas renderer) overlaps no placed label, no box in ``avoid`` and no marker box
    (``markers``: x, y, size in points) and stays inside the axes is kept, otherwise the one with the least overlap.  A
    label beyond the first ring gets a thin leader line.  Returns (annotation, text box) per label.  The figure needs a
    canvas (FigureCanvasAgg)."""

    import matplotlib.patheffects as effects
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    ax.apply_aspect()
    axbox = ax.get_window_extent(renderer)
    scale = ax.figure.dpi / 72.0
    obstacles = []
    for x, y, size in markers:
        mx, my = ax.transData.transform((x, y))
        half = size * scale / 2.0
        obstacles.append(Bbox.from_extents(mx - half, my - half, mx + half, my + half))
    stroke = [effects.withStroke(linewidth=2.0, foreground="black")] if outline else None
    obstacles += list(avoid)
    placed: list[tuple[Any, Any]] = []
    for label in labels:
        x, y, text = label[:3]
        tint = label[3] if len(label) > 3 else color
        best = None
        for radius in LABEL_RINGS:
            for ux, uy in LABEL_DIRECTIONS:
                dx, dy = radius * ux, radius * uy
                artist = ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                                     ha="left" if dx > 0 else "right" if dx < 0 else "center",
                                     va="bottom" if dy > 0 else "top" if dy < 0 else "center", fontsize=fontsize,
                                     color=tint, fontweight=weight, path_effects=stroke, zorder=zorder,
                                     annotation_clip=False)
                box = artist.get_window_extent(renderer)
                inside = axbox.x0 <= box.x0 and box.x1 <= axbox.x1 and axbox.y0 <= box.y0 and box.y1 <= axbox.y1
                cost = sum(_overlap_area(box, other) for other in [*(b for _a, b in placed), *obstacles])
                cost += 0.0 if inside else 1e12
                if best is None or cost < best[2]:
                    if best is not None:
                        best[0].remove()
                    best = (artist, box, cost, dx, dy, radius)
                else:
                    artist.remove()
                if best[2] == 0.0:
                    break
            if best[2] == 0.0:
                break
        artist, box, _cost, dx, dy, radius = best
        if radius > LABEL_RINGS[0]:
            px, py = ax.transData.transform((x, y))
            (lx0, ly0), (lx1, ly1) = ax.transData.inverted().transform([(px, py), (px + dx * scale, py + dy * scale)])
            ax.add_line(Line2D([lx0, lx1], [ly0, ly1], color=tint, lw=0.6, alpha=0.85, zorder=zorder - 0.5))
        placed.append((artist, box))
    return placed


# ---------------------------------------------------------------- writing

def write_blocked(out: Path, stage: str, reasons: Mapping[str, Any], **extra: Any) -> Path:
    """E5_BLOCKED_<stage>_<n>.json once (n = the first free number, so earlier attempts stay)."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out / f"E5_BLOCKED_{stage}_{n}.json").exists():
        n += 1
    path = out / f"E5_BLOCKED_{stage}_{n}.json"
    write_once(path, _jsonable({"version": VERSION, "experiment": "E5", "blocked": True, "stage": stage, "utc": utc(),
                                "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": stage, "file": str(path), "reasons": sorted(reasons)}))
    return path


def write_failed(out: Path, stage: str, error: BaseException) -> Path:
    """E5_FAILED_<stage>_<n>.json once: an error after the map check (not a BLOCKED condition) is recorded in the
    folder that already holds MAP_CHECK.*, then raised again."""

    import traceback

    out = Path(out)
    n = 1
    while (out / f"E5_FAILED_{stage}_{n}.json").exists():
        n += 1
    path = out / f"E5_FAILED_{stage}_{n}.json"
    write_once(path, _jsonable({"version": VERSION, "experiment": "E5", "failed": True, "stage": stage, "utc": utc(),
                                "error": type(error).__name__, "message": str(error)[:4000],
                                "traceback": traceback.format_exception(error)[-12:]}))
    return path


def versions() -> dict[str, Any]:
    """Interpreter and library versions that decide the figures' bytes and the CPU V scores."""

    import matplotlib
    import PIL

    record = {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "pillow": PIL.__version__}
    torch = sys.modules.get("torch")
    record["torch"] = getattr(torch, "__version__", None) if torch is not None else None
    if record["torch"] is None:
        try:
            from importlib.metadata import version

            record["torch"] = version("torch")
        except Exception:  # noqa: BLE001  (an absent torch only drops the number)
            record["torch"] = None
    return record


def save_figure_once(figure: Any, path: Path, **kwargs: Any) -> str:
    """Save a matplotlib Figure once (pending file + rename; no creation date in a PDF) and return its SHA-256."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    fmt = path.suffix.lstrip(".").lower()
    metadata = {"CreationDate": None} if fmt == "pdf" else {"Software": None} if fmt == "png" else None
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    figure.savefig(pending, format=fmt, metadata=metadata, **kwargs)
    os.replace(pending, path)
    return sha256_file(path)


def _clock(ms: int) -> str:
    seconds = int(ms) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def draw_case(path: Path, case: Mapping[str, Any], image: np.ndarray) -> str:
    """case_<r>.pdf: V on the time axis with kills, objectives and frame times (left), F(tau) positions and kill
    positions on the whole map (middle) and around the engagement kills (right), the case facts below.  Players marked
    likely dead (Ruling 34) are hollow grey; labels are placed so that they do not overlap."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    tau, end = int(case["tau_ms"]), int(case["endpoint_ms"])
    axis = case["time_axis"]
    rel = (np.asarray(axis["times_ms"], dtype=np.float64) - tau) / 1000.0
    v = np.asarray(axis["v"], dtype=np.float64)
    figure = Figure(figsize=(14.5, 6.6))
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.050, 0.400, 0.430, 0.480))
    full = figure.add_axes((0.505, 0.400, 0.235, 0.480))
    zoom = figure.add_axes((0.755, 0.400, 0.235, 0.480))

    # time axis
    ax.axvspan(0.0, (end - tau) / 1000.0, color="#efefef", zorder=0)
    for frame in sorted({int(f) for f in axis["frame_ms"] if int(f) >= 0}):
        if rel[0] <= (frame - tau) / 1000.0 <= rel[-1]:
            ax.axvline((frame - tau) / 1000.0, color="#9a9a9a", ls=":", lw=1.0, zorder=1)
    for kill in case["other_kills_in_window"]:
        ax.axvline((kill["time_ms"] - tau) / 1000.0, color=COLOURS[kill["killer_team"]], ls=(0, (2, 2)), lw=0.9,
                   alpha=0.85, zorder=1)
    for x in (0.0, (end - tau) / 1000.0):
        ax.axvline(x, color="black", ls="--", lw=0.9, zorder=2)
    ax.plot(rel, v, "-o", color="#222222", ms=3.0, lw=1.3, zorder=4)
    ylim = time_axis_limits(v)
    ax.set_ylim(*ylim)
    height = ylim[1] - ylim[0]
    if ylim[0] < 0.5 < ylim[1]:
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=1)
    span_x = max(rel[-1] - rel[0], 1.0)
    last_x, level = None, 0
    for kill in case["kills"]:
        x = (kill["time_ms"] - tau) / 1000.0
        level = level + 1 if last_x is not None and x - last_x < 0.025 * span_x else 0
        last_x = x
        ax.axvline(x, color=COLOURS[kill["killer_team"]], lw=1.6, alpha=0.9, zorder=3)
        ax.annotate(str(kill["order"]), xy=(x, ylim[1]), xytext=(0, 2 + 9 * level), textcoords="offset points",
                    color=COLOURS[kill["killer_team"]], ha="center", va="bottom", fontsize=8, fontweight="bold",
                    annotation_clip=False)
    base = ylim[0] + 0.035 * height
    last_x, level = None, 0
    for event in sorted(case["objective_events_in_window"], key=lambda e: e["time_from_tau_s"]):
        x = event["time_from_tau_s"]
        ax.plot([x], [base], marker="^", ms=6.5, color=COLOURS[event["team"]], markeredgecolor="white", zorder=5)
        level = (level + 1) % 3 if last_x is not None and x - last_x < 0.03 * span_x else 0
        last_x = x
        ax.text(x, base + (0.05 + 0.16 * level) * height, event["category"], rotation=90, ha="center", va="bottom",
                fontsize=6.3, color=COLOURS[event["team"]], zorder=6,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.set_xlim(rel[0] - 1.5, rel[-1] + 1.5)
    reasons = case.get("endpoint_reasons") or []
    start_note = "" if rel[0] <= -LEAD_MS / 1000.0 else "; the grid starts at 0:00 (game start)"
    ax.set_xlabel(f"time from \u03c4 (s);  \u03c4 = {_clock(tau)} game time, first kill = \u03c4 + 10 s, h90 endpoint = "
                  f"\u03c4 + {(end - tau) / 1000.0:.1f} s" + (f" ({endpoint_text(reasons)})" if reasons else "") + start_note,
                  fontsize=8.5)
    ax.set_ylabel(f"calibrated final V (blue win probability)\ny axis span {height:.3f} (zoomed to the data)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    figure.suptitle(f"{case['title']}   (Q_DEV engagement {case['engagement_key'][:10]}\u2026:"
                    f"{case['engagement_key'].split(':')[-1][:10]}\u2026)", fontsize=10.5, y=0.975)

    # maps
    if case.get("zoom_window"):
        (cx, cy), half = case["zoom_window"]["centre"], case["zoom_window"]["half"]
    else:
        member_xy = [(k["x"], k["y"]) for k in case["kills"] if k["xy_valid"]]
        cx, cy, half = zoom_window(member_xy) if member_xy else (7375.0, 7430.0, 7495.0)
    x0, y1 = (float(c) for c in minimap_xy(cx - half, cy - half))
    x1, y0 = (float(c) for c in minimap_xy(cx + half, cy + half))

    def inside(px: float, py: float) -> bool:
        return x0 <= px <= x1 and y0 <= py <= y1

    players = []
    for player in case["positions_at_F_tau"]:
        if player["known"]:
            px, py = (float(c) for c in minimap_xy(player["x"], player["y"]))
            players.append((px, py, player))
    kill_points = [(*(float(c) for c in minimap_xy(k["x"], k["y"])), k) for k in case["kills"] if k["xy_valid"]]
    other_points = [(*(float(c) for c in minimap_xy(k["x"], k["y"])), k) for k in case["other_kills_in_window"]
                    if k["xy_valid"]]
    full.set_xlim(-0.5, MAP_SIDE - 0.5)
    full.set_ylim(MAP_SIDE - 0.5, -0.5)
    zoom.set_xlim(x0, x1)
    zoom.set_ylim(y1, y0)
    for panel, is_zoom in ((full, False), (zoom, True)):
        panel.imshow(image, interpolation="bilinear" if is_zoom else "antialiased")
        panel.set_xlim(*((x0, x1) if is_zoom else (-0.5, MAP_SIDE - 0.5)))
        panel.set_ylim(*((y1, y0) if is_zoom else (MAP_SIDE - 0.5, -0.5)))
        size = (70.0, 150.0, 40.0) if is_zoom else (40.0, 90.0, 26.0)     # player, engagement kill, other kill (points^2)
        markers = []
        for px, py, player in players:
            if player.get("likely_dead"):
                panel.scatter([px], [py], s=size[0], facecolors="none", edgecolors=DEAD_COLOUR, linewidths=1.4, zorder=4)
            else:
                panel.scatter([px], [py], s=size[0], color=COLOURS[player["team"]], edgecolors="white", linewidths=0.9,
                              zorder=4)
            markers.append((px, py, math.sqrt(size[0])))
        for px, py, kill in other_points:
            panel.scatter([px], [py], s=size[2], marker="x", color=COLOURS[kill["killer_team"]], linewidths=1.1,
                          alpha=0.85, zorder=5)
            markers.append((px, py, math.sqrt(size[2])))
        for px, py, kill in kill_points:
            panel.scatter([px], [py], s=size[1], marker="X", color=COLOURS[kill["killer_team"]], edgecolors="white",
                          linewidths=0.9, zorder=6)
            markers.append((px, py, math.sqrt(size[1])))
        if is_zoom:
            numbers = place_labels(panel, [(px, py, str(kill["order"])) for px, py, kill in kill_points], markers,
                                   fontsize=9, color="yellow", weight="bold", zorder=7.5)
            names = [(px, py, player["champion"], DEAD_COLOUR if player.get("likely_dead") else "white")
                     for px, py, player in players if player["champion"] and inside(px, py)]
            place_labels(panel, names, markers, fontsize=7.0, avoid=[box for _artist, box in numbers])
        else:
            names = [(px, py, player["champion"], DEAD_COLOUR if player.get("likely_dead") else "white")
                     for px, py, player in players if player["champion"] and not inside(px, py)]
            place_labels(panel, names, markers, fontsize=6.0)
        panel.set_axis_off()
    full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1.0, ls="--", zorder=8))
    full.set_title(f"F(\u03c4) positions (frame {_clock(case['F_pre_ms'])}, \u03c4 \u2212 "
                   f"{(tau - case['F_pre_ms']) / 1000.0:.1f} s)\nand kill positions", fontsize=8.5)
    zoom.set_title(f"zoom (dashed box, {2 * half:,.0f} \u00d7 {2 * half:,.0f} map units):\n"
                   f"around the engagement kills", fontsize=8.5)

    figure.text(0.050, 0.300, "\n".join(case["summary_lines"]), ha="left", va="top", fontsize=8.0, linespacing=1.4)
    handles = [Line2D([], [], color="#222222", marker="o", ms=3, lw=1.3,
                      label="V every 5 s from \u03c4 \u2212 60 s, and at the endpoint"),
               Line2D([], [], color="black", ls="--", lw=0.9, label="\u03c4 and the h90 endpoint (label window shaded)"),
               Line2D([], [], color="#9a9a9a", ls=":", lw=1.0, label="frame times (positions and gold update)"),
               Line2D([], [], color=COLOURS["blue"], lw=1.6, label="engagement kill by blue (numbered)"),
               Line2D([], [], color=COLOURS["red"], lw=1.6, label="engagement kill by red (numbered)"),
               Line2D([], [], color="#555555", ls=(0, (2, 2)), lw=0.9, label="other kill (colour = killer's team)"),
               Line2D([], [], color="#555555", marker="^", ls="", ms=6, label="objective taken (colour = team)"),
               Line2D([], [], color=COLOURS["blue"], marker="o", ls="", ms=6, markeredgecolor="white",
                      label="blue player at F(\u03c4)"),
               Line2D([], [], color=COLOURS["red"], marker="o", ls="", ms=6, markeredgecolor="white",
                      label="red player at F(\u03c4)"),
               Line2D([], [], color=DEAD_COLOUR, marker="o", ls="", ms=6, markerfacecolor="none", markeredgewidth=1.4,
                      label="likely dead at F(\u03c4) (on own last death position)"),
               Line2D([], [], color="#555555", marker="X", ls="", ms=8, markeredgecolor="white",
                      label="engagement kill position (x: other kill)")]
    figure.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.4, frameon=False, bbox_to_anchor=(0.505, 0.315))
    return save_figure_once(figure, path)


def draw_map_check(path: Path, image: np.ndarray, check: Mapping[str, Any]) -> str:
    """MAP_CHECK.png: the formula extent, the measured border and all twelve turrets with their nearest lane pixel."""

    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    figure = Figure(figsize=(7.6, 8.0))
    ax = figure.add_axes((0.03, 0.12, 0.94, 0.80))
    ax.imshow(image)
    extent = check["formula_extent_px"]
    ax.add_patch(Rectangle((extent["left"], extent["top"]), extent["right"] - extent["left"],
                           extent["bottom"] - extent["top"], fill=False, edgecolor="yellow", ls="--", lw=1.0))
    border = check["border"]
    ax.plot([border["left"]["measured"], border["right"]["measured"], CENTRE, CENTRE],
            [CENTRE, CENTRE, border["top"]["measured"], border["bottom"]["measured"]], "s", color="orange", ms=5)
    for turret in check["turrets"]:
        x, y = turret["pixel_xy"]
        colour = COLOURS[TEAM_NAMES[turret["team"]]]
        ax.add_patch(Circle((x, y), TOLERANCE_PX, fill=False, edgecolor=colour, lw=0.8, ls="-" if turret["checked"] else ":"))
        ax.scatter([x], [y], s=34, marker="o", facecolors=colour if turret["checked"] else "none", edgecolors=colour,
                   linewidths=1.3, zorder=4)
        if turret["nearest_lane_pixel"]:
            nx, ny = turret["nearest_lane_pixel"]
            ax.plot([x, nx], [y, ny], color="white", lw=0.8, zorder=3)
            ax.scatter([nx], [ny], s=14, marker="+", color="white", zorder=5)
        ax.text(x + 8, y - 8, f"{turret['distance_px']:.2f}", color="white", fontsize=6.5, zorder=6)
    ax.set_xlim(-0.5, MAP_SIDE - 0.5)
    ax.set_ylim(MAP_SIDE - 0.5, -0.5)
    ax.set_axis_off()
    verdict = "PASSED" if check["passed"] else "FAILED: " + ", ".join(check["failures"])
    ax.set_title(f"E5 map check (Rulings 26, 26b): border max |diff| {check['max_border_abs_diff_px']:.2f} px, "
                 f"8 checked turrets max {check['max_checked_turret_distance_px']:.2f} px; tolerance "
                 f"{TOLERANCE_PX:.2f} px. {verdict}", fontsize=8.5)
    handles = [Line2D([], [], color="yellow", ls="--", label="formula extent 25.34-490.16 px"),
               Line2D([], [], color="orange", marker="s", ls="", label="measured border (row/column 256)"),
               Line2D([], [], color="#777777", marker="o", ls="", label="checked turret (filled; 10.24 px circle)"),
               Line2D([], [], color="#777777", marker="o", ls="", markerfacecolor="none",
                      label="top/bottom inner turret, not judged (Ruling 26b)"),
               Line2D([], [], color="black", marker="+", ls="", label="nearest lane-colour pixel; number = distance px")]
    figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    return save_figure_once(figure, path, dpi=150)


# ---------------------------------------------------------------- the run

def _stage(started: str, tick: float, **extra: Any) -> dict[str, Any]:
    return {"started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "peak_memory_bytes_main_process": peak_memory_bytes(), **extra}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _load(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in fields}


def load_map_image(path: Path = MAP_IMAGE) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def load_sources() -> dict[str, Any]:
    """The PRESPEC_E5 section 2 arrays.  LABELS_h90 is cut to its Q_DEV rows field by field as it is read."""

    q = _load(Q_ROWS, Q_FIELDS)
    with np.load(LABEL_TABLE, allow_pickle=False) as data:
        own = np.flatnonzero(data["role"] == Q_DEV)
        labels = {name: data[name][own] for name in LABEL_FIELDS}
    ka = _load(KA_LABELS, ("engagement_key", "blue_kills", "red_kills"))
    objective = np.load(OBJECTIVE_COUNTS, mmap_mode="r", allow_pickle=False)
    preds = {cell: _load(Q_RUN / "jobs" / job / "pred_q_dev.npz", ("row", "p")) for cell, job in JOBS.items()}
    results = {cell: read_json(Q_RUN / "jobs" / job / "RESULT.json") for cell, job in JOBS.items()}
    return {"q": q, "labels": labels, "ka": ka, "objective": objective, "preds": preds, "results": results}


def keys_sha256(keys: np.ndarray) -> str:
    """q_worker.keys_sha256 (q_worker.py:73-74), restated so the driver does not import the worker."""

    import hashlib

    return hashlib.sha256(b"\n".join(bytes(k) for k in np.asarray(keys))).hexdigest()


def table_checks(sources: Mapping[str, Any], built: Mapping[str, Any]) -> dict[str, Any]:
    """Row and match counts against Q_ROWS_h90.json and PRESPEC_E5, and each prediction file against its RESULT.json."""

    q, dev = sources["q"], built["rows"]["q_row"]
    counts = read_json(Q_RUN / "Q_ROWS_h90.json")["counts"]["Q_DEV"]
    out: dict[str, Any] = {"rows": int(dev.size), "matches": built["checks"]["matches"],
                           "q_rows_json": {"rows": counts["rows"], "matches": counts["matches"]},
                           "expected": {"rows": EXPECTED_ROWS, "matches": EXPECTED_MATCHES}, "cells": {}}
    ok = out["rows"] == counts["rows"] == EXPECTED_ROWS and out["matches"] == counts["matches"] == EXPECTED_MATCHES
    keys_digest = keys_sha256(np.asarray(q["engagement_key"])[dev])
    for cell, job in JOBS.items():
        result = sources["results"][cell]["scores"]["q_dev"]
        pred = sources["preds"][cell]
        rows = np.asarray(pred["row"], dtype=np.int64)
        brier = brier_check(pred["p"], np.asarray(q["y"], dtype=np.float64)[rows],
                            np.asarray(q["weight_raw"], dtype=np.float64)[rows], result["raw_match_brier"])
        entry = {"job": job, "pred_sha256": sha256_file(Q_RUN / "jobs" / job / "pred_q_dev.npz"),
                 "recorded_pred_sha256": result["pred_sha256"], "rows": int(rows.size), "recorded_rows": result["rows"],
                 "q_dev_keys_sha256": keys_digest, "recorded_q_dev_keys_sha256": sources["results"][cell]["q_dev_keys_sha256"],
                 "status": sources["results"][cell]["status"], "brier": brier}
        entry["passed"] = bool(brier["passed"] and entry["pred_sha256"] == entry["recorded_pred_sha256"]
                               and entry["rows"] == entry["recorded_rows"] == EXPECTED_ROWS
                               and keys_digest == entry["recorded_q_dev_keys_sha256"] and entry["status"] == "completed")
        ok = ok and entry["passed"]
        out["cells"][cell] = entry
    out["passed"] = bool(ok)
    return out


def champion_names(patch: str = PATCH) -> tuple[dict[int, str], dict[str, Any]]:
    """Data Dragon champion id -> display name for the patch, the file checked against its recorded SHA-256."""

    from studies.cache_source_audit_20261002.config import DATADRAGON_EXPECTED_SHA256, DATADRAGON_PATHS

    path = DATADRAGON_PATHS[patch]
    digest = sha256_file(path)
    if digest != DATADRAGON_EXPECTED_SHA256[patch]:
        raise ValueError(f"{path} differs from its recorded SHA-256")
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    return {int(record["key"]): str(record["name"]) for record in data.values()}, {"path": str(path), "sha256": digest}


class CaseContext:
    """What every case needs once: the projection, the bank shards read so far, the final V and its bundle."""

    def __init__(self) -> None:
        from studies.label_production_20261005 import controls_states as cs
        from studies.label_production_20261005 import queries, score
        from studies.v_full_preprocess_20261003.artifacts import load_bundle

        self.cs, self.queries = cs, queries
        self.projection = cs.Projection()
        if self.projection.root.resolve() != Path(queries.PROJECTION).resolve():
            raise ValueError("the bank projection differs from the label projection")
        self.projection_sha256 = self.projection.file_hashes()
        self.v = score.LockedV(RUN_3, "final", device="cpu")
        self.bundle_dir = Path(queries.PREPROCESS) / "final_REFIT_TRAIN"
        self.bundle = load_bundle(self.bundle_dir)
        self.names, self.names_record = champion_names()
        self._banks: dict[int, Any] = {}
        self.bank_dirs: list[Path] = []

    def source(self, match: int, canonical: str) -> Any:
        projection = self.projection
        if projection.key(match) != canonical:
            raise ValueError(f"match {match}: projection key differs from the label key")
        if int(projection.matches["role"][match]) != Q_DEV:
            raise ValueError(f"match {match} is not a Q_DEV match")
        shard = int(projection.matches["shard"][match])
        if shard not in self._banks:
            self._banks[shard] = self.cs.BankShard(shard, projection.binding(shard), projection.source_root)
            self.bank_dirs.append(self._banks[shard].directory)
        return self._banks[shard].source(int(projection.matches["local_match"][match]), match, canonical)


def build_case(rule: str, row: int, built: Mapping[str, Any], selection: Mapping[str, Any], ctx: CaseContext) -> dict[str, Any]:
    """States, V, kills and F(tau) positions of one chosen engagement, as facts (module docstring, stage cases)."""

    from studies.label_production_20261005.controls_states import BANK_ADDED_PREFIXES
    from studies.observable_state_20261002 import population
    from studies.observable_state_20261002.bank import unpack_bank
    from studies.observable_state_20261002.kernel import ObservableMatch
    from studies.tog_experiments_20261012.e1_gate import xy_at_times

    t, r = built["table"], built["rows"]
    value = {name: t[name][row] for name in t}
    fact = {name: r[name][row] for name in r}
    key = value["engagement_key"].decode("ascii")
    canonical, component = fact["canonical_match_key"].decode("ascii"), fact["component_key"].decode("ascii")
    match = int(fact["match_index"])
    source = ctx.source(match, canonical)
    if str(source.entry["patch"]) != PATCH:
        raise ValueError(f"{key}: patch {source.entry['patch']} is not {PATCH}")
    tau, end = int(fact["pre_ms"]), int(fact["post_ms"])
    times = time_grid(tau, end)
    build = ctx.cs.build_match(source, times)
    if not np.array_equal(np.asarray(build.query["query_ms"], dtype=np.int64), np.asarray(times, dtype=np.int64)):
        raise ValueError(f"{key}: built state times differ from the grid")
    semantic = ctx.cs.semantic_rows(build.states)
    logit, raw, v = ctx.v.score(ctx.queries.transform(semantic, ctx.bundle))
    i_tau, i_end = times.index(tau), len(times) - 1
    repro = {"pre": abs(float(v[i_tau]) - float(fact["p_pre"])), "post": abs(float(v[i_end]) - float(fact["p_post"]))}
    if max(repro.values()) > V_REPRO_ATOL:
        raise RuntimeError(f"{key}: rescored V differs from the labels by {repro}")
    clock = np.asarray(source.view["minute_ts"], dtype=np.int64)
    frame_index = np.asarray(build.query["query_frame_index"], dtype=np.int64)
    frame_ms = np.where(frame_index >= 0, clock[np.maximum(frame_index, 0)], -1)
    if int(frame_ms[i_tau]) != int(fact["F_pre_ms"]) or int(frame_ms[i_end]) != int(fact["F_post_ms"]):
        raise ValueError(f"{key}: held frames differ from the labels' F_pre_ms / F_post_ms")
    xy, support = xy_at_times(source.view, [tau], [int(frame_index[i_tau])])

    model_bank = {name: array for name, array in source.view.items() if not name.startswith(BANK_ADDED_PREFIXES)}
    reopened = ObservableMatch.from_bank(unpack_bank(model_bank), str(source.entry["patch"]))
    components = [c for c in population.build_population(reopened) if str(c["component_key"]) == component]
    if len(components) != 1:
        raise ValueError(f"{key}: {len(components)} population components carry the key")
    comp = components[0]
    if (int(comp["f"]), int(comp["L"]), int(comp["pre"]), int(comp["F_pre_ms"]), int(comp["n_members"])) != (
            int(fact["first_kill_ms"]), int(fact["last_kill_ms"]), tau, int(fact["F_pre_ms"]), int(value["n_kills"])):
        raise ValueError(f"{key}: the population component differs from the label row or eval_n_kills")
    reasons = endpoint_reasons(int(fact["truncation_bits"]))
    if int(comp["e_h90"]) != end or sorted(comp["truncation_reasons_h90"]) != sorted(reasons):
        raise ValueError(f"{key}: the population h90 endpoint or its reasons differ from the label")
    team_ids = np.asarray(reopened.team_ids, dtype=np.int64)
    champion_ids = np.asarray(reopened.static_ids, dtype=np.int64)[:, 0]
    normalized = population.normalize_champion_kills(reopened)
    kills, others = kill_records(normalized, comp["member_source_ordinals"], team_ids, champion_ids, ctx.names,
                                 (times[0], end))
    members = {int(o) for o in comp["member_source_ordinals"]}
    victims = sorted({int(k["victim_id"]) for k in normalized if int(k["source_ordinal"]) in members
                      and k["victim_id"] is not None and 1 <= int(k["victim_id"]) <= len(team_ids)})
    frame_xy = np.asarray(xy[0], dtype=np.float64)
    known = [bool(support[0]) and math.isfinite(px) and math.isfinite(py) and px != 0.0 and py != 0.0
             for px, py in frame_xy.tolist()]
    dead = likely_dead(frame_xy, known, normalized, int(fact["F_pre_ms"]))
    positions = []
    for slot in range(len(team_ids)):
        px, py = (float(c) for c in frame_xy[slot])
        positions.append({"team": TEAM_NAMES.get(int(team_ids[slot])), "champion": ctx.names.get(int(champion_ids[slot])),
                          "x": px if math.isfinite(px) else None, "y": py if math.isfinite(py) else None,
                          "known": known[slot], "likely_dead": dead[slot]["likely_dead"],
                          "last_death_ms_at_or_before_frame": dead[slot]["last_death_ms"],
                          "minimap_px": [float(c) for c in minimap_xy(px, py)] if known[slot] else None})
    dead_names = [p["champion"] or p["team"] for p in positions if p["likely_dead"]]
    for kill in kills + others:
        kill["time_from_tau_s"] = None if kill["time_ms"] is None else (kill["time_ms"] - tau) / 1000.0
        kill["minimap_px"] = [float(c) for c in minimap_xy(kill["x"], kill["y"])] if kill["xy_valid"] else None
    by_team = {name: sum(k["killer_team"] == name for k in kills) for name in ("blue", "red", "environment")}
    objectives = objective_records(reopened.query(end)["objectives"]["history"], (times[0], end), tau=tau)
    kill_points = [(k["x"], k["y"]) for k in kills if k["xy_valid"]]
    victim_points = [(positions[pid - 1]["x"], positions[pid - 1]["y"]) for pid in victims
                     if positions[pid - 1]["known"] and not positions[pid - 1]["likely_dead"]]
    basis = "engagement kill positions and their victims' F(tau) positions (victims marked likely dead left out)"
    cx, cy, half = zoom_window(kill_points + victim_points) if kill_points else (7375.0, 7430.0, 7495.0)
    if kill_points and half > ZOOM_HALF_CAP:
        cx, cy, half = zoom_window(kill_points)
        basis = f"engagement kill positions only (with the victims' F(tau) positions the half side exceeds {ZOOM_HALF_CAP:,.0f})"
    zoom = {"centre": [cx, cy], "half": half, "basis": basis if kill_points else "whole map (no kill position recorded)",
            "rule": "square: half side = largest offset from the centre + 900, at least 1,800 map units, moved inside the "
                    "map; presentation only"}

    ylim = time_axis_limits(v)
    p_q, p_pt, y, svi = float(value["p_q"]), float(value["p_pt"]), int(value["y"]), float(value["svi"])
    closer = closer_prediction(p_q, p_pt, y)
    title = {"a": f"(a) Confidently wrong: |p_q \u2212 y| = {abs(p_q - y):.3f} \u2265 {selection['a_threshold']}, "
                  f"{int(value['n_kills'])} kills, largest |svi|",
             "b": f"(b) Apart from the state baseline: |p_q \u2212 p_pt| = {abs(p_q - p_pt):.3f}, p_q closer to y",
             "c": f"(c) Monotonicity violation: one {'blue' if value['kill_side_blue'] else 'red'} kill, "
                  f"V moved toward {'blue' if svi > 0 else 'red'}"}[rule]
    lines = [f"p_q (P1 LightGBM) = {p_q:.3f},  p_pt (PT_flex) = {p_pt:.3f},  y = {y}",
             f"|p_q \u2212 y| = {abs(p_q - y):.3f},  |p_pt \u2212 y| = {abs(p_pt - y):.3f}  (closer: {closer})",
             f"svi = V(end) \u2212 V(\u03c4) = {float(fact['p_post']):.3f} \u2212 {float(fact['p_pre']):.3f} = {svi:+.3f} "
             f"(toward {'blue' if svi > 0 else 'red' if svi < 0 else 'neither'})",
             f"same frame (F_pre = F_post): {'yes' if value['same_frame'] else 'no'};  objective counts changed in "
             f"[\u03c4, end]: {'yes' if value['objective'] else 'no'}",
             f"h90 endpoint: \u03c4 + {(end - tau) / 1000.0:.1f} s ({endpoint_text(reasons)})",
             f"time axis: y from {ylim[0]:.3f} to {ylim[1]:.3f} (span {ylim[1] - ylim[0]:.3f}, zoomed to the data)",
             f"engagement kills: {int(value['n_kills'])} (by blue {by_team['blue']}, red {by_team['red']}, "
             f"environment {by_team['environment']})"]
    shown = kills[:6]
    lines += [f"  kill {k['order']}: {k['killer_champion'] or k['killer_team']} ({k['killer_team']}) \u2192 "
              f"{k['victim_champion']} ({k['victim_team']}) at \u03c4 + {k['time_from_tau_s']:.1f} s" for k in shown]
    if len(kills) > len(shown):
        lines.append(f"  ... {len(kills) - len(shown)} more kills (E5_CASES.json)")
    if dead_names:
        lines += [f"likely dead at F(\u03c4), drawn hollow grey: {', '.join(dead_names)}",
                  "  (position = own last death position; the data carry no alive flag)"]
    case = {"rule": rule, "rule_name": RULE_NAMES[rule], "rule_text": RULE_TEXT[rule], "title": title,
            "threshold": selection["a_threshold"] if rule == "a" else None,
            "engagement_key": key, "canonical_match_key": canonical, "component_key": component, "match_index": match,
            "q_row": int(fact["q_row"]), "split": "Q_DEV", "patch": PATCH,
            "p_q": p_q, "p_pt": p_pt, "y": y, "svi": svi, "abs_p_q_minus_y": abs(p_q - y),
            "abs_p_pt_minus_y": abs(p_pt - y), "abs_p_q_minus_p_pt": abs(p_q - p_pt), "closer_prediction": closer,
            "n_kills_eval": int(value["n_kills"]), "ka_blue_kills": int(fact["blue_kills"]),
            "ka_red_kills": int(fact["red_kills"]), "ka_kills": int(value["ka_kills"]),
            "kill_side_blue": int(value["kill_side_blue"]) if int(value["ka_kills"]) == 1 else None,
            "kill_side_blue_note": "1 when the single team-credited kill (KA) was blue's, 0 when red's (rule c); null "
                                   "when the KA kill count is not 1 (see ka_blue_kills and ka_red_kills)",
            "same_frame": int(value["same_frame"]),
            "objective_change": int(value["objective"]), "engagement_kills_by_killer_team": by_team,
            "tau_ms": tau, "game_time_tau": _clock(tau), "first_kill_ms": int(fact["first_kill_ms"]),
            "last_kill_ms": int(fact["last_kill_ms"]), "endpoint_ms": end, "truncation_bits": int(fact["truncation_bits"]),
            "endpoint_reasons": reasons, "endpoint_text": endpoint_text(reasons), "F_pre_ms": int(fact["F_pre_ms"]),
            "F_post_ms": int(fact["F_post_ms"]), "query_pre": int(fact["query_pre"]), "query_post": int(fact["query_post"]),
            "v_label_pre": float(fact["p_pre"]), "v_label_post": float(fact["p_post"]),
            "v_rescored_pre": float(v[i_tau]), "v_rescored_post": float(v[i_end]), "v_rescore_abs_diff": repro,
            "kills": kills, "other_kills_in_window": others, "objective_events_in_window": objectives, "zoom_window": zoom,
            "time_axis": {"times_ms": times, "time_from_tau_s": [(x - tau) / 1000.0 for x in times],
                          "v": [float(x) for x in v], "raw_p": [float(x) for x in raw], "logit": [float(x) for x in logit],
                          "frame_index": frame_index.tolist(), "frame_ms": frame_ms.tolist(), "y_limits": list(ylim),
                          "y_axis_span": ylim[1] - ylim[0], "v_range": [float(v.min()), float(v.max())]},
            "positions_at_F_tau": positions, "unknown_positions": sum(not p["known"] for p in positions),
            "likely_dead_rule": LIKELY_DEAD_RULE, "likely_dead_players": dead_names, "likely_dead_count": len(dead_names),
            "summary_lines": lines, "figure": f"case_{rule}.pdf"}
    return case


def run(out: Path = TOG_E5) -> int:
    """The whole E5 figure run (module docstring).  0 when done, 2 when BLOCKED."""

    started, tick = utc(), time.perf_counter()
    out = Path(out)
    names = [MAP_JSON, MAP_PNG, CASES_JSON, *(f"case_{rule}.pdf" for rule in RULES)]
    existing = [name for name in names if (out / name).exists()]
    if existing:
        raise FileExistsError(f"{out}: {existing} exist; the E5 outputs are written once")
    out.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}

    s, t0 = utc(), time.perf_counter()
    image = load_map_image()
    try:
        turrets = lane_turrets(json.loads(MAP_ANCHORS.read_text(encoding="utf-8")))
        check = map_check(image, turrets)
    except Blocked as blocked:
        write_blocked(out, blocked.stage, blocked.reasons)
        return 2
    map_inputs = [MAP_IMAGE, MAP_ANCHORS]
    map_payload = {"version": VERSION, "experiment": "E5", "rulings": ["26", "26b"],
                   "rule": "PRESPEC_E5 section 4: (i) border within 10.24 px of the formula extent; (ii) the eight lane "
                           "turrets on the drawn lanes (six outer, two mid inner) within 10.24 px of a lane-colour pixel; "
                           "the four top/bottom inner turrets are measured but not judged (Ruling 26b)",
                   "formula": "X = (29 + (g_x + 120) * 532 / 14990) * 512 / 586, Y = (561 - (g_y + 120) * 532 / 15100) "
                              "* 512 / 586",
                   "image": str(MAP_IMAGE), "anchors": str(MAP_ANCHORS), **check, "versions": versions(),
                   "run": run_record(rows=None, matches=None, inputs=map_inputs, code=[ROOT / name for name in CODE_FILES[:5]],
                                     started_utc=s)}
    write_once(out / MAP_JSON, _jsonable(map_payload))
    png_sha = draw_map_check(out / MAP_PNG, image, check)
    stages["map"] = _stage(s, t0, passed=check["passed"])
    print(json.dumps({"map_check": check["passed"], "border_max_px": check["max_border_abs_diff_px"],
                      "turret_max_px": check["max_checked_turret_distance_px"]}))
    if not check["passed"]:
        write_blocked(out, "map", {name: True for name in check["failures"]}, map_check=str(out / MAP_JSON))
        return 2

    stage = {"name": "gate"}
    try:
        return _after_map(out, image, check, png_sha, stages, started, tick, stage)
    except Exception as error:  # noqa: BLE001  (recorded in the output folder, then raised again)
        write_failed(out, stage["name"], error)
        raise


def _after_map(out: Path, image: np.ndarray, check: Mapping[str, Any], png_sha: str, stages: dict[str, Any], started: str,
               tick: float, stage: dict[str, str]) -> int:
    """The stages after a passed map check (module docstring); ``stage["name"]`` follows the running stage."""

    s, t0 = utc(), time.perf_counter()
    gate = read_json(GATE) or {}
    gate_result = gate_check(gate, {name: sha256_file(ROOT / name) if (ROOT / name).is_file() else None
                                    for name in gate.get("run", {}).get("code_sha256", {})}, ignore=GATE_IGNORE)
    gate_result.update(file=str(GATE), sha256=sha256_file(GATE), ignored_reason=GATE_IGNORE)
    stages["gate"] = _stage(s, t0, passed=gate_result["passed"])
    if not gate_result["passed"]:
        write_blocked(out, "gate", {"equality_gate": gate_result})
        return 2

    stage["name"] = "table"
    s, t0 = utc(), time.perf_counter()
    sources = load_sources()
    try:
        built = assemble_table(sources["q"], sources["labels"], sources["ka"], sources["objective"], sources["preds"])
    except Blocked as blocked:
        write_blocked(out, blocked.stage, blocked.reasons)
        return 2
    checks = table_checks(sources, built)
    if not checks["passed"]:
        raise RuntimeError(f"table checks failed: {json.dumps(checks, default=_json_default)[:2000]}")
    del sources
    stages["table"] = _stage(s, t0, rows=checks["rows"], matches=checks["matches"])

    stage["name"] = "select"
    s, t0 = utc(), time.perf_counter()
    selection = e5_cases.select_cases(built["table"])
    audit = selection_audit(built["table"], selection)
    keys = built["table"]["engagement_key"]
    chosen = {}
    for rule in RULES:
        if selection[rule] is not None:
            hits = np.flatnonzero(keys == selection[rule])
            if hits.size != 1:
                raise RuntimeError(f"case {rule}: {hits.size} table rows carry the chosen key")
            chosen[rule] = int(hits[0])
    stages["select"] = _stage(s, t0)
    print(json.dumps({rule: (None if selection[rule] is None else selection[rule].decode("ascii")) for rule in RULES}
                     | {"a_threshold": selection["a_threshold"]}))

    stage["name"] = "cases"
    s, t0 = utc(), time.perf_counter()
    ctx = CaseContext()
    cases, figures = {}, {}
    for rule in RULES:
        if rule not in chosen:
            cases[rule] = None
            continue
        case = build_case(rule, chosen[rule], built, selection, ctx)
        cases[rule] = case
    _jsonable(cases)                       # the facts must serialize (no NaN) before any figure is written
    stages["states_and_scores"] = _stage(s, t0)
    stage["name"] = "figures"
    s, t0 = utc(), time.perf_counter()
    for rule, case in cases.items():
        if case is not None:
            figures[rule] = {"file": str(out / case["figure"]), "sha256": draw_case(out / case["figure"], case, image)}
    stages["figures"] = _stage(s, t0)

    inputs = [Q_ROWS, Q_RUN / "Q_ROWS_h90.json", LABEL_TABLE, KA_LABELS, OBJECTIVE_COUNTS, GATE, MAP_IMAGE, MAP_ANCHORS,
              Path(ctx.names_record["path"])]
    for job in JOBS.values():
        inputs += [Q_RUN / "jobs" / job / "pred_q_dev.npz", Q_RUN / "jobs" / job / "RESULT.json"]
    inputs += [RUN_3 / "final" / name for name in ("V_LOCK.json", "CONTEXT_PLAN.json", "TRAINING_ACCEPTANCE.json")]
    inputs += [Path(ctx.v.identity["checkpoint"])]
    inputs += sorted(path for path in ctx.bundle_dir.iterdir() if path.is_file())
    inputs += [ctx.projection.root / name for name in ("RUN_MANIFEST.json", "BANK_METADATA.json", "MATCH_TABLE.npy",
                                                       "QUERY_TABLE.npy", "MATCH_QUERIES.npy")]
    for directory in ctx.bank_dirs:
        inputs += [directory / name for name in ("BANK.npz", "LAYOUT.json", "MATCH_LEDGER.jsonl", "SHARD_MANIFEST.json",
                                                 "STATIC_REPAIR.json")]
    code = [ROOT / name for name in CODE_FILES] + [ROOT / name for name in ctx.cs.CODE_FILES
                                                   if name not in CODE_FILES]
    stage["name"] = "facts"
    payload = {
        "version": VERSION, "experiment": "E5", "split": "Q_DEV", "patch": PATCH, "supersedes": SUPERSEDES,
        "statement": "The three cases are illustrations, not statistical evidence (PRESPEC_E5 section 1).",
        "prespec": document_record(PRESPEC), "git_head": git_head(),
        "rules": {rule: {"name": RULE_NAMES[rule], "text": RULE_TEXT[rule]} for rule in RULES}, "tie_and_distinct": TIE_TEXT,
        "sign": SIGN_TEXT, "seed": None, "selection_is_deterministic": True,
        "selection": {**{rule: (None if selection[rule] is None else selection[rule].decode("ascii")) for rule in RULES},
                      "a_threshold": selection["a_threshold"]},
        "selection_audit": audit,
        "rows": checks["rows"], "matches": checks["matches"], "table": {"checks": built["checks"], "sources": checks},
        "gate": gate_result,
        "map_check": {"json": str(out / MAP_JSON), "json_sha256": sha256_file(out / MAP_JSON), "png": str(out / MAP_PNG),
                      "png_sha256": png_sha, "passed": check["passed"], "max_border_abs_diff_px": check["max_border_abs_diff_px"],
                      "max_checked_turret_distance_px": check["max_checked_turret_distance_px"]},
        "time_axis_rule": "states at tau - 60 s + 5 s k up to the h90 endpoint (times before 0 left out) plus the endpoint, "
                          "built by controls_states.build_match + semantic_rows and scored by score.LockedV('final', cpu) "
                          "through queries.transform; V = calibrated final V (labels' p)",
        "minimap_rule": "F(tau) = held frame at tau (e1_gate.xy_at_times, spatial.py's frame rules); a position that is not "
                        "finite or has a zero coordinate is unknown and not drawn (the population rule, headcount_tau.py "
                        "233-235); a known position equal to the player's own last death position is drawn hollow grey as "
                        "likely dead; kill positions = recorded CHAMPION_KILL x, y",
        "likely_dead_rule": LIKELY_DEAD_RULE,
        "v": {**ctx.v.identity, "bundle": str(ctx.bundle_dir), "rescore_atol": V_REPRO_ATOL,
              "note": "labels were scored on CUDA; here on CPU (no GPU), compared at tau and the endpoint"},
        "champion_names": ctx.names_record, "projection_files_sha256": ctx.projection_sha256,
        "cases": cases, "figures": figures, "stages": stages, "versions": versions(),
        "run": run_record(rows=checks["rows"], matches=checks["matches"], inputs=inputs, code=code, started_utc=started),
        "elapsed_seconds": time.perf_counter() - tick,
    }
    write_once(out / CASES_JSON, _jsonable(payload))
    print(json.dumps({"written": str(out / CASES_JSON), "figures": {r: f["file"] for r, f in figures.items()}}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", default=str(TOG_E5), help="output folder (default tog_e5_2; another folder for a dry run)")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only; set before torch loads (controls_states, score) and CUDA starts
    from studies.label_production_20261005.controls_states import below_normal_priority

    below_normal_priority()
    return run(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
