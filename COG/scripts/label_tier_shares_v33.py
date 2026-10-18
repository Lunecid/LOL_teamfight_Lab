"""Which step of the market_event rule decides each v3.3 engagement label (item C4-label-tier-shares).

sec_label.tex (Step 4 and Step 5 of the label) states the rule in the order gameplay/labels.py applies it
(_compute_label_market_event, then _lex_refine), and needs the share of engagements each step decides:

    swing       |Delta| > LABEL_GOLD_DEADZONE (300 g): the sign of the priced gold swing decides
    kills       else K != 0: net cluster kills inside the disc, first kill <= t <= last kill
    survivors   else N != 0: blue minus red living champions in the last frame at or before the last kill
    structures  else S != 0: net plates, structures and epic monsters inside the disc in the window
    draw        else: tie policy "drop" stores y_market_event = -1

labels.py returns only the verdict.  This script recovers the deciding step without changing labels.py, in
two independent ways, and refuses any row on which they disagree:

  1. arithmetic -- decide_tier() re-runs _compute_label_market_event's swing sum and _lex_refine's K, N and S
     on the same inputs, calling labels.py's own helpers (_resolve_label_window, _events_in_window,
     _split_label_type, attribute_events, _event_price_table, _apply_price_table_variant, _first_tower_ts,
     _label_event_team_sign, _priced_event_gold) and gameplay.pipeline_interp.interpolate_node_global, and
     derives the verdict from the first step that is not zero;
  2. call trace -- the labels.py call behind every stored y_market_event (scripts/build_corpus_shard.py
     label_ref under registry variant "market_event" and tie policy "drop") runs with compute_label,
     _lex_refine and gameplay.pipeline_interp.interpolate_node_global wrapped by recorders, in this process
     only.  compute_label's recorded arguments are the inputs decide_tier receives.  The swing decided iff
     _lex_refine was never entered; K decided iff it was entered and the survivor frame was never read;
     N, S or a draw iff the frame was read, at decide_tier's query time, and N decided iff the blue-minus-red
     alive count of the frame labels.py read is not zero (S versus a draw then follows from the verdict).

Per row, the verdict of (1), the verdict of (2) and the stored y_market_event of corpus_shards_v33 must be equal,
and (1)'s step must match (2)'s trace.  One failure aborts the shard (exit code 3, no npz) and the whole run.
Each worker first runs a synthetic self-test (one engagement per step, the dead-zone boundary |Delta| = 300 g,
a kill outside the first-last kill interval, a turret execute, an event outside the attribution disc).

Rows are the corpus rows, rebuilt exactly as scripts/build_label_sidecars_v33.py rebuilds them, by calling that
script's own functions: partition (build_corpus_shard.shard_match_ids), build_fight_index under
LOL_CFG_PRESET=v3.3 and the build's effective overrides, build_ms_sequence's row guards (row_label_for_ref),
then verify_identity row for row against corpus_shards_v33/shard_XXX.npz (groups, engage_ts, patch, the four
participation / presence counts, y, y_market_event).

Shares and intervals (aggregation in the parent, after every shard passed):
  * share_of_labelled  -- step / engagements with y_market_event in {0, 1} (the 532,547 of sec_label.tex);
    "refined" = kills + survivors + structures (|Delta| <= 300 g);
  * share_of_detected  -- step / all corpus rows, draws included (the 566,452 detected engagements);
  * blue_win_rate      -- y_market_event = 1 among the rows each step decides;
  * swing_vs_kill_count -- among swing-decided rows, K of the same sign as Delta, K = 0, K of the opposite sign
    (how often the priced swing overrules the kill count K);
  * groups: overall; participation scale class (smaller side's count n_min from cluster_blue / cluster_red:
    pick n_min <= 1, skirmish 2..3, teamfight >= 4, unknown without a record -- corpus manifest "scale", as
    scripts/run_scale_decomposition.py scale_class(teamfight_min=4)); game-minute band of the engagement start
    engage_ts (lt2 = before minute 2, then 2-10, 10-20, 20-30, 30+ as TIME_BANDS of the evidence-block and
    kill-less analyses);
  * 95 % intervals: match-clustered percentile bootstrap (numpy default_rng(seed); per replicate
    len(matches) integers over all matches of the analysed rows, as run_scale_decomposition.cluster_bootstrap),
    one multiplicity vector per replicate for every share, group and difference, so differences between scale
    classes and between minute bands are paired.

Full run (32 shards, <= 4 worker processes, 1 index process each):

    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/label_tier_shares_v33.py \
        --out-dir D:/LOL_Project/fusion_2615/features/tog_revision/C4-label-tiers/label_tier_shards_v33 \
        --output D:/LOL_Project/fusion_2615/features/tog_revision/C4-label-tiers/label_tier_shares_v33.json \
        --parallel 4 --index-workers 1

Smoke: ``--shards 0,31 --limit-matches-per-shard 60 --parallel 2 --index-workers 1`` (rows stay a corpus prefix).
``--aggregate-only`` recomputes the JSON from existing shard outputs; ``--self-test`` runs only the synthetic check.

Writes, per shard, tiers_shard_XXX.npz (tier int8 0..4 = swing, kills, survivors, structures, draw; y_market_event;
swing_gold; K; N; S; n_events attributed; survivor_frame_age_ms; groups, engage_ts, patch, cluster_blue/red,
present_blue/red, corpus_row), tiers_shard_XXX.json (identity, trace and self-test checks, counts, timing) and
tiers_shard_XXX.log; manifest.json in --out-dir; the aggregate JSON at --output.
"""
from __future__ import annotations

import os

# BLAS threads of the aggregation (the parent); workers get 1 through their environment.  Must precede numpy.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ITEM = "C4-label-tier-shares"
DEFAULT_CORPUS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
DEFAULT_ROOT = Path("D:/LOL_Project/fusion_2615/features/tog_revision/C4-label-tiers")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "label_tier_shards_v33"
DEFAULT_OUTPUT = DEFAULT_ROOT / "label_tier_shares_v33.json"
SEED = 7
IDENTITY_EXIT = 3

TIERS = ("swing", "kills", "survivors", "structures", "draw")
SWING, KILLS, SURVIVORS, STRUCTURES, DRAW = range(5)
NO_LABEL = -1                    # compute_label returned None for a reason other than a draw (never expected)
MARKET_EVENT_TYPES = ("market_event", "event_market", "kill_bounty_lex")   # labels.compute_label dispatch
DROP_TIE_POLICIES = ("drop", "exclude", "none")
STRUCTURE_TYPES = ("ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED")   # labels._lex_refine
MINUTE_BANDS = (("lt2", 0.0, 2.0), ("2-10", 2.0, 10.0), ("10-20", 10.0, 20.0), ("20-30", 20.0, 30.0), ("30+", 30.0, None))
SCALE_CLASSES = ("pick", "skirmish", "teamfight", "unknown")

# Counts the full run must reproduce (checked, never used as inputs).
MANUSCRIPT_COUNTS = {
    # docs/tog_manuscript/sec_label.tex: "33,905 of the 566,452 detected engagements (5.99%), leaving 532,547 labelled
    # engagements from 191,940 matches, of which blue won 50.8%"; A2-label-family/label_sidecars_v33/manifest.json n_rows.
    "rows_total": 566452, "rows_labelled": 532547, "rows_draw": 33905, "matches_labelled": 191940,
    # FEAT/scale_decomposition_v33_market_event.json by_participation_scale n (sec_definition.tex): labelled rows per class.
    "labelled_pick": 101798, "labelled_skirmish": 320878, "labelled_teamfight": 109829, "labelled_unknown": 42,
}
# A2-label-family/label_family_v33.json rowsets.own.variants.market_event positive_rate (sec_label.tex comment)
MANUSCRIPT_BLUE_WIN_RATE = 0.50785

SHARD_ROW_KEYS = ("groups", "engage_ts", "patch", "cluster_blue", "cluster_red", "present_blue", "present_red")

DEVIATIONS = [
    "Rows are rebuilt with scripts/build_label_sidecars_v33.py's own functions (row_label_for_ref, verify_identity), "
    "so that script's declared deviations apply: build_tabular_Xy's feature-sequence guards are not evaluated, hence "
    "every shard is verified row for row against its corpus shard instead of being trusted.",
    "build_fight_index may run with a different worker count than the v3.3 build (--index-workers); refs keep task "
    "order and the identity check confirms the rows.",
    "--limit-matches-per-shard (smoke only) keeps the first N matches of each shard's own order; the rows are then "
    "checked against the corpus prefix.",
    "The call trace patches gameplay.labels.compute_label, gameplay.labels._lex_refine and "
    "gameplay.pipeline_interp.interpolate_node_global with recorders around each label_ref call, in the worker "
    "process only; the recorders call the original functions with unchanged arguments and return their results.",
    "decide_tier reads the survivor frame for every row (also where the swing or K decides) so that N is reported "
    "for every engagement; labels.py reads it only when it reaches that step, which the trace checks.",
    "The bootstrap resamples all matches that hold an analysed row (194,676 in the full corpus, including matches "
    "whose engagements are all draws), not only the 191,940 matches with a labelled engagement; shares of labelled "
    "engagements are ratio estimates within each replicate.",
    "A replicate whose denominator is zero for a share (possible only for very small groups such as 'unknown' or "
    "'lt2') is left out of that share's interval; n_boot_valid records how many replicates remain.",
]


# ----------------------------------------------------------------------------------------------------- helpers
def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git_state() -> tuple:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(PROJECT_ROOT),
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "", None


def _peak_rss_mb():
    """Peak working set of this process in MB (Windows psapi; ru_maxrss elsewhere); None if unavailable."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
            return None
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return None


def _clean(o):
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------ the deciding step
def decide_tier(cache, tm, t_start, *, engage_ts=None, label_end_ts=None, horizon_ms=None, first_kill_ts=None,
                last_kill_ts=None, interp_node_global, anchor_xy=None) -> dict:
    """The step of the market_event rule that decides this engagement, by re-running labels.py's arithmetic.

    Same signature as gameplay.labels.compute_label; the body repeats compute_label (window, attribution,
    REQUIRE_SIGNAL_IN_HORIZON), _compute_label_market_event (swing) and _lex_refine (K, N, S, tie) statement for
    statement, calling labels.py's helpers, so the float sum of the swing and every comparison are the same.
    Returns tier (SWING .. DRAW, NO_LABEL where compute_label would return None without reaching a draw), the
    verdict y (-1 for a draw under the drop policy), swing_gold, K, N (float, blue minus red alive), S,
    n_events (attributed), survivor_frame_age_ms, and whether the survivor frame exists (alive column present).
    """
    from core.config import cfg
    from data.events_index import _events_in_window
    from gameplay import labels as L

    out = {"tier": NO_LABEL, "y": -1, "swing_gold": float("nan"), "K": 0, "N": float("nan"), "S": 0,
           "n_events": 0, "survivor_frame_age_ms": -1, "alive_measure_ts": -1,
           "alive_column": L.NODE_IDX.get("alive", None) is not None}
    win = L._resolve_label_window(cache, t_start, engage_ts=engage_ts, label_end_ts=label_end_ts, horizon_ms=horizon_ms)
    if win is None:
        return out
    s_ms, e_ms, _ = win
    evs = _events_in_window(cache, s_ms, e_ms)
    label_type, attribution = L._split_label_type(str(getattr(cfg, "LABEL_TYPE", "micro_win")))
    if label_type not in MARKET_EVENT_TYPES:
        raise ValueError(f"decide_tier reproduces market_event only; cfg.LABEL_TYPE is {cfg.LABEL_TYPE!r}")
    evs = L.attribute_events(evs, anchor_xy, attribution)
    out["n_events"] = len(evs)
    if getattr(cfg, "REQUIRE_SIGNAL_IN_HORIZON", False):
        if not any(str(e.get("type", "")).upper() in ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL") for e in evs):
            return out
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()
    if tie_policy not in DROP_TIE_POLICIES:
        raise ValueError(f"draws are identifiable only under tie policy 'drop' (y_market_event = -1); cfg has {tie_policy!r}")

    # _compute_label_market_event: the priced swing
    deadzone = float(getattr(cfg, "LABEL_GOLD_DEADZONE", 300.0))
    table = L._apply_price_table_variant(L._event_price_table())
    first_tower = L._first_tower_ts(cache) if table else None
    gd = 0.0
    for e in evs:
        et = str(e.get("type", "")).upper()
        sign = L._label_event_team_sign(e, tm)
        if sign == 0:
            continue
        if et == "CHAMPION_KILL":
            g = max(0.0, L.safe_float(e.get("bounty", 0.0))) + max(0.0, L.safe_float(e.get("shutdownBounty", 0.0)))
            assists = e.get("assistingParticipantIds", [])
            g += float(table.get("kills", 0.0)) + float(table.get("assists", 0.0)) * (len(assists) if isinstance(assists, list) else 0)
            gd += float(sign) * g
        else:
            gd += float(sign) * L._priced_event_gold(e, table, first_tower)

    # _lex_refine: K (cluster kills), S (structures), N (survivors at the last kill)
    kd = 0
    struct = 0
    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            if first_kill_ts is not None and last_kill_ts is not None:
                kill_ts = int(e.get("timestamp", 0) or 0)
                if kill_ts < first_kill_ts or kill_ts > last_kill_ts:
                    continue
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                kd += 1
            elif tm.get(killer, 0) == 200:
                kd -= 1
        elif et in STRUCTURE_TYPES:
            struct += L._label_event_team_sign(e, tm)
    alive_measure_ts = e_ms if not (last_kill_ts and last_kill_ts > 0) else last_kill_ts
    out["alive_measure_ts"] = int(alive_measure_ts)
    alive_idx = L.NODE_IDX.get("alive", None)
    blue_alive = red_alive = None
    if alive_idx is not None:
        node_end, _ = interp_node_global(cache, alive_measure_ts)
        tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
        blue_alive = float(node_end[np.where(tids == 100)[0], alive_idx].sum())
        red_alive = float(node_end[np.where(tids == 200)[0], alive_idx].sum())
        ts = np.asarray(cache["minute_ts"])
        fi = 0 if len(ts) == 1 else max(0, int(np.searchsorted(ts, int(alive_measure_ts), side="right")) - 1)
        out["survivor_frame_age_ms"] = int(alive_measure_ts) - int(ts[fi])
        out["N"] = blue_alive - red_alive

    if gd > deadzone:
        tier, y = SWING, 1
    elif gd < -deadzone:
        tier, y = SWING, 0
    elif kd != 0:
        tier, y = KILLS, (1 if kd > 0 else 0)
    elif alive_idx is not None and blue_alive != red_alive:
        tier, y = SURVIVORS, (1 if blue_alive > red_alive else 0)
    elif struct != 0:
        tier, y = STRUCTURES, (1 if struct > 0 else 0)
    else:
        tier, y = DRAW, -1
    out.update(tier=tier, y=y, swing_gold=float(gd), K=int(kd), S=int(struct))
    return out


class LabelTrace:
    """Recorders around the labels.py call path, installed only inside ``active()``.

    compute_label: arguments and result; _lex_refine: calls and result; interp: calls of the survivor-frame
    reader handed to compute_label.  ``interp_module`` is gameplay.pipeline_interp, whose attribute
    scripts/build_corpus_shard.py label_ref imports at call time.
    """

    def __init__(self, labels_module, interp_module):
        self.L = labels_module
        self.PI = interp_module
        self.orig_compute_label = labels_module.compute_label
        self.orig_lex_refine = labels_module._lex_refine
        self.orig_interp = interp_module.interpolate_node_global
        self.reset()

    def reset(self):
        self.compute_calls = []
        self.lex_calls = 0
        self.lex_results = []
        self.interp_calls = 0
        self.interp_queries = []
        self.interp_nodes = []

    def wrap_interp(self, fn):
        def interp(*a, **k):
            self.interp_calls += 1
            res = fn(*a, **k)
            self.interp_queries.append(int(a[1]) if len(a) > 1 else int(k.get("q_ms", -1)))
            self.interp_nodes.append(np.array(res[0], copy=True))
            return res
        return interp

    @contextmanager
    def active(self):
        def compute_label(*a, **k):
            res = self.orig_compute_label(*a, **k)
            self.compute_calls.append((a, dict(k), res))
            return res

        def lex_refine(*a, **k):
            self.lex_calls += 1
            res = self.orig_lex_refine(*a, **k)
            self.lex_results.append(res)
            return res

        self.reset()
        self.L.compute_label = compute_label
        self.L._lex_refine = lex_refine
        self.PI.interpolate_node_global = self.wrap_interp(self.orig_interp)
        try:
            yield self
        finally:
            self.L.compute_label = self.orig_compute_label
            self.L._lex_refine = self.orig_lex_refine
            self.PI.interpolate_node_global = self.orig_interp


def trace_problem(d: dict, y_call: int, trace: LabelTrace):
    """None when the recorded call path is the one the recomputed step implies, else a description."""
    if len(trace.compute_calls) != 1:
        return f"compute_label called {len(trace.compute_calls)} times"
    tier = d["tier"]
    if tier == NO_LABEL:
        return "decide_tier reached no verdict (window or signal guard) although the row is a corpus row"
    exp_lex = 0 if tier == SWING else 1
    exp_interp = 0 if tier in (SWING, KILLS) else (1 if d["alive_column"] else 0)
    if trace.lex_calls != exp_lex or trace.interp_calls != exp_interp:
        return (f"tier {TIERS[tier]} implies _lex_refine x{exp_lex} and survivor frame x{exp_interp}; "
                f"recorded x{trace.lex_calls} and x{trace.interp_calls}")
    if tier in (KILLS, SURVIVORS, STRUCTURES) and trace.lex_results[0] != d["y"]:
        return f"_lex_refine returned {trace.lex_results[0]!r}, tier {TIERS[tier]} gives {d['y']}"
    if tier == DRAW and trace.lex_results[0] is not None:
        return f"_lex_refine returned {trace.lex_results[0]!r} for a draw"
    if trace.interp_calls == 1:
        # the frame labels.py read: same query time, same blue-minus-red alive count as decide_tier's N
        from gameplay import labels as L
        q = trace.interp_queries[0]
        if q != d["alive_measure_ts"]:
            return f"labels.py read the survivor frame at {q} ms, decide_tier at {d['alive_measure_ts']} ms"
        tm = trace.compute_calls[0][0][1]
        alive_idx = L.NODE_IDX["alive"]
        node = trace.interp_nodes[0]
        tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
        n_seen = float(node[np.where(tids == 100)[0], alive_idx].sum()) - float(node[np.where(tids == 200)[0], alive_idx].sum())
        if not (n_seen == d["N"]):
            return f"N from the frame labels.py read is {n_seen}, decide_tier has {d['N']}"
        if (tier == SURVIVORS) != (n_seen != 0.0):
            return f"tier {TIERS[tier]} but N from the frame labels.py read is {n_seen}"
    if int(y_call) != int(d["y"]):
        return f"labels.py verdict {y_call} != recomputed {d['y']} (tier {TIERS[tier]})"
    return None


# ---------------------------------------------------------------------------------------------- self-test
def run_self_test() -> dict:
    """Synthetic engagements through labels.compute_label (traced) and decide_tier; raises on disagreement.

    Uses the cfg of the calling process (the worker: v3.3 preset, variant market_event, tie policy drop).
    Expected steps are asserted when the price table has the v3.3 pooled kills 20 / assists 45 / plates 120 g
    and the dead zone is 300 g; agreement between the two paths is asserted always.
    """
    from core.config import cfg
    from gameplay import labels as L
    from gameplay import pipeline_interp as PI

    alive = L.NODE_IDX["alive"]
    tm = {i: (100 if i <= 5 else 200) for i in range(1, 11)}
    minute_ts = np.array([0, 60000, 120000], dtype=np.int64)

    def cache(events, dead_red=0):
        node = np.zeros((3, 10, len(L.NODE_IDX)), dtype=np.float32)
        node[:, :, alive] = 1.0
        node[0, 5:5 + dead_red, alive] = 0.0
        return {"minute_ts": minute_ts, "node_minute": node, "global_minute": np.zeros((3, 4), np.float32),
                "events": [dict(e) for e in events]}

    def interp(c, q):
        ts = c["minute_ts"]
        i = max(0, int(np.searchsorted(ts, int(q), side="right")) - 1)
        return c["node_minute"][i], c["global_minute"][i]

    def kill(ts, killer, victim, bounty, assists=(), pos=None):
        e = {"type": "CHAMPION_KILL", "timestamp": ts, "killerId": killer, "victimId": victim, "bounty": bounty,
             "shutdownBounty": 0, "assistingParticipantIds": list(assists)}
        if pos is not None:
            e["position"] = {"x": pos[0], "y": pos[1]}
        return e

    def plate(ts, victim_team, pos=None):
        e = {"type": "TURRET_PLATE_DESTROYED", "timestamp": ts, "teamId": victim_team, "laneType": "MID_LANE"}
        if pos is not None:
            e["position"] = {"x": pos[0], "y": pos[1]}
        return e

    trade = [kill(20000, 1, 6, 300), kill(25000, 6, 1, 300)]
    cases = [
        # name, events, dead red champions in frame 0, anchor, expected step, expected y
        ("swing_blue", [kill(20000, 1, 6, 300)], 0, None, SWING, 1),                          # 300 + 20 = 320 > 300
        ("swing_red_with_assist", [kill(20000, 6, 1, 300, (7,))], 0, None, SWING, 0),         # -(300 + 20 + 45)
        ("deadzone_boundary_is_refined", [kill(20000, 1, 6, 280)], 0, None, KILLS, 1),         # 280 + 20 = 300, not > 300
        ("kill_outside_interval_then_structure", [kill(35000, 1, 6, 100), plate(25000, 100)], 0, None, STRUCTURES, 0),
        ("survivors", trade, 1, None, SURVIVORS, 1),                                             # K = 0, 5 v 4 alive
        ("draw", trade, 0, None, DRAW, -1),
        ("turret_execute_is_no_kill", [kill(20000, 0, 6, 300)], 0, None, DRAW, -1),              # killerId 0: s(e) = 0
        ("outside_disc_not_attributed", [kill(20000, 1, 6, 300, pos=(14000, 14000)), plate(25000, 100, pos=(5000, 5000))],
         0, (5000.0, 5000.0), STRUCTURES, 0),
    ]
    table = L._apply_price_table_variant(L._event_price_table())
    v33_prices = (float(table.get("kills", -1)) == 20.0 and float(table.get("assists", -1)) == 45.0
                  and float(table.get("plates", -1)) == 120.0 and float(getattr(cfg, "LABEL_GOLD_DEADZONE", -1)) == 300.0
                  and float(getattr(cfg, "CLUSTER_MAX_DIAMETER", -1)) == 4264.0
                  and float(getattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", 0.0) or 0.0) == 0.0)
    tracer = LabelTrace(L, PI)
    report = {}
    for name, events, dead_red, anchor, exp_tier, exp_y in cases:
        c = cache(events, dead_red)
        kwargs = dict(engage_ts=10000, label_end_ts=40000, first_kill_ts=20000, last_kill_ts=30000, anchor_xy=anchor)
        with tracer.active():
            res = L.compute_label(c, tm, -1, interp_node_global=tracer.wrap_interp(interp), **kwargs)
        y_call = -1 if res is None else int(res)
        d = decide_tier(c, tm, -1, interp_node_global=interp, **kwargs)
        problem = trace_problem(d, y_call, tracer)
        if problem:
            raise AssertionError(f"self-test {name}: {problem}")
        if v33_prices and (d["tier"], d["y"]) != (exp_tier, exp_y):
            raise AssertionError(f"self-test {name}: step {TIERS[d['tier']] if d['tier'] >= 0 else d['tier']} y {d['y']}, "
                                 f"expected {TIERS[exp_tier]} y {exp_y} (swing {d['swing_gold']}, K {d['K']}, N {d['N']}, S {d['S']})")
        report[name] = {"tier": TIERS[d["tier"]], "y": d["y"], "swing_gold": d["swing_gold"], "K": d["K"], "N": d["N"], "S": d["S"]}
    return {"ok": True, "expected_steps_checked": bool(v33_prices), "cases": report}


# --------------------------------------------------------------------------------------------------- worker
def run_worker(args) -> int:
    t_start = time.time()
    SC = _load_script("build_label_sidecars_v33")
    from core.config import CACHE_DIR, cfg
    from core.timeutils import _get_bin_ms, _get_context_ms, _get_horizon_ms
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from gameplay import labels as L
    from gameplay import pipeline_interp as PI
    from gameplay.pipeline import compute_label_targets
    from train.baseline import _ref_anchor_xy, _ref_engage_ts, _ref_first_kill_ts, _ref_label_end_ts, _ref_last_kill_ts
    bcs = SC._load_script("build_corpus_shard")

    m = SC.load_corpus_manifest(args.corpus)
    wrong = {k: getattr(cfg, k, None) for k, v in SC.V33_DETECTOR.items() if float(getattr(cfg, k, -1)) != float(v)}
    wrong.update({k: getattr(cfg, k, None) for k, v in SC.V33_LABEL_BASE.items() if getattr(cfg, k, None) != v})
    if getattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", None):
        wrong["LABEL_EVENT_PRICE_OVERRIDES"] = cfg.LABEL_EVENT_PRICE_OVERRIDES
    if wrong:
        raise SystemExit(f"[shard {args.shard}] cfg is not the v3.3 base state (run with LOL_CFG_PRESET=v3.3): {wrong}")
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.DUMP_FIGHTS = False
    if args.index_workers > 0:
        cfg.FIGHT_INDEX_NUM_WORKERS = int(args.index_workers)

    row_label = str(m["label"]["row_label"])
    row_tie = str(m["label"]["row_tie_policy"])
    extra_tie = str(m["label"]["extra_tie_policy"])
    if extra_tie not in DROP_TIE_POLICIES:
        raise SystemExit(f"[shard {args.shard}] y_market_event was stored under tie policy {extra_tie!r}; draws are not -1")
    variant = L.get_label_variant("market_event")
    if variant.overrides or not variant.stored:
        raise SystemExit(f"[shard {args.shard}] registry variant market_event is not the stored, override-free label: {variant.as_dict()}")

    out_dir = Path(args.out_dir)
    stem = f"tiers_shard_{args.shard:03d}"
    with L.cfg_override({"LABEL_TIE_POLICY": extra_tie}), L.variant_cfg(variant):
        self_test = run_self_test()
    print(f"[shard {args.shard}] self-test ok ({len(self_test['cases'])} cases, expected steps checked: "
          f"{self_test['expected_steps_checked']})", flush=True)

    corpus_path = Path(args.corpus) / f"shard_{args.shard:03d}.npz"
    with np.load(corpus_path, allow_pickle=False) as blob:
        corpus = {k: blob[k] for k in blob.files if k != "X"}
    if "y_market_event" not in corpus:
        raise SystemExit(f"[shard {args.shard}] {corpus_path} has no y_market_event")

    full_mine = bcs.shard_match_ids(bcs.cached_match_ids(CACHE_DIR), args.shard, int(m["num_shards"]),
                                    n_matches=m.get("n_matches"), seed=int(m["seed"]))
    pos = {mid: i for i, mid in enumerate(full_mine)}
    corpus_pos = np.array([pos.get(str(g), -1) for g in corpus["groups"]], dtype=np.int64)
    if (corpus_pos < 0).any() or (np.diff(corpus_pos) < 0).any():
        msg = (f"[shard {args.shard}] partition mismatch: {int((corpus_pos < 0).sum())} corpus rows belong to matches outside "
               f"this shard's partition of {CACHE_DIR} or appear out of shard order")
        json.dump({"shard": args.shard, "error": msg}, open(out_dir / f"{stem}.mismatch.json", "w", encoding="utf-8"), indent=1)
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
        return IDENTITY_EXIT
    limit = int(args.limit_matches_per_shard or 0)
    mine = full_mine[:limit] if limit > 0 else full_mine
    print(f"[shard {args.shard}] matches={len(mine)} of {len(full_mine)}", flush=True)

    t0 = time.time()
    with L.cfg_override({"LABEL_TYPE": row_label, "LABEL_TIE_POLICY": row_tie}):
        refs = build_fight_index(cache_match_ids=mine)
    t_index = time.time() - t0
    print(f"[shard {args.shard}] refs={len(refs)} ({t_index:.0f}s)", flush=True)

    ref_fields = (_ref_engage_ts, _ref_label_end_ts, _ref_first_kill_ts, _ref_last_kill_ts, _ref_anchor_xy)
    ctx_ms, bin_ms, horizon_ms = _get_context_ms(), _get_bin_ms(), _get_horizon_ms()
    gap_ms = int(getattr(cfg, "PREDICTION_GAP_MS", 0))

    tracer = LabelTrace(L, PI)
    used, y_row = [], []
    cols = {k: [] for k in ("y_market_event", "y_tier_verdict", "tier", "swing_gold", "K", "N", "S", "n_events",
                            "survivor_frame_age_ms")}
    problems: list = []
    n_problems = 0
    t0 = time.time()
    t_guard = t_call = t_tier = 0.0
    for mid, rs in SC._refs_by_match(refs):
        pack = load_match_cache(mid)
        if not pack:
            continue
        keep = []
        t1 = time.perf_counter()
        with L.cfg_override({"LABEL_TYPE": row_label, "LABEL_TIE_POLICY": row_tie}):
            for r in rs:
                y = SC.row_label_for_ref(pack, r, ctx_ms=ctx_ms, bin_ms=bin_ms, horizon_ms=horizon_ms,
                                         prediction_gap_ms=gap_ms, compute_label_targets=compute_label_targets,
                                         ref_fields=ref_fields)
                if y is not None:
                    keep.append((r, y))
        t_guard += time.perf_counter() - t1
        if not keep:
            continue
        tm = bcs.team_map_int(pack)
        with L.cfg_override({"LABEL_TIE_POLICY": extra_tie}), L.variant_cfg(variant):
            for r, yr in keep:
                t1 = time.perf_counter()
                with tracer.active():
                    y_call = bcs.label_ref(pack, tm, r)
                t2 = time.perf_counter()
                t_call += t2 - t1
                if len(tracer.compute_calls) == 1:
                    a, k, _ = tracer.compute_calls[0]
                    d = decide_tier(*a, **{**k, "interp_node_global": tracer.orig_interp})
                else:
                    d = {"tier": NO_LABEL, "y": -1, "swing_gold": float("nan"), "K": 0, "N": float("nan"), "S": 0,
                         "n_events": 0, "survivor_frame_age_ms": -1, "alive_measure_ts": -1, "alive_column": True}
                t_tier += time.perf_counter() - t2
                problem = trace_problem(d, y_call, tracer)
                if problem:
                    n_problems += 1
                    if len(problems) < 20:
                        problems.append({"match_id": str(r.match_id), "engage_ts": int(r.t_start_ts), "problem": problem,
                                         "y_call": int(y_call), "tier": int(d["tier"]), "y_tier": int(d["y"]),
                                         "swing_gold": d["swing_gold"], "K": d["K"], "N": d["N"], "S": d["S"]})
                used.append(r)
                y_row.append(yr)
                cols["y_market_event"].append(int(y_call))
                cols["y_tier_verdict"].append(int(d["y"]))
                for key in ("tier", "swing_gold", "K", "N", "S", "n_events", "survivor_frame_age_ms"):
                    cols[key].append(d[key])
    t_rows = time.time() - t0

    side = {
        "groups": np.array([r.match_id for r in used], dtype=str),
        "engage_ts": np.array([r.t_start_ts for r in used], dtype=np.int64),
        "patch": np.array([r.patch for r in used], dtype=str),
        "cluster_blue": np.array([r.det_cluster_blue for r in used], dtype=np.int16),
        "cluster_red": np.array([r.det_cluster_red for r in used], dtype=np.int16),
        "present_blue": np.array([r.det_present_blue for r in used], dtype=np.int16),
        "present_red": np.array([r.det_present_red for r in used], dtype=np.int16),
        "y_row": np.array(y_row, dtype=np.int8),
        "y_market_event": np.array(cols["y_market_event"], dtype=np.int8),
    }
    tier = np.array(cols["tier"], dtype=np.int8)
    y_tier = np.array(cols["y_tier_verdict"], dtype=np.int8)
    checks = {"trace_or_verdict_problems": int(n_problems), "problem_examples": problems}
    try:
        if n_problems:
            raise SC.SidecarIdentityError(f"{n_problems} rows where the recomputed step, the call trace or the verdicts disagree; "
                                          f"first: {problems[0]}")
        identity = SC.verify_identity(side, corpus, prefix=limit > 0, included_matches=(mine if limit > 0 else None),
                                      stored_variants=("market_event",))
        stored = np.asarray(corpus["y_market_event"])[:len(y_tier)]
        bad = np.flatnonzero(y_tier != stored)
        if len(bad):
            i = int(bad[0])
            raise SC.SidecarIdentityError(f"recomputed verdict vs corpus y_market_event: {len(bad)} rows differ; first at row {i}: "
                                          f"{int(y_tier[i])} vs {int(stored[i])} (tier {int(tier[i])})")
        if (tier < 0).any():
            raise SC.SidecarIdentityError(f"{int((tier < 0).sum())} rows reached no verdict")
        if not np.array_equal(tier == DRAW, stored == -1):
            raise SC.SidecarIdentityError("draw rows (tier) != rows with stored y_market_event = -1")
        checks.update(identity=identity, recomputed_verdict_equals_stored_rows=int(len(y_tier)),
                      labels_call_equals_stored_rows=int(len(y_tier)), trace_consistent_rows=int(len(y_tier)))
    except SC.SidecarIdentityError as e:
        msg = f"[shard {args.shard}] MISMATCH against {corpus_path}: {e}"
        json.dump(_clean({"shard": args.shard, "error": msg, "n_rows": len(used), "checks": checks}),
                  open(out_dir / f"{stem}.mismatch.json", "w", encoding="utf-8"), indent=1)
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
        return IDENTITY_EXIT
    print(f"[shard {args.shard}] identity ok: {len(used)} rows; recomputed verdict == labels.py call == stored "
          f"y_market_event on every row; trace consistent", flush=True)

    counts = {t: int((tier == i).sum()) for i, t in enumerate(TIERS)}
    print(f"[shard {args.shard}] steps: " + ", ".join(f"{t} {c}" for t, c in counts.items()), flush=True)
    tmp = out_dir / f"{stem}.partial.npz"
    np.savez_compressed(
        tmp, corpus_row=np.arange(len(used), dtype=np.int32), **side, tier=tier, y_tier_verdict=y_tier,
        swing_gold=np.array(cols["swing_gold"], dtype=np.float64), K=np.array(cols["K"], dtype=np.int16),
        N=np.array(cols["N"], dtype=np.float32), S=np.array(cols["S"], dtype=np.int16),
        n_events=np.array(cols["n_events"], dtype=np.int32),
        survivor_frame_age_ms=np.array(cols["survivor_frame_age_ms"], dtype=np.int64))
    os.replace(tmp, out_dir / f"{stem}.npz")
    git_commit, git_dirty = _git_state()
    info = {
        "shard": args.shard, "corpus_shard": str(corpus_path), "cache_dir": str(CACHE_DIR),
        "git_commit": git_commit, "git_dirty": git_dirty, "preset": os.environ.get("LOL_CFG_PRESET"),
        "n_matches_shard": len(full_mine), "n_matches_used": len(mine), "limit_matches_per_shard": limit or None,
        "n_refs": len(refs), "n_rows": len(used), "n_matches_with_rows": int(len(np.unique(side["groups"]))),
        "tier_counts": counts, "self_test": self_test, "checks": checks, "price_table": SC.check_price_table(m),
        "timing_s": {"fight_index": round(t_index, 1), "rows": round(t_rows, 1), "row_guards": round(t_guard, 1),
                     "labels_call": round(t_call, 1), "decide_tier": round(t_tier, 1), "total": round(time.time() - t_start, 1)},
        "peak_rss_mb": _peak_rss_mb(),
    }
    json.dump(_clean(info), open(out_dir / f"{stem}.json", "w", encoding="utf-8"), indent=1)
    print(f"[shard {args.shard}] wrote {out_dir / (stem + '.npz')} in {time.time() - t_start:.0f}s "
          f"(peak {info['peak_rss_mb']} MB)", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- aggregation
def scale_classes(blue: np.ndarray, red: np.ndarray, scale: dict) -> np.ndarray:
    """Participation scale class per row, as scripts/run_scale_decomposition.py scale_class with the manifest's cuts."""
    pick_max = int(scale.get("pick_max", 1))
    skirmish_min = int(scale.get("skirmish_min", 2))
    teamfight_min = int(scale.get("teamfight_min", 4))
    blue = np.asarray(blue, dtype=np.int64)
    red = np.asarray(red, dtype=np.int64)
    smaller = np.minimum(blue, red)
    out = np.full(len(smaller), "unknown", dtype=object)
    out[smaller <= pick_max] = "pick"
    out[(smaller >= skirmish_min) & (smaller < teamfight_min)] = "skirmish"
    out[smaller >= teamfight_min] = "teamfight"
    out[(blue < 0) | (red < 0)] = "unknown"
    return out


def minute_band_masks(engage_ts: np.ndarray) -> dict:
    minute = np.asarray(engage_ts, dtype=np.float64) / 60000.0
    out = {}
    for name, lo, hi in MINUTE_BANDS:
        sel = minute >= lo
        if hi is not None:
            sel &= minute < hi
        out[name] = sel
    return out


def load_shards(out_dir: Path, shards: list) -> dict:
    keys = SHARD_ROW_KEYS + ("y_market_event", "tier", "swing_gold", "K", "N", "S")
    parts = {k: [] for k in keys}
    per_shard = {}
    for i in shards:
        stem = f"tiers_shard_{i:03d}"
        info_path = out_dir / f"{stem}.json"
        if not info_path.exists() or not (out_dir / f"{stem}.npz").exists():
            raise SystemExit(f"shard {i}: {out_dir / stem}.npz / .json missing")
        info = json.load(open(info_path, encoding="utf-8"))
        chk = info.get("checks") or {}
        if not (chk.get("identity") or {}).get("ok") or chk.get("trace_or_verdict_problems") != 0:
            raise SystemExit(f"shard {i}: checks did not pass ({info_path})")
        with np.load(out_dir / f"{stem}.npz", allow_pickle=False) as z:
            for k in keys:
                parts[k].append(z[k])
        per_shard[str(i)] = {"n_rows": info["n_rows"], "tier_counts": info["tier_counts"],
                             "limit_matches_per_shard": info.get("limit_matches_per_shard"),
                             "identity_ok": True, "rows_checked": chk.get("recomputed_verdict_equals_stored_rows"),
                             "self_test_ok": bool((info.get("self_test") or {}).get("ok")),
                             "git_commit": info.get("git_commit"), "git_dirty": info.get("git_dirty"),
                             "timing_s": info.get("timing_s"), "peak_rss_mb": info.get("peak_rss_mb")}
    return {k: np.concatenate(v) for k, v in parts.items()}, per_shard


RATIOS = {
    "share_of_labelled": {**{t: (t, "labelled") for t in TIERS[:4]}, "refined": ("refined", "labelled")},
    "share_of_detected": {t: (t, "all") for t in TIERS},
    "blue_win_rate": {**{t: (f"blue_{t}", t) for t in TIERS[:4]}, "labelled": ("blue_labelled", "labelled")},
    "swing_vs_kill_count": {"same_sign": ("swing_k_same", "swing"), "no_net_kills": ("swing_k_zero", "swing"),
                            "opposite_sign": ("swing_k_opposite", "swing")},
}
DIFF_METRICS = [("share_of_labelled", t) for t in (*TIERS[:4], "refined")] + [("share_of_detected", "draw")]
DIFF_PAIRS = [("teamfight", "pick"), ("skirmish", "pick"), ("teamfight", "skirmish"),
              ("10-20", "2-10"), ("20-30", "10-20"), ("30+", "20-30"), ("30+", "2-10")]


def row_indicators(a: dict) -> dict:
    tier = a["tier"].astype(np.int64)
    y = a["y_market_event"].astype(np.int64)
    k_sign = np.sign(a["K"].astype(np.int64))
    g_sign = np.sign(a["swing_gold"])
    ind = {"all": np.ones(len(tier), dtype=bool), "labelled": tier != DRAW, "refined": (tier >= KILLS) & (tier <= STRUCTURES)}
    for i, t in enumerate(TIERS):
        ind[t] = tier == i
    for t in TIERS[:4]:
        ind[f"blue_{t}"] = ind[t] & (y == 1)
    ind["blue_labelled"] = y == 1
    ind["swing_k_same"] = ind["swing"] & (k_sign != 0) & (k_sign == g_sign)
    ind["swing_k_zero"] = ind["swing"] & (k_sign == 0)
    ind["swing_k_opposite"] = ind["swing"] & (k_sign != 0) & (k_sign != g_sign)
    return ind


def bootstrap_shares(groups_col: np.ndarray, group_masks: dict, ind: dict, n_boot: int, seed: int, batch: int = 32):
    """Point shares, percentile CIs and paired differences; one match-multiplicity vector per replicate."""
    uniq, inv = np.unique(groups_col, return_inverse=True)
    n_m = len(uniq)
    names = list(ind)
    gnames = list(group_masks)
    colkeys = [(g, n) for g in gnames for n in names]
    A = np.zeros((n_m, len(colkeys)), dtype=np.float32)
    for j, (g, n) in enumerate(colkeys):
        A[:, j] = np.bincount(inv, weights=(group_masks[g] & ind[n]).astype(np.float64), minlength=n_m)
    point_sums = A.sum(axis=0, dtype=np.float64)
    col = {ck: j for j, ck in enumerate(colkeys)}
    rng = np.random.default_rng(seed)
    sums = np.empty((max(int(n_boot), 0), len(colkeys)), dtype=np.float64)
    b = 0
    while b < n_boot:
        nb = min(batch, n_boot - b)
        C = np.empty((nb, n_m), dtype=np.float32)
        for r in range(nb):
            C[r] = np.bincount(rng.integers(0, n_m, size=n_m), minlength=n_m)
        sums[b:b + nb] = C @ A
        b += nb

    def ratio(numv, denv):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denv > 0, numv / np.where(denv > 0, denv, 1.0), np.nan)

    draws = {}
    results = {}
    for g in gnames:
        gres = {"counts": {n: int(round(point_sums[col[(g, n)]])) for n in ("all", "labelled", *TIERS)},
                "n_matches": int((A[:, col[(g, "all")]] > 0).sum())}
        for block, stats in RATIOS.items():
            gres[block] = {}
            for sname, (num, den) in stats.items():
                pn, pd = point_sums[col[(g, num)]], point_sums[col[(g, den)]]
                est = float(pn / pd) if pd > 0 else float("nan")
                dr = ratio(sums[:, col[(g, num)]], sums[:, col[(g, den)]]) if n_boot > 0 else np.empty(0)
                draws[(g, block, sname)] = dr
                ok = dr[np.isfinite(dr)]
                gres[block][sname] = {
                    "estimate": est, "numerator": int(round(pn)), "denominator": int(round(pd)),
                    "ci_2.5": float(np.percentile(ok, 2.5)) if len(ok) else None,
                    "ci_97.5": float(np.percentile(ok, 97.5)) if len(ok) else None,
                    "se_boot": float(ok.std(ddof=1)) if len(ok) > 1 else None, "n_boot_valid": int(len(ok)),
                }
        results[g] = gres
    diffs = {}
    for block, sname in DIFF_METRICS:
        key = f"{block}.{sname}"
        diffs[key] = {}
        for ga, gb in DIFF_PAIRS:
            if ga not in results or gb not in results:
                continue
            ea, eb = results[ga][block][sname]["estimate"], results[gb][block][sname]["estimate"]
            d = draws[(ga, block, sname)] - draws[(gb, block, sname)] if n_boot > 0 else np.empty(0)
            ok = d[np.isfinite(d)]
            diffs[key][f"{ga}_minus_{gb}"] = {
                "estimate": (ea - eb) if (ea is not None and eb is not None) else None,
                "ci_2.5": float(np.percentile(ok, 2.5)) if len(ok) else None,
                "ci_97.5": float(np.percentile(ok, 97.5)) if len(ok) else None,
                "n_boot_valid": int(len(ok)),
            }
    return results, diffs, {"n_matches": int(n_m), "n_boot": int(n_boot), "seed": int(seed)}


def _pct(s: dict) -> str:
    if s.get("estimate") is None or not math.isfinite(s["estimate"]):
        return "n/a"
    lo, hi = s.get("ci_2.5"), s.get("ci_97.5")
    ci = f" [{100 * lo:.2f}, {100 * hi:.2f}]" if lo is not None and hi is not None else ""
    return f"{100 * s['estimate']:.2f}%{ci}"


def aggregate(out_dir: Path, shards: list, output: Path, corpus_manifest: dict, *, n_boot: int, seed: int,
              full_expected: bool, run_meta: dict) -> dict:
    t0 = time.time()
    a, per_shard = load_shards(out_dir, shards)
    n = len(a["tier"])
    scale = corpus_manifest.get("scale") or {"pick_max": 1, "skirmish_min": 2, "teamfight_min": 4}
    cls = scale_classes(a["cluster_blue"], a["cluster_red"], scale)
    bands = minute_band_masks(a["engage_ts"])
    group_masks = {"overall": np.ones(n, dtype=bool)}
    group_masks.update({c: cls == c for c in SCALE_CLASSES})
    group_masks.update(bands)
    ind = row_indicators(a)
    results, diffs, boot = bootstrap_shares(a["groups"].astype(str), group_masks, ind, n_boot, seed)

    labelled = a["y_market_event"] >= 0
    got = {
        "rows_total": int(n), "rows_labelled": int(labelled.sum()), "rows_draw": int((a["tier"] == DRAW).sum()),
        "matches_labelled": int(len(np.unique(a["groups"][labelled]))),
        **{f"labelled_{c}": int(((cls == c) & labelled).sum()) for c in SCALE_CLASSES},
    }
    blue_rate = float((a["y_market_event"][labelled] == 1).mean()) if labelled.any() else float("nan")
    reproduction = {k: {"expected": v, "got": got[k], "equal": got[k] == v} for k, v in MANUSCRIPT_COUNTS.items()}
    reproduction["blue_win_rate_5dp"] = {"expected": MANUSCRIPT_BLUE_WIN_RATE, "got": round(blue_rate, 5),
                                         "equal": round(blue_rate, 5) == MANUSCRIPT_BLUE_WIN_RATE}
    all_equal = all(v["equal"] for v in reproduction.values())
    band_hits = np.sum(np.vstack(list(bands.values())), axis=0) if n else np.zeros(0, dtype=np.int64)

    o = results["overall"]
    headline = {
        "share_of_labelled_percent_ci": {t: _pct(o["share_of_labelled"][t]) for t in (*TIERS[:4], "refined")},
        "draws_share_of_detected_percent_ci": _pct(o["share_of_detected"]["draw"]),
        "counts": o["counts"],
        "text": (f"Of {o['counts']['labelled']:,} labelled engagements, the gold swing decides "
                 f"{_pct(o['share_of_labelled']['swing'])}, K {_pct(o['share_of_labelled']['kills'])}, "
                 f"N {_pct(o['share_of_labelled']['survivors'])} and S {_pct(o['share_of_labelled']['structures'])}; "
                 f"draws are {_pct(o['share_of_detected']['draw'])} of {o['counts']['all']:,} detected engagements "
                 "(95% match-clustered bootstrap intervals)."),
    }
    record = {
        "item": ITEM, "script": "scripts/label_tier_shares_v33.py", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        **run_meta,
        "full_corpus": bool(full_expected),
        "rule": {
            "order": list(TIERS),
            "swing": "|Delta| > LABEL_GOLD_DEADZONE: sign of the priced swing (gameplay/labels.py _compute_label_market_event)",
            "kills": "else K != 0: net CHAMPION_KILL sides with first_kill_ts <= t <= last_kill_ts among attributed events (_lex_refine kd)",
            "survivors": "else N != 0: blue minus red alive over all ten participants, interpolate_node_global at the last kill (_lex_refine)",
            "structures": "else S != 0: net sides of ELITE_MONSTER_KILL, BUILDING_KILL, TURRET_PLATE_DESTROYED among attributed events (_lex_refine struct)",
            "draw": "else: tie policy drop, y_market_event = -1",
            "deadzone_g": 300.0, "label_key": "y_market_event", "tie_policy": corpus_manifest["label"]["extra_tie_policy"],
        },
        "groups": {
            "scale_class": {"source": "cluster_blue / cluster_red (participation counts), smaller side n_min; "
                                      "scripts/run_scale_decomposition.py scale_class", "cuts": scale},
            "minute_band": {"source": "engage_ts / 60000 (engagement start tau)",
                            "bands_minutes": {nm: [lo, hi] for nm, lo, hi in MINUTE_BANDS},
                            "rows_in_no_band": int((band_hits == 0).sum()),
                            "rows_in_more_than_one_band": int((band_hits > 1).sum())},
        },
        "bootstrap": {**boot, "method": "match-clustered percentile bootstrap (2.5 / 97.5); one multiplicity vector per "
                                       "replicate shared by every share, group and difference (paired)",
                      "clusters": "every match holding an analysed row, draws included"},
        "checks": {
            "shards": len(per_shard), "rows_checked_verdict_equals_stored": int(sum(s["rows_checked"] or 0 for s in per_shard.values())),
            "every_shard_identity_ok": all(s["identity_ok"] for s in per_shard.values()),
            "every_shard_self_test_ok": all(s["self_test_ok"] for s in per_shard.values()),
            "reproduces_manuscript_counts": reproduction if full_expected else None,
            "reproduces_manuscript_counts_all_equal": all_equal if full_expected else None,
            "counts_seen": got, "blue_win_rate": blue_rate,
        },
        "headline": headline,
        "overall": results["overall"],
        "by_scale_class": {c: results[c] for c in SCALE_CLASSES},
        "by_minute_band": {nm: results[nm] for nm, _, _ in MINUTE_BANDS},
        "differences": diffs,
        "deviations": DEVIATIONS,
        "shards": per_shard,
        "aggregation_s": round(time.time() - t0, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".partial")
    json.dump(_clean(record), open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, output)
    print(headline["text"], flush=True)
    if full_expected and not all_equal:
        print("WARNING: the full run does not reproduce the manuscript counts: "
              + json.dumps({k: v for k, v in reproduction.items() if not v["equal"]}), flush=True)
    print(f"wrote {output}", flush=True)
    return record


# --------------------------------------------------------------------------------------------------- parent
def _tail(path: Path, n: int = 40) -> str:
    try:
        return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])
    except OSError:
        return ""


def run_parent(args) -> int:
    t_run = time.time()
    SC = _load_script("build_label_sidecars_v33")
    m = SC.load_corpus_manifest(args.corpus)
    num_shards = int(m["num_shards"])
    shards = SC.parse_shards(args.shards, num_shards)
    out_dir = Path(args.out_dir)
    output = Path(args.output)
    for p in (out_dir, output.parent):
        if p.resolve() == Path(args.corpus).resolve() or "match_cache" in str(p).lower() or "A2-label-family" in str(p):
            raise SystemExit(f"refusing to write into {p}")
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(args.limit_matches_per_shard or 0)
    full_expected = bool(shards == list(range(num_shards)) and not limit)
    git_commit, git_dirty = _git_state()
    run_meta = {"git_commit": git_commit, "git_dirty": git_dirty, "python": sys.executable, "preset": "v3.3",
                "corpus": {"dir": str(args.corpus), **{k: m.get(k) for k in ("run_id", "git_commit", "seed", "num_shards", "n_matches")},
                           "label": m.get("label"), "scale": m.get("scale")},
                "shard_dir": str(out_dir), "shards_requested": shards, "limit_matches_per_shard": limit or None}

    manifest_path = out_dir / "manifest.json"
    if args.aggregate_only:
        aggregate(out_dir, shards, output, m, n_boot=args.n_boot, seed=args.seed, full_expected=full_expected, run_meta=run_meta)
        return 0

    todo = []
    for i in shards:
        stem = f"tiers_shard_{i:03d}"
        existing = [out_dir / f"{stem}{s}" for s in (".npz", ".json", ".mismatch.json")]
        if any(p.exists() for p in existing):
            if args.skip_existing and (out_dir / f"{stem}.npz").exists() and (out_dir / f"{stem}.json").exists():
                info = json.load(open(out_dir / f"{stem}.json", encoding="utf-8"))
                chk = info.get("checks") or {}
                if (chk.get("identity") or {}).get("ok") and chk.get("trace_or_verdict_problems") == 0 \
                        and (info.get("limit_matches_per_shard") or 0) == limit:
                    print(f"[shard {i}] already built with the same settings; skipping", flush=True)
                    continue
            if not args.overwrite:
                raise SystemExit(f"{out_dir} already holds outputs for shard {i}; pass --overwrite or --skip-existing")
            for p in existing:
                if p.exists():
                    p.unlink()
        todo.append(i)

    env = dict(os.environ)
    env["LOL_CFG_PRESET"] = "v3.3"
    env.setdefault("LOL_OUTPUT_ROOT", "D:/LOL_Project")
    inherited = {}
    if env.get("LOL_CFG_OVERRIDES"):
        try:
            inherited = json.loads(env["LOL_CFG_OVERRIDES"])
        except json.JSONDecodeError as e:
            raise SystemExit(f"inherited LOL_CFG_OVERRIDES is not JSON: {e}")
    effective_build = m.get("effective_overrides") or {}
    conflicts = {k: (inherited[k], effective_build[k]) for k in inherited if k in effective_build and inherited[k] != effective_build[k]}
    extra_keys = sorted(set(inherited) - set(effective_build))
    if conflicts or extra_keys:
        raise SystemExit(f"LOL_CFG_OVERRIDES differs from the v3.3 build: conflicts={conflicts} extra={extra_keys}")
    env["LOL_CFG_OVERRIDES"] = json.dumps(effective_build)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["PYTHONIOENCODING"] = "utf-8"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"

    manifest = {"item": ITEM, "script": "scripts/label_tier_shares_v33.py", "run_id": time.strftime("%Y%m%d_%H%M%S"),
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), **run_meta, "effective_overrides": effective_build,
                "lol_output_root": env.get("LOL_OUTPUT_ROOT"), "parallel": args.parallel, "index_workers": args.index_workers,
                "output": str(output), "deviations": DEVIATIONS, "shards": {}, "complete": False}
    if manifest_path.exists():
        old = json.load(open(manifest_path, encoding="utf-8"))
        for k, v in (old.get("shards") or {}).items():
            if int(k) not in todo:
                manifest["shards"][k] = v

    def dump():
        json.dump(_clean(manifest), open(manifest_path, "w", encoding="utf-8"), indent=1)
    dump()

    running: dict = {}
    pending = list(todo)
    failed: list = []
    while pending or running:
        while pending and len(running) < args.parallel and not failed:
            i = pending.pop(0)
            log = open(out_dir / f"tiers_shard_{i:03d}.log", "w", encoding="utf-8")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--shard", str(i), "--corpus", str(args.corpus),
                   "--out-dir", str(out_dir), "--output", str(output), "--index-workers", str(args.index_workers)]
            if limit:
                cmd += ["--limit-matches-per-shard", str(limit)]
            running[i] = (subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"[shard {i}] started ({len(running)} running, {len(pending)} pending)", flush=True)
        time.sleep(args.poll_s)
        for i in list(running):
            proc, log, t_i = running[i]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[i]
            entry = {"rc": int(rc), "wall_s": round(time.time() - t_i, 1)}
            info_path = out_dir / f"tiers_shard_{i:03d}.json"
            if rc == 0 and info_path.exists():
                info = json.load(open(info_path, encoding="utf-8"))
                entry.update({"n_rows": info["n_rows"], "n_matches_used": info.get("n_matches_used"),
                              "identity_ok": bool(info["checks"]["identity"]["ok"]), "tier_counts": info["tier_counts"],
                              "file": f"tiers_shard_{i:03d}.npz", "timing_s": info["timing_s"], "peak_rss_mb": info.get("peak_rss_mb")})
            manifest["shards"][str(i)] = entry
            dump()
            print(f"[shard {i}] rc={rc} after {entry['wall_s']:.0f}s" + (f" rows={entry.get('n_rows')}" if rc == 0 else ""), flush=True)
            if rc != 0:
                failed.append(i)
                print("!" * 100, flush=True)
                print(f"[shard {i}] FAILED (rc={rc}{', mismatch' if rc == IDENTITY_EXIT else ''}); log tail:", flush=True)
                print(_tail(out_dir / f"tiers_shard_{i:03d}.log"), flush=True)
                print("!" * 100, flush=True)
                for j, (p, lg, _) in list(running.items()):
                    p.terminate()
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    lg.close()
                    manifest["shards"][str(j)] = {"rc": None, "terminated": True}
                running.clear()
                pending.clear()
                dump()

    ok = [i for i in shards if (manifest["shards"].get(str(i)) or {}).get("identity_ok")]
    manifest["complete"] = len(ok) == len(shards) and not failed
    manifest["full"] = bool(manifest["complete"] and full_expected)
    manifest["n_rows"] = int(sum((manifest["shards"].get(str(i)) or {}).get("n_rows", 0) for i in shards))
    manifest["finished_shards"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump()
    if failed or not manifest["complete"]:
        raise SystemExit(f"label tier run ABORTED: shard(s) {failed} failed; see {out_dir}")
    aggregate(out_dir, shards, output, m, n_boot=args.n_boot, seed=args.seed, full_expected=full_expected, run_meta=run_meta)
    manifest["aggregate"] = str(output)
    manifest["wall_clock_s"] = round(time.time() - t_run, 1)
    dump()
    print(f"done in {(time.time() - t_run) / 60:.1f} min: {len(ok)}/{len(shards)} shards, {manifest['n_rows']} rows, "
          f"full={manifest['full']}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="per-shard outputs")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="aggregate JSON")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus_shards_v33 (read only)")
    ap.add_argument("--shards", default="all", help="'all' or e.g. '0,31' / '0-7'")
    ap.add_argument("--parallel", type=int, default=4, help="worker processes (each single-threaded with --index-workers 1)")
    ap.add_argument("--index-workers", type=int, default=1, help="build_fight_index processes per shard (1 = in-process)")
    ap.add_argument("--limit-matches-per-shard", type=int, default=0, help="smoke tests: first N matches of each shard")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true", help="recompute the aggregate JSON from existing shard outputs")
    ap.add_argument("--self-test", action="store_true", help="run only the synthetic self-test (set LOL_CFG_PRESET=v3.3)")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.self_test:
        from gameplay import labels as L
        with L.cfg_override({"LABEL_TIE_POLICY": "drop"}), L.variant_cfg("market_event"):
            print(json.dumps(_clean(run_self_test()), indent=1))
        return 0
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
