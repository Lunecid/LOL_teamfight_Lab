#!/usr/bin/env python3
"""ev4_r1_record: build the DRAFT of record 1 (v4-exact plan, "기록 1") and a Korean summary.

The draft command never writes the final lock.  Its output is a draft that the author reads, corrects and signs
off; the locked record 1 is a separate file written after that sign-off by the author-only 'lock' command.  The draft carries
  "record": "1 DRAFT ...", "status": "DRAFT", "is_final_lock": false, "usable_as_record1": false
and its file name starts with ``record1_DRAFT_``.

Inputs (all read-only; nothing from 15.16, 16.x or KR is read except the plan-disclosed exposure sentence, and the
v3.3 / evr p3 summaries, whose 15.16 values are read only to prove that none of them is written out):
  * preset v4-exact (core/presets.py) and its canonical sha256; record 1A; every file in records/ with its sha256
    (decision records embedded);
  * git HEAD / dirty state; sha256 of the v4-exact code;
  * StateV3 / setup column-name hashes; Data Dragon v2 index and tables, item / objective rule files (sha256);
  * ev4_01 detect diags for 15.14 / 15.15 (yields by cohort x clean x isolated, per ok match; remakes; r2_repro);
  * ev4_02 extract manifests (rows per match, hashes) and the 15.14 extract arrays for the balance table;
  * E2 cumulative table: v3.3 legacy ('ref') and evr from the p3 summaries (15.14 / 15.15 rows only), r2_repro,
    v4 all, v4 isolated, v4 isolated + clean, by cohort;
  * balance on 15.14: standardised mean differences (clean vs non-clean isolated engagements) of pre-tau covariates
    by cohort; if max |SMD| > 0.1 the disclosure sentence is added;
  * the fit-V frozen manifest: ev4_v_models.assert_v_usable() is called and the draft is refused when V is not usable
    (``--no-fit-v`` drafts with the V block marked pending instead);
  * the q / PT candidate grids, fixed here;
  * prior-exposure disclosure, the "no 15.16 count before record 1" compliance scan, E1 disclosures and deviations.

  * the OOF V export of the fit-V run (oof_v: oof_manifest sha256 + the five fold bundle sha256) and prices_1514.json
    (sha256); either missing -> a BLOCKING pending item and "lockable": false;
  * the top-level "V_frozen_bundle_sha256" (the only key the held-out consumers read, via
    ev4_record_lock.assert_record1_locked / ev4_04_labels.check_record1_v).

Pending items: "pending_blocking" (B01 ...; V / oof_v / prices missing, sample detect run, r2_repro outside 3 %, no
SMD) make the draft not lockable; "pending_author_confirmation" (A01 ...) need the author's written resolution.

CLI
  python scripts/exact_v4/ev4_r1_record.py --detect DIR --extract DIR (--fit-v DIR | --no-fit-v)
         [--p3 FILE ...] [--scan-root DIR ...] [--out-dir DIR] [--prices FILE] [--allow-drift FILE ...]
         [--smoke] [--no-verify-chunks]
  Defaults: --detect <OUT>/stage2/detect, --extract <OUT>/stage2/extract, --p3 the p3 evr summary,
  --scan-root <OUT>/stage2, --out-dir <OUT>/records, --prices <OUT>/stage2/prices_1514.json.
  --smoke is required when any input is a sample run or the V manifest is a smoke run; a smoke draft is named
  record1_DRAFT_SMOKE_<ts>.json and may not be written into records/.  --no-verify-chunks needs --smoke.  Code
  changed since the detect / extract / fit-V outputs is refused without --smoke unless every changed file is listed
  with --allow-drift.

  python scripts/exact_v4/ev4_r1_record.py lock --draft FILE --draft-sha256 HEX --author-signoff "Name / YYYY-MM-DD"
         --resolutions FILE [--records-dir DIR] [--smoke]
  AUTHOR ONLY, never run by an agent: writes records/record1_<ts>.json (record "1", status LOCKED) from a lockable
  draft; the real lock asks for a typed confirmation at an interactive terminal.  --smoke locks a smoke draft into a
  directory other than records/ (tests only; the result is refused by assert_record1_locked).
Exit codes: 0 draft written / locked; 2 V not usable; 3 compliance / leak refusal; 1 any other refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
for _p in (str(WT), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_common as EC  # noqa: E402
import ev4_record_lock as RL  # noqa: E402
import ev4_v_models as VM  # noqa: E402

from core.presets import PRESETS  # noqa: E402
from gameplay.setup_features import SETUP_COLUMNS, SETUP_NAME_HASH  # noqa: E402
from gameplay.split_guard import SELECTION_PATCHES, normalize_patch  # noqa: E402
from gameplay.state_value_v3 import (LEVEL_DEN, STATE_V3_COLUMNS, STATE_V3_NAME_HASH, STATE_VERSION,  # noqa: E402
                                     V_ONLY_COLUMNS, name_hash, q_columns)

# ------------------------------------------------------------------ locations
PROJECT = WT.parents[1]                                         # C:/Users/todtj/문서/LOL_Teamfight
OUT_BASE = EC.OUT_BASE
RECORDS_DIR = OUT_BASE / "records"
STAGE2 = OUT_BASE / "stage2"
PLAN = PROJECT / "outputs" / "diag_survival_dbscan_20260925" / "docs" / "REESTIMATION_PLAN_V4_EXACT_20260925.md"
P3_DIR = PROJECT / "outputs" / "diag_survival_dbscan_20260925" / "p3"
P3_DEFAULT = (P3_DIR / "summary_evr_eps3927.json",)
P3_CACHE = "D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13"   # p3_detection_arms.CACHE_MAIN (all its matches)
RECORD1A = RECORDS_DIR / "record1a_boundaries_20260925T114606Z.json"
PRESET = "v4-exact"
DD2_DIR = WT / "config" / "game_rules" / "datadragon_v2"
RULE_FILES = ("config/game_rules/item_effect_flags.json", "config/game_rules/item_rules_15x.json",
              "config/game_rules/item_rules_16x.json", "config/game_rules/objective_rules.json",
              "config/game_rules/event_prices.json", "config/game_rules/map_anchors.json")
DR_RULES = WT.parents[1] / "scripts" / "engagement_labels_v3_rules.py"   # = gameplay.labels_exact.DEFAULT_DR_RULES
CODE_FILES = (                                                  # union of the ev4_01 / ev4_02 / ev4_03 code lists
    "core/presets.py", "core/config.py",
    "gameplay/exact_population.py", "gameplay/cohorts_exact.py", "gameplay/event_survival.py",
    "gameplay/respawn_rules.py", "gameplay/fights.py", "gameplay/fight_clustering.py", "gameplay/fight_postmerge.py",
    "gameplay/grid_guard.py", "gameplay/split_guard.py", "gameplay/item_state.py", "gameplay/champion_attributes.py",
    "gameplay/objective_timers.py", "gameplay/state_value_v3.py", "gameplay/state_value_v2.py",
    "gameplay/state_value.py", "gameplay/setup_features.py", "gameplay/labels_exact.py", "gameplay/role_inference.py",
    "gameplay/comparison_gates.py", "gameplay/anchors.py")

TRAIN_PATCH, SELECT_PATCH = "15.14", "15.15"
PATCHES = (TRAIN_PATCH, SELECT_PATCH)
COHORTS_V4 = ("T", "S", "ASYM", "P")
COHORTS_V33 = ("T", "S", "P")                    # v3.3 / evr scale classes (no ASYM)
R2_BASELINE_PER_MATCH = 19.6                     # RUNBOOK section 1 (R2 prototype, remakes included)
R2_TOL_REL = 0.03
DRAFT_PREFIX = "record1_DRAFT_"
FINAL_PREFIX = "record1_"                        # the locked record: record1_<ts>.json (lock command only)
DEFAULT_PRICES = STAGE2 / "prices_1514.json"     # ev4_04_labels prices (15.14 only)
PRICES_FORMAT = "ev4_prices_1"                   # = ev4_04_labels.PRICES_FORMAT
OOF_FORMAT = "ev4_oof_v_1"                       # = ev4_04_labels.OOF_FORMAT
OOF_BLOCK_FILE = "oof_v_block.json"              # = ev4_04_labels.OOF_BLOCK_FILE (ev4_03b_oof_v.py sidecar)
OOF_BLOCK_FORMAT = "ev4_oof_v_block_1"           # = ev4_04_labels.OOF_BLOCK_FORMAT
N_FOLDS = 5
LEAK_MIN = 1000                                  # held-out values below this are too common to test as substrings

# held-out patches: 15.16-15.29 (15.16 test, KR 15.18-15.22) and 16.x-19.x (external)
HELD_OUT_PATCH_RE = re.compile(r"(?<![\d.])(?:15\.(?:1[6-9]|2\d)|1[6-9]\.\d{1,2})(?![\d])")
TEXT_SUFFIXES = (".txt", ".md", ".log")

# ------------------------------------------------------------------ fixed grids (written into the record)
Q_CANDIDATE_GRIDS: Dict[str, Any] = {
    "fixed_in": "record 1 draft (R1); plan section 6 'q' and decision 9",
    "per_cohort": list(COHORTS_V4),
    "learners": {
        "logistic": {
            "C": [0.001, 0.01, 0.1, 1.0],
            "penalty": "l2", "intercept": "unpenalised",
            "standardise": "StandardScaler (mean and sd) fitted on the cohort's 15.14 training rows",
            "solver": "lbfgs", "max_iter": 4000},
        "lgbm": {
            "num_leaves": [15, 31],
            "objective": "binary", "learning_rate": 0.05, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "max_rounds": 2000, "early_stopping": True,
            "early_stopping_detail (draft default)": {
                "patience_rounds": 100,
                "stop_rows": "inner 10% of the 15.14 matches, sha256('<mid>:ev4_05:inner') mod 10 == 0; no refit"},
            "max_bin (draft default)": 255, "deterministic": True, "num_threads": 4, "seed": 20260925},
    },
    "selection": {
        "patch": SELECT_PATCH, "metric": "Brier on the cohort's 15.15 clean and isolated rows (unit row weights)",
        "between_learners": "lowest 15.15 Brier; tie rule: if Brier(logistic) - Brier(lgbm) <= 0.0005, logistic",
        "tie_tol": 0.0005, "tie_winner": "logistic",
        "within_learner (draft default)": "every grid point is trained on 15.14 and scored on 15.15; the best grid "
                                          "point per learner (15.15 Brier) enters the between-learner choice",
    },
    "inputs (draft note)": {
        "state": "StateV3 q_columns (all StateV3 columns except the V-only frame age)",
        "q_columns_n": len(q_columns()), "q_columns_name_hash": name_hash(q_columns()),
        "setup": "position group (SETUP_COLUMNS) only for clean engagements; nested groups (E8) in record 2",
    },
}
PT_CANDIDATE_GRIDS: Dict[str, Any] = {
    "PT_linear": {
        "features": ["p_pre", "time_minutes"], "scaler": "StandardScaler (mean and sd)",
        "model": "logistic, L2, C = 1.0 (legacy rr12 definition, no grid)"},
    "PT_flex": {
        "A8_fix": "B-spline bases are centred only (StandardScaler(with_mean=True, with_std=False)), not standardised; "
                  "the C grid is widened and extended when an edge is chosen",
        "features": "B-spline(p_pre; n_knots_p, degree 3, constant extrapolation, no bias column) + "
                    "B-spline(time_minutes; n_knots_t, degree 3) + their tensor product",
        "n_knots_p": [4, 6, 8], "n_knots_t": [4], "degree": 3,
        "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "edge_rule": "if the chosen C is the smallest (largest) grid value, add the next half-decade value beyond that "
                     "edge (0.0003, 0.0001, ... / 30, 100, ...) and refit; repeat until the choice is interior, at "
                     "most 3 extensions per side; every candidate's 15.15 Brier is saved",
        "selection": "per cohort on 15.15 Brier (ties: 15.15 log loss), the same rows as q",
        "legacy_for_reference": "rr12: n_knots_p {4, 6} x C {0.01, 0.1, 1}, StandardScaler(with_std=True); every "
                                "winner at C = 0.01 (grid edge); near-constant spline columns effectively unpenalised"},
    "calibration": "one calibrator rule for q and all baselines, fixed before scoring (A7): record 2",
}

PRIOR_EXPOSURE = {
    "statement_ko": "사전 노출 사실: evr 15.16 한타(T) 29,856건과 v3.3 결과는 기록 1 전에 이미 보았다.",
    "statement_en": "Prior exposure: the evr 15.16 teamfight count (T = 29,856) and the v3.3 results were seen "
                    "before record 1.",
    "source": "plan section '기록 1이 잠그는 것' (the only 15.16 number in this draft)",
}
PLAN_DISCLOSED_HELDOUT = {("evr", "15.16|T"): 29856}          # allowed in the output (plan text above): this exact
#                                                               p3 key path AND this exact value, nothing else

# ------------------------------------------------------------------ balance (E2)
SMD_THRESHOLD = 0.1
GOLD_DEN = 25000.0                          # gameplay/pipeline_cache.py DEN_TOT_G (totalGold_norm = gold / 25,000)
BALANCE_COVARIATES: Tuple[Tuple[str, str], ...] = (
    ("game_minute", "tau / 60,000"),
    ("gold_diff", "sum of blue totalGold - sum of red totalGold at tau - 1 ms (last frame <= t)"),
    ("abs_gold_diff", "|gold_diff|"),
    ("level_diff", "sum of blue levels - sum of red levels at tau - 1 ms (LEVEL_UP corrected)"),
    ("abs_level_diff", "|level_diff|"),
    ("alive_blue", "kill-event alive count, blue, at tau"),
    ("alive_red", "kill-event alive count, red, at tau"),
    ("alive_total", "alive_blue + alive_red"),
    ("kills_so_far", "blue_kills + red_kills before tau"),
    ("dragons", "elemental dragons taken so far (both teams)"),
    ("towers", "turrets destroyed so far (both teams)"),
    ("wave_phase_s", "minion-wave phase at tau: ((tau - 65,000) mod 30,000) / 1,000 s, i.e. tau mod 30 s aligned to "
                     "the wave spawns (first wave 1:05, then every 30 s); 0 = a wave has just spawned"),
    ("early_game", "1 if tau < 14:00 (840,000 ms; laning phase, turret plates still up), else 0"),
    ("wave_phase_s_early", "wave_phase_s of early-game rows only (NaN after 14:00, so excluded from its SMD)"),
)
WAVE_FIRST_SPAWN_MS = 65_000
WAVE_PERIOD_MS = 30_000
EARLY_GAME_END_MS = 840_000
# gold at tau: an event-updated gold field is used when the StateV3 columns provide one (first match, per player
# slot); otherwise the minute-frame totalGold_norm (last frame <= tau - 1 ms) is the only gold StateV3 carries.
EVENT_GOLD_FIELDS = ("totalGold_event_norm", "totalGold_at_t_norm", "gold_at_t_norm")
FRAME_GOLD_FIELD = "totalGold_norm"
DISCLOSURE_SMD_KO = ("클린 교전과 비클린 교전의 교전 전 공변량 균형은 완전하지 않다(15.14 고립 교전, 최대 |SMD| = {v:.3f}, "
                     "{cohort}의 {cov}). 클린 선택이 교전 상황과 무관하다는 가정은 근사이며, 이 차이는 클린 대 비클린 "
                     "비교(E11)와 클린 표본 결과의 해석에 반영한다.")
DISCLOSURE_SMD_EN = ("Pre-tau covariate balance between clean and non-clean engagements is not complete (15.14 isolated "
                     "engagements, max |SMD| = {v:.3f}, {cov} in {cohort}); clean selection is only approximately "
                     "independent of the engagement situation, and this is taken into account when reading the clean "
                     "versus non-clean comparison (E11) and results on the clean sample.")

COMPLIANCE_KO = ("기록 1 전 15.16 수 미열람: 이 초안을 만들 때 stage 2 산출물에 15.16(및 16.x, KR 15.18–15.22) 자료가 "
                 "없음을 확인했다. 초안에 나오는 15.16 수는 계획서가 공개한 사전 노출 수(evr T 29,856) 하나뿐이다.")


class DraftRefused(RuntimeError):
    """The draft is not written.  .code is the process exit code."""

    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# ================================================================== small helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _finite(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(WT)).replace("\\", "/")
    except ValueError:
        return str(p)


# ================================================================== provenance blocks
def preset_block(record1a: Path = RECORD1A) -> Dict[str, Any]:
    values = dict(PRESETS[PRESET])
    rec = read_json(record1a)
    locked = rec.get("locked") or {}
    bad = {k: [values.get(k), v] for k, v in locked.items() if values.get(k) != v}
    if bad or not locked:
        raise DraftRefused(f"preset {PRESET!r} disagrees with record 1A locked values: {bad or 'no locked block'}")
    presets_sha = sha256_file(WT / "core" / "presets.py")
    want = next((v for k, v in (rec.get("hashes") or {}).items()
                 if k.replace("\\", "/").endswith("core/presets.py")), None)
    return {"name": PRESET, "values": values, "values_sha256": canonical_sha256(values),
            "sha_rule": "sha256 of json.dumps(PRESETS['v4-exact'], sort_keys=True, separators=(',', ':'), "
                        "ensure_ascii=False) (= ev4_00_manifest.preset_hash / ev4_01 preset_sha256)",
            "presets_py_sha256": presets_sha, "record1a_presets_py_sha256": want,
            "presets_py_matches_record1a": want == presets_sha, "record1a_locked_agree": True}


def records_block(records_dir: Path = RECORDS_DIR) -> Dict[str, Any]:
    """sha256 of every file in records/ (drafts excluded); JSON decision records embedded."""
    files, embedded = {}, {}
    for p in sorted(Path(records_dir).rglob("*")):
        if not p.is_file() or p.name.startswith("record1_"):
            continue
        key = str(p.relative_to(records_dir)).replace("\\", "/")
        files[key] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        if p.suffix == ".json" and p.parent == Path(records_dir) and not p.name.startswith("manifest_"):
            embedded[key] = read_json(p)
    pre = EC.predecisions_info()                                  # refuses an edited pre-decision record
    return {"dir": str(records_dir), "files": files, "decision_records": embedded,
            "predecisions_pinned": {"path": pre["path"], "sha256": pre["sha256"], "pinned_in": "ev4_common.py"}}


def _latest(records: Mapping[str, Any], prefix: str) -> Tuple[Optional[str], Dict[str, Any]]:
    keys = sorted(k for k in records["decision_records"] if k.startswith(prefix))
    return (keys[-1], records["decision_records"][keys[-1]]) if keys else (None, {})


def git_block(root: Path = WT) -> Dict[str, Any]:
    def run(*a: str) -> bytes:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, check=True, timeout=60).stdout
    try:
        tracked = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no").decode("utf-8", "replace")
                   .splitlines() if ln.strip()]
        untracked = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=all", "--", "scripts/exact_v4",
                                          "tests", "gameplay", "core", "config").decode("utf-8", "replace")
                     .splitlines() if ln.startswith("??")]
        return {"head": run("rev-parse", "HEAD").decode().strip(),
                "branch": run("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
                "dirty_tracked": bool(tracked), "dirty_tracked_paths": tracked,
                "untracked_in_code_dirs": untracked,
                "diff_head_sha256": hashlib.sha256(run("diff", "HEAD", "--binary")).hexdigest()}
    except Exception as e:  # git missing
        return {"error": str(e)}


def code_block() -> Dict[str, Any]:
    out: Dict[str, Optional[str]] = {}
    for p in sorted((WT / "scripts" / "exact_v4").glob("*.py")):
        out[rel(p)] = sha256_file(p)
    for p in sorted((WT / "tests").glob("test_ev4_*.py")) + sorted((WT / "tests").glob("test_exact_*.py")):
        out[rel(p)] = sha256_file(p)
    for f in CODE_FILES:
        p = WT / f
        out[f] = sha256_file(p) if p.is_file() else None
    out["DR/engagement_labels_v3_rules.py"] = sha256_file(DR_RULES) if DR_RULES.is_file() else None
    return out


def code_drift(current: Mapping[str, Optional[str]], recorded: Mapping[str, Mapping[str, str]]) -> Dict[str, Any]:
    """For each output (detect diag, extract manifest, fit-V manifest): files whose recorded sha256 differs now."""
    out = {}
    for src, files in recorded.items():
        diff = {f: {"recorded": s, "now": current.get(f)} for f, s in (files or {}).items()
                if f in current and current.get(f) != s}
        missing = sorted(f for f in (files or {}) if f not in current)
        out[src] = {"n_files": len(files or {}), "changed": diff, "not_hashed_now": missing}
    return out


def drift_policy(drift: Mapping[str, Mapping[str, Any]], allow_drift: Sequence[str], smoke: bool) -> Dict[str, Any]:
    """Code changed since the outputs: refused for a real draft unless every changed file is listed explicitly
    (--allow-drift FILE, the code_sha256 key); a smoke draft records it.  Returns the 'code_drift_allowed' block."""
    changed = sorted({f for src in drift.values() for f in src["changed"]})
    allow = sorted({str(f).replace("\\", "/") for f in allow_drift})
    unlisted = [f for f in changed if f not in allow]
    if unlisted and not smoke:
        raise DraftRefused(f"code changed since the outputs were written ({unlisted}); a real draft refuses code "
                           "drift unless each changed file is listed with --allow-drift FILE after checking it")
    return {"listed": allow, "changed": changed, "listed_but_unchanged": [f for f in allow if f not in changed],
            "unlisted_allowed_by_smoke": unlisted if smoke else [],
            "allowed_by": "smoke" if unlisted else ("--allow-drift" if changed else None)}


def state_block(extract_manifests: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    st_now, se_now = name_hash(STATE_V3_COLUMNS), name_hash(SETUP_COLUMNS)
    if st_now != STATE_V3_NAME_HASH or se_now != SETUP_NAME_HASH:
        raise DraftRefused("StateV3 / setup column names no longer match their frozen name hashes")
    per_patch = {}
    for patch, m in extract_manifests.items():
        per_patch[patch] = {"STATE_V3_NAME_HASH": m.get("STATE_V3_NAME_HASH"), "SETUP_NAME_HASH": m.get("SETUP_NAME_HASH"),
                            "state_version": m.get("state_version")}
        if m.get("STATE_V3_NAME_HASH") != STATE_V3_NAME_HASH or m.get("SETUP_NAME_HASH") != SETUP_NAME_HASH:
            raise DraftRefused(f"extract {patch} was built with other StateV3 / setup columns")
    return {"STATE_V3_NAME_HASH": STATE_V3_NAME_HASH, "SETUP_NAME_HASH": SETUP_NAME_HASH,
            "state_version": STATE_VERSION, "n_state_columns": len(STATE_V3_COLUMNS),
            "n_setup_columns": len(SETUP_COLUMNS), "v_only_columns": sorted(V_ONLY_COLUMNS),
            "q_columns_name_hash": name_hash(q_columns()), "hash_rule": "sha256 of the newline-joined column names",
            "extract_manifests": per_patch}


def game_rules_block() -> Dict[str, Any]:
    """Data Dragon v2 index + table files and the rule files: sha256 of the bytes (no match data)."""
    idx = DD2_DIR / "index.json"
    tables = {p.name: sha256_file(p) for p in sorted(DD2_DIR.glob("*.json"))}
    rules = {f: (sha256_file(WT / f) if (WT / f).is_file() else None) for f in RULE_FILES}
    index = read_json(idx)
    versions = {k: v.get("version") for k, v in (index.get("patches") or {}).items()}
    return {"datadragon_v2_dir": rel(DD2_DIR), "index_sha256": sha256_file(idx), "index_versions": versions,
            "tables_sha256": tables, "tables_set_sha256": canonical_sha256(tables), "rule_files_sha256": rules,
            "item_effect_flags_sha256": rules.get("config/game_rules/item_effect_flags.json")}


# ================================================================== detect / extract
def _check_selection_patch(p: Any, what: str) -> str:
    pp = normalize_patch(str(p))
    if pp not in SELECTION_PATCHES:
        raise DraftRefused(f"{what}: patch {pp} is not a selection patch", code=3)
    return pp


def load_diag(detect_dir: Path, patch: str, verify_outputs: bool = True) -> Tuple[Dict[str, Any], str]:
    p = Path(detect_dir) / f"diag_{patch}.json"
    if not p.is_file():
        raise DraftRefused(f"detect diag missing: {p}")
    d = read_json(p)
    if _check_selection_patch(d.get("patch"), str(p)) != patch:
        raise DraftRefused(f"{p} is for patch {d.get('patch')}, expected {patch}")
    if (d.get("access") or {}).get("held_out"):
        raise DraftRefused(f"{p}: held-out access recorded", code=3)
    if verify_outputs:
        for name, o in (d.get("outputs") or {}).items():
            f = Path(detect_dir) / name
            if not f.is_file() or sha256_file(f) != o.get("sha256"):
                raise DraftRefused(f"detect output {f} missing or changed since the diag was written")
    return d, sha256_file(p)


def is_sample_diag(d: Mapping[str, Any]) -> bool:
    s = d.get("selection") or {}
    return not (s.get("sample") is None and s.get("limit") is None
                and s.get("n_selected") == s.get("n_list_patch"))


def detect_yields(d: Mapping[str, Any]) -> Dict[str, Any]:
    status = dict(d.get("status") or {})
    n_ok = int(status.get("ok", 0))
    if n_ok <= 0:
        raise DraftRefused(f"diag {d.get('patch')}: no ok matches")
    v4 = d["v4"]
    cxc = v4["cohort_x_clean_x_isolated"]
    cells, by_c = [], {}
    for c in COHORTS_V4:
        by_c[c] = {}
        for cl in (0, 1):
            for iso in (0, 1):
                n = int(((cxc.get(c) or {}).get(f"clean{cl}") or {}).get(f"isolated{iso}", 0))
                cells.append({"cohort": c, "clean": cl, "isolated": iso, "n": n, "per_ok_match": n / n_ok})
                by_c[c][f"clean{cl}_isolated{iso}"] = n
    tot = sum(x["n"] for x in cells)
    if tot != int(v4["n"]):
        raise DraftRefused(f"diag {d.get('patch')}: cohort x clean x isolated sums to {tot}, v4.n = {v4['n']}")
    for c in COHORTS_V4:
        if by_c[c]["clean1_isolated1"] != int((v4.get("clean_isolated_by_cohort") or {}).get(c, -1)):
            raise DraftRefused(f"diag {d.get('patch')}: clean&isolated {c} disagrees with clean_isolated_by_cohort")
    iso_by_c = {c: by_c[c]["clean0_isolated1"] + by_c[c]["clean1_isolated1"] for c in COHORTS_V4}
    ci_by_c = {c: by_c[c]["clean1_isolated1"] for c in COHORTS_V4}
    r2 = d.get("r2_repro") or {}
    sel = d.get("selection") or {}
    n_sel = int(sel.get("n_selected") or 0)
    r2_per_sel = (int(r2.get("n", 0)) / n_sel) if n_sel else None
    rk = d.get("remakes") or {}
    return {
        "selection": {k: sel.get(k) for k in ("n_list_patch", "n_selected", "sample", "limit", "seed",
                                               "match_ids_sha256", "matches_json_sha256")},
        "full_run": not is_sample_diag(d), "status": status, "n_ok": n_ok,
        "v4": {"n": int(v4["n"]), "per_ok_match": int(v4["n"]) / n_ok,
               "by_cohort": {c: int((v4.get("by_cohort") or {}).get(c, 0)) for c in COHORTS_V4},
               "isolated_by_cohort": iso_by_c, "clean_isolated_by_cohort": ci_by_c,
               "clean_isolated_per_ok_match": {c: ci_by_c[c] / n_ok for c in COHORTS_V4},
               "t5": v4.get("t5"), "t5_clean_isolated": v4.get("t5_clean_isolated"),
               "cells": cells},
        "remakes": {k: rk.get(k) for k in ("threshold_ms", "rule", "n_selected_before_exclusion", "n_excluded",
                                           "n_excluded_list_stage", "n_excluded_event_stage",
                                           "n_duration_vs_game_end_mismatch")}
                   | {"match_ids": sorted([x.get("match_id") for x in (rk.get("excluded_list_stage") or [])]
                                          + [x.get("match_id") if isinstance(x, Mapping) else x
                                             for x in (rk.get("excluded_event_stage") or [])])},
        "r2_repro": {"n": int(r2.get("n", 0)),
                     "by_cohort": {c: int((r2.get("by_cohort") or {}).get(c, 0)) for c in COHORTS_V4},
                     "per_ok_match": _finite(r2.get("per_match")), "per_selected_match": r2_per_sel,
                     "baseline_per_match": R2_BASELINE_PER_MATCH,
                     "rel_diff_per_selected": (r2_per_sel / R2_BASELINE_PER_MATCH - 1) if r2_per_sel else None,
                     "within_3pct": (abs(r2_per_sel / R2_BASELINE_PER_MATCH - 1) <= R2_TOL_REL) if r2_per_sel else None,
                     "baseline_note": "R2 baseline predates the remake rule; compare per selected match (RUNBOOK 1)"},
        "params": (d.get("params") or {}), "cache_dir": d.get("cache_dir"),
        "preset": d.get("preset"), "code_git": (d.get("code") or {}).get("git"),
        "predecisions": d.get("predecisions"),
    }


def load_extract_manifest(extract_dir: Path, patch: str) -> Tuple[Dict[str, Any], str]:
    p = Path(extract_dir) / patch / "manifest.json"
    if not p.is_file():
        raise DraftRefused(f"extract manifest missing: {p}")
    m = read_json(p)
    if _check_selection_patch(m.get("patch"), str(p)) != patch:
        raise DraftRefused(f"{p} is for patch {m.get('patch')}, expected {patch}")
    return m, sha256_file(p)


def extract_summary(m: Mapping[str, Any], sha: str) -> Dict[str, Any]:
    keep = ("patch", "sample", "sample_limit", "n_matches", "match_ids_sha256", "n_chunks", "chunk_size", "rows",
            "rows_per_match", "match_status", "item_check", "h_distribution", "h_source", "STATE_V3_NAME_HASH",
            "SETUP_NAME_HASH", "state_version", "git_head", "plan_hash", "bytes_total")
    out = {k: m.get(k) for k in keep}
    out["manifest_sha256"] = sha
    out["remakes"] = {k: (m.get("remakes") or {}).get(k) for k in ("threshold_ms", "n_excluded", "match_ids")}
    sc = m.get("spot_check") or {}
    out["spot_check"] = {k: sc.get(k) for k in ("ran", "n", "failures", "n_clean_checked")}
    out["match_list"] = m.get("match_list")
    return out


# ================================================================== p3 (v3.3 legacy / evr)
class HeldOutValues:
    """15.16 (and pooled) numbers from the p3 summaries.  Never serialised; used only by leak_check."""
    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: List[Tuple[str, int]] = []

    def add(self, path: str, v: Any) -> None:
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            self._items.append((path, int(v)))

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:                 # never show values
        return f"<HeldOutValues n={len(self._items)}>"


def read_p3(paths: Sequence[Path], arms: Sequence[str] = ("ref", "evr")) -> Tuple[Dict[str, Any], HeldOutValues]:
    """15.14 / 15.15 per-patch rows of the p3 summaries.  Any other number (15.16 rows, pooled counts, n_listed ...)
    goes into HeldOutValues and nowhere else."""
    held = HeldOutValues()
    rows: Dict[str, Dict[str, Dict[str, int]]] = {}
    files = []
    for path in paths:
        blob = read_json(path)
        files.append({"path": str(path), "sha256": sha256_file(Path(path))})

        def walk(v: Any, keys: List[str]) -> None:
            if isinstance(v, Mapping):
                for k, x in v.items():
                    walk(x, keys + [str(k)])
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    walk(x, keys + [str(i)])
            else:
                held.add("/".join(keys), v)

        for top, v in blob.items():
            if top != "per_patch":
                walk(v, [top])
        for arm, d in (blob.get("per_patch") or {}).items():
            for k, v in (d or {}).items():
                patch, _, cohort = str(k).partition("|")
                if patch in PATCHES and arm in arms:
                    prev = rows.setdefault(arm, {}).setdefault(patch, {}).get(cohort)
                    if prev is not None and prev != int(v):
                        raise DraftRefused(f"p3 summaries disagree on {arm} {patch}|{cohort}")
                    rows[arm][patch][cohort] = int(v)
                elif patch not in PATCHES:
                    held.add(f"per_patch/{arm}/{k}", v)
    missing = [a for a in arms if a not in rows or any(p not in rows[a] for p in PATCHES)]
    if missing:
        raise DraftRefused(f"p3 summaries lack 15.14 / 15.15 rows for arms {missing}")
    return {"files": files, "arms": {"ref": "v3.3 legacy detector (frame survival, presence gate)",
                                      "evr": "event survival + respawn-aware 5-s grid (evr arm)"},
            "rows": rows, "note": "15.14 / 15.15 rows only; the files also hold 15.16 and pooled counts, which "
                                  "are not copied"}, held


def leak_check(texts: Iterable[str], held: HeldOutValues, min_value: int = LEAK_MIN) -> Dict[str, Any]:
    """Refuse when a held-out p3 number (>= min_value) appears as a standalone number in any output text.  The only
    exemption is the plan-disclosed pair: key path 'per_patch/evr/15.16|T' WITH value 29,856 (the same path holding
    another value, or another path holding 29,856, is checked like any other).  The error names the key path only,
    never the value."""
    blob = "\n".join(texts)
    allowed = {("per_patch/" + a + "/" + k, int(v)) for (a, k), v in PLAN_DISCLOSED_HELDOUT.items()}
    hits, n_checked, n_exempt = [], 0, 0
    for path, v in held:
        if (path, v) in allowed:
            n_exempt += 1
            continue
        if v < min_value:
            continue
        n_checked += 1
        for form in (str(v), f"{v:,}"):
            if re.search(rf"(?<![\w.,]){re.escape(form)}(?![\w]|,\d)", blob):
                hits.append(path)
                break
    if hits:
        raise DraftRefused(f"held-out p3 values would be written out (key paths: {sorted(set(hits))})", code=3)
    return {"n_held_out_values_checked": n_checked, "min_value": min_value, "hits": 0,
            "allowed": sorted(p for p, _v in allowed), "allowed_rule": "exact key path and exact value",
            "n_exempt": n_exempt}


# ================================================================== E2 cumulative table
def _row(stage: str, source: str, counts: Mapping[str, Optional[int]], basis: Optional[int], basis_note: str,
         cohorts: Sequence[str]) -> Dict[str, Any]:
    total = sum(int(counts[c]) for c in cohorts if counts.get(c) is not None)
    return {"stage": stage, "source": source,
            "by_cohort": {c: (int(counts[c]) if counts.get(c) is not None else None) for c in COHORTS_V4},
            "total": total, "n_matches_basis": basis, "basis_note": basis_note,
            "per_match_total": (total / basis) if basis else None,
            "per_match_by_cohort": {c: (int(counts[c]) / basis if (basis and counts.get(c) is not None) else None)
                                    for c in COHORTS_V4}}


def _same_dir(a: Any, b: Any) -> bool:
    def norm(x: Any) -> str:
        return str(x or "").replace("\\", "/").rstrip("/").lower()
    return norm(a) == norm(b)


def e2_table(yields: Mapping[str, Mapping[str, Any]], p3: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for patch in PATCHES:
        y = yields[patch]
        n_list = y["selection"].get("n_list_patch")
        same_cache = _same_dir(y.get("cache_dir"), P3_CACHE)
        rows = [
            _row("v3.3_legacy", "p3 ref", p3["rows"]["ref"][patch], n_list,
                 "n_list_patch of the detect run: p3 ran over every match of the cache "
                 f"{P3_CACHE} (remakes included); detect cache is the same: {same_cache}", COHORTS_V33),
            _row("evr", "p3 evr", p3["rows"]["evr"][patch], n_list, "as v3.3_legacy", COHORTS_V33),
            _row("r2_repro", "detect diag r2_repro", y["r2_repro"]["by_cohort"], y["n_ok"],
                 "ok matches of the detect run (remakes excluded)", COHORTS_V4),
            _row("v4_all", "detect diag v4", y["v4"]["by_cohort"], y["n_ok"], "ok matches", COHORTS_V4),
            _row("v4_isolated", "detect diag v4 (isolated)", y["v4"]["isolated_by_cohort"], y["n_ok"], "ok matches",
                 COHORTS_V4),
            _row("v4_isolated_clean", "detect diag v4 (isolated and clean)", y["v4"]["clean_isolated_by_cohort"],
                 y["n_ok"], "ok matches", COHORTS_V4),
        ]
        for prev, cur in zip(rows, rows[1:]):
            cur["delta_total_vs_previous"] = cur["total"] - prev["total"]
            if prev["per_match_total"] is not None and cur["per_match_total"] is not None:
                cur["delta_per_match_vs_previous"] = cur["per_match_total"] - prev["per_match_total"]
        out[patch] = {"rows": rows, "v4_is_sample": not y["full_run"], "legacy_denominator_same_cache": same_cache,
                      "counts_comparable": bool(y["full_run"]),
                      "note": ("v3.3 / evr cohorts are presence-based scale classes (T / S / P, no ASYM); v4 cohorts "
                               "are kill-credit counts (T / S / ASYM / P). Cohort columns are not the same definition "
                               "across these rows; totals and per-match rates are the comparable quantities.")
                              + ("" if y["full_run"] else " The detect run is a SAMPLE: its counts cover "
                                 f"{y['selection'].get('n_selected')} matches, the p3 rows the whole patch; compare "
                                 "per-match rates only.")}
    return out


# ================================================================== balance (15.14)
def smd(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """(mean_a - mean_b) / sqrt((var_a + var_b) / 2), ddof = 1.  None when a group has < 2 finite values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = a.mean(), b.mean()
    s = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if s == 0.0:
        return 0.0 if ma == mb else math.copysign(math.inf, ma - mb)
    return float((ma - mb) / s)


SLOT_BLUE = ("blue_top_", "blue_jungle_", "blue_middle_", "blue_bottom_", "blue_utility_")
SLOT_RED = tuple(p.replace("blue_", "red_") for p in SLOT_BLUE)


def gold_field(names: Sequence[str]) -> Dict[str, Any]:
    """Which StateV3 gold field the balance table uses for gold at tau (documented in the record)."""
    nm = set(names)
    for f in EVENT_GOLD_FIELDS:
        if all(p + f in nm for p in SLOT_BLUE + SLOT_RED):
            return {"field": f, "source": "event-updated gold at tau - 1 ms (StateV3 column)", "event_updated": True,
                    "den": GOLD_DEN}
    if not all(p + FRAME_GOLD_FIELD in nm for p in SLOT_BLUE + SLOT_RED):
        raise DraftRefused(f"StateV3 has no per-player gold field ({FRAME_GOLD_FIELD})")
    return {"field": FRAME_GOLD_FIELD, "event_updated": False, "den": GOLD_DEN,
            "source": "minute frame: '<side>_<role>_totalGold_norm' x 25,000 from the last frame <= tau - 1 ms. StateV3 "
                      f"has no event-updated gold column (checked: {list(EVENT_GOLD_FIELDS)}), so gold at tau is not "
                      "available from events in the state; kills / levels (LEVEL_UP) / objectives are event values "
                      "at tau - 1 ms, gold is the frame value (non-clean rows carry older gold by construction)"}


def _col_index(names: Sequence[str]) -> Dict[str, Any]:
    ix = {n: i for i, n in enumerate(names)}
    blue, red = list(SLOT_BLUE), list(SLOT_RED)
    gf = gold_field(names)["field"]
    towers = ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET", "OTHER")
    need = {
        "gold_b": [ix[p + gf] for p in blue], "gold_r": [ix[p + gf] for p in red],
        "lvl_b": [ix[p + "level_norm"] for p in blue], "lvl_r": [ix[p + "level_norm"] for p in red],
        "kills": [ix["blue_kills"], ix["red_kills"]], "dragons": [ix["blue_dragons"], ix["red_dragons"]],
        "towers": [ix[f"{s}_tower_{t}"] for s in ("blue", "red") for t in towers],
    }
    return need


def covariates_from_rows(eng: "Any", X: np.ndarray, names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Pre-tau covariates for engagement rows (eng: DataFrame with tau / alive_blue / alive_red; X: StateV3 at
    tau - 1 ms, rows aligned with eng)."""
    k = _col_index(names)
    X = np.asarray(X, dtype=np.float64)
    gold = (X[:, k["gold_b"]].sum(1) - X[:, k["gold_r"]].sum(1)) * GOLD_DEN
    lvl = (X[:, k["lvl_b"]].sum(1) - X[:, k["lvl_r"]].sum(1)) * LEVEL_DEN
    ab, ar = eng["alive_blue"].to_numpy(float), eng["alive_red"].to_numpy(float)
    tau = eng["tau"].to_numpy(np.int64)
    phase = np.mod(tau - WAVE_FIRST_SPAWN_MS, WAVE_PERIOD_MS) / 1000.0
    early = (tau < EARLY_GAME_END_MS).astype(float)
    return {"game_minute": tau.astype(float) / 60000.0, "gold_diff": gold, "abs_gold_diff": np.abs(gold),
            "level_diff": lvl, "abs_level_diff": np.abs(lvl), "alive_blue": ab, "alive_red": ar,
            "alive_total": ab + ar, "kills_so_far": X[:, k["kills"]].sum(1), "dragons": X[:, k["dragons"]].sum(1),
            "towers": X[:, k["towers"]].sum(1), "wave_phase_s": phase, "early_game": early,
            "wave_phase_s_early": np.where(early == 1.0, phase, np.nan)}


def load_balance_rows(extract_dir: Path, manifest: Mapping[str, Any], verify_chunks: bool = True):
    """(eng DataFrame, covariate dict) for all 15.14 extract engagement rows (isolated by construction)."""
    import pandas as pd
    d = Path(extract_dir) / TRAIN_PATCH
    if normalize_patch(str(manifest.get("patch"))) != TRAIN_PATCH:
        raise DraftRefused("balance: the extract manifest is not 15.14")
    engs, covs = [], []
    chunks = manifest.get("chunks") or {}
    if not chunks:
        raise DraftRefused("balance: the 15.14 extract manifest lists no chunks")
    for cid in sorted(chunks):
        info = chunks[cid]
        npz, pq = d / f"chunk_{cid}.npz", d / f"chunk_{cid}_eng.parquet"
        if verify_chunks:
            for f in (npz, pq):
                want = ((info.get("files") or {}).get(f.name) or {}).get("sha256")
                if want is None or not f.is_file() or sha256_file(f) != want:
                    raise DraftRefused(f"balance: {f} missing or changed since the extract manifest was written")
        eng = pd.read_parquet(pq, columns=["row", "match_id", "tau", "cohort", "clean", "isolated", "alive_blue",
                                           "alive_red", "frame_age_ms"])
        with np.load(npz, allow_pickle=False) as z:
            names = [str(x) for x in z["state_columns"]]
            if name_hash(names) != STATE_V3_NAME_HASH:
                raise DraftRefused(f"balance: {npz} has other StateV3 columns")
            X = z["eng_X"]
            if len(eng) != X.shape[0] or not np.array_equal(eng["row"].to_numpy(), np.arange(len(eng))):
                raise DraftRefused(f"balance: {pq} rows do not align with {npz} eng_X")
            covs.append(covariates_from_rows(eng, X, names))
        engs.append(eng)
    eng = pd.concat(engs, ignore_index=True)
    cov = {k: np.concatenate([c[k] for c in covs]) for k in covs[0]}
    if (eng["isolated"] != 1).any():
        raise DraftRefused("balance: non-isolated rows in the 15.14 extract")
    return eng, cov


def _nanmean(a: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def balance_table(eng: "Any", cov: Mapping[str, np.ndarray], threshold: float = SMD_THRESHOLD,
                  gold_info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    clean = eng["clean"].to_numpy().astype(int) == 1
    coh = eng["cohort"].astype(str).to_numpy()
    groups = {c: coh == c for c in COHORTS_V4}
    groups["all"] = np.ones(len(eng), dtype=bool)
    table, best = {}, (None, None, None)
    for g, m in groups.items():
        rows = {}
        for name, _desc in BALANCE_COVARIATES:
            v = cov[name]
            a, b = v[m & clean], v[m & ~clean]
            s = smd(a, b)
            rows[name] = {"smd": s, "mean_clean": _nanmean(a), "mean_nonclean": _nanmean(b)}
            if s is not None and (best[0] is None or abs(s) > abs(best[0])):
                best = (s, g, name)
        table[g] = {"n_clean": int((m & clean).sum()), "n_nonclean": int((m & ~clean).sum()), "covariates": rows,
                    "max_abs_smd": max((abs(r["smd"]) for r in rows.values() if r["smd"] is not None), default=None)}
    max_abs = abs(best[0]) if best[0] is not None else None
    flag = max_abs is not None and max_abs > threshold
    return {
        "patch": TRAIN_PATCH, "population": "15.14 isolated engagements of the extract (remakes excluded)",
        "groups": "clean (tau frame age < 10 s) vs non-clean",
        "covariates": [{"name": n, "definition": d} for n, d in BALANCE_COVARIATES],
        "smd_definition": "(mean_clean - mean_nonclean) / sqrt((var_clean + var_nonclean) / 2), ddof = 1, unweighted, "
                          "rows as units (descriptive)",
        "by_cohort": table,
        "max_abs_smd": max_abs, "max_at": {"cohort": best[1], "covariate": best[2], "smd": best[0]},
        "rule": f"if max |SMD| over cohorts (T, S, ASYM, P, all) and covariates > {threshold}, the disclosure "
                "sentence is added (plan E2 balance; risk 4, wave-phase selection of clean engagements)",
        "threshold": threshold, "rule_triggered": bool(flag),
        "disclosure_ko": DISCLOSURE_SMD_KO.format(v=max_abs, cohort=best[1], cov=best[2]) if flag else None,
        "disclosure_en": DISCLOSURE_SMD_EN.format(v=max_abs, cohort=best[1], cov=best[2]) if flag else None,
        "caveat": "gold / level / kills / objectives are StateV3 at tau - 1 ms; gold and CS come from the last minute "
                  "frame, so non-clean rows carry older gold values by construction",
        "gold_field": dict(gold_info) if gold_info is not None else None,
        "wave_phase": {"first_spawn_ms": WAVE_FIRST_SPAWN_MS, "period_ms": WAVE_PERIOD_MS,
                       "early_game_end_ms": EARLY_GAME_END_MS,
                       "note": "risk 4 (wave-phase selection of clean engagements): the phase is a linear value on a "
                               "30-s circle, so its SMD is a first check only"},
    }


# ================================================================== V (fit-V frozen manifest)
def v_block(fit_v: Optional[Path], allow_smoke: bool, extract_shas: Mapping[str, str]) -> Dict[str, Any]:
    grids = {"fixed_in": "ev4_v_models.GRIDS (before record 1)", "grids": VM.GRIDS,
             "selection_rule": f"lowest raw V_SELECT (15.15) log loss; within {VM.SELECT_TOL} the simpler of "
                               f"{list(VM.SIMPLICITY_ORDER)} wins"}
    if fit_v is None:
        return {"status": "PENDING: fit-V not given (--no-fit-v); record 1 cannot be locked without it",
                "usable": None, "candidate_grids": grids}
    p = Path(fit_v)
    mpath = p / "frozen_manifest.json" if p.is_dir() else p
    try:
        vf = VM.assert_v_usable(mpath, allow_smoke=allow_smoke)
    except VM.VNotUsable as e:
        raise DraftRefused(f"V is not usable, draft refused: {e}", code=2) from e
    m = read_json(mpath)
    fdir = mpath.parent
    bad = [k for k, s in (m.get("files_sha256") or {}).items()
           if not (fdir / k).is_file() or sha256_file(fdir / k) != s]
    if bad:
        raise DraftRefused(f"fit-V files missing or changed since the frozen manifest: {bad}")
    bundle = fdir / "V_frozen" / "bundle.json"
    if not bundle.is_file() or sha256_file(bundle) != vf.get("bundle_sha256"):
        raise DraftRefused("V_frozen/bundle.json does not match the manifest bundle_sha256")
    try:                                     # bundle format 1 (first fit) or 2 (side marker, optional recalibration)
        structure = VM.read_bundle_structure(bundle.parent, vf.get("bundle_sha256"))
    except RuntimeError as e:
        raise DraftRefused(f"V_frozen bundle: {e}") from e
    for flag in ("side_marker", "recalibrated"):
        if flag in vf and bool(vf[flag]) != bool(structure[flag]):
            raise DraftRefused(f"frozen manifest V_frozen.{flag} = {vf[flag]}, V_frozen/bundle.json has "
                               f"{structure[flag]}")
    if m.get("state_v3_name_hash") != STATE_V3_NAME_HASH:
        raise DraftRefused("fit-V was run on other StateV3 columns")
    pre = ((m.get("decisions") or {}).get("predecisions_record") or {}).get("sha256")
    if pre != EC.PREDECISIONS_SHA256:
        raise DraftRefused("fit-V manifest does not carry the pinned pre-decision record")
    for role, patch in (("train", TRAIN_PATCH), ("select", SELECT_PATCH)):
        got = ((m.get("inputs") or {}).get(role) or {}).get("manifest_sha256")
        if got != extract_shas.get(patch):
            raise DraftRefused(f"fit-V {role} input is not the {patch} extract given here "
                               f"(manifest sha {got} vs {extract_shas.get(patch)})")
    rep_path = fdir / "report_e4.json"
    rep = read_json(rep_path) if rep_path.is_file() else {}
    chosen = m.get("chosen")
    mart = (rep.get("martingale") or {}).get(chosen) if chosen else None
    return {
        "status": "usable", "usable": True, "smoke": bool(m.get("smoke")), "pilot": bool(m.get("pilot")),
        "frozen_manifest": {"path": str(mpath), "sha256": sha256_file(mpath)},
        "report_e4": {"path": str(rep_path), "sha256": sha256_file(rep_path) if rep_path.is_file() else None},
        "chosen": chosen, "selection": m.get("selection"), "stop_rule": m.get("stop_rule"),
        "V_frozen": vf, "candidates_bundle_sha256": m.get("candidates"), "grids": m.get("grids"),
        "budget": m.get("budget"), "inputs": m.get("inputs"), "input_checks": m.get("input_checks"),
        "code_sha256": m.get("code_sha256"), "git": m.get("git"),
        "census": rep.get("census"), "reload_max_abs_diff": rep.get("reload_max_abs_diff"),
        "final_fits": rep.get("final_fits"), "martingale_chosen": mart,
        "V_structure": {**structure, "variant": vf.get("variant")},
        "recalibrated": bool(structure["recalibrated"]),
        "v_revision": m.get("v_revision"),
        "recalibration_report": ((rep.get("v_revision") or {}).get("recalibration")),
        "candidate_grids": grids,
        "warnings": ([f"fit-V git dirty paths: {m['git'].get('dirty_paths')}"]
                     if (m.get("git") or {}).get("dirty_paths") else []),
    }


# ================================================================== OOF V export and 15.14 prices (needed by R6)
def _pending(status: str) -> Dict[str, Any]:
    return {"status": "PENDING: " + status, "ok": False}


def oof_block(fit_v: Optional[Path], v: Mapping[str, Any], allow_smoke: bool) -> Dict[str, Any]:
    """The frozen manifest's 'oof_v' export (ev4_04_labels.OOF_SPEC): oof_manifest.json sha256 and the five fold
    bundle sha256 (each fold_k/bundle.json re-hashed).  Missing -> a PENDING block (the draft is then not lockable);
    present but inconsistent -> refusal."""
    if fit_v is None or not v.get("usable"):
        return _pending("fit-V not given; the OOF V export (oof_v) cannot be recorded")
    p = Path(fit_v)
    mpath = p / "frozen_manifest.json" if p.is_dir() else p
    blk = read_json(mpath).get("oof_v")
    source = "frozen_manifest"
    side = mpath.parent / OOF_BLOCK_FILE
    if not isinstance(blk, Mapping) and side.is_file():         # ev4_03b_oof_v.py sidecar (frozen manifest untouched)
        sb = read_json(side)
        rel = str(sb.get("dir_rel") or "")
        if sb.get("format") != OOF_BLOCK_FORMAT or sb.get("frozen_manifest_sha256") != sha256_file(mpath) or \
                not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise DraftRefused(f"oof_v: {side} is not an {OOF_BLOCK_FORMAT} sidecar of this frozen manifest")
        blk = {**sb, "dir": str(mpath.parent / rel)}
        source = OOF_BLOCK_FILE
    if not isinstance(blk, Mapping) or not blk.get("dir") or not blk.get("manifest_sha256"):
        return _pending("the fit-V frozen manifest has no 'oof_v' block and there is no oof_v_block.json next to it "
                        "(run scripts/exact_v4/ev4_03b_oof_v.py; 15.14 training labels need out-of-fold V)")
    d = Path(blk["dir"])
    om_path = d / "oof_manifest.json"
    if not om_path.is_file() or sha256_file(om_path) != blk["manifest_sha256"]:
        raise DraftRefused(f"oof_v: {om_path} missing or not the manifest_sha256 of the frozen manifest")
    om = read_json(om_path)
    vb = (v.get("V_frozen") or {}).get("bundle_sha256")
    problems = []
    if om.get("format") != OOF_FORMAT:
        problems.append(f"format {om.get('format')!r}")
    if om.get("train_patch") != TRAIN_PATCH:
        problems.append(f"train_patch {om.get('train_patch')!r}")
    if om.get("frozen_bundle_sha256") != vb:
        problems.append("frozen_bundle_sha256 is not this V_frozen")
    if om.get("kind") != (v.get("V_frozen") or {}).get("kind"):
        problems.append("kind differs from the chosen V")
    if om.get("pilot"):
        problems.append("pilot OOF set")
    if om.get("smoke") and not allow_smoke:
        problems.append("smoke OOF set without --smoke")
    folds = om.get("folds") or {}
    if sorted(folds) != [str(k) for k in range(N_FOLDS)]:
        problems.append(f"folds {sorted(folds)}")
    fold_sha = {}
    for k in range(N_FOLDS):
        f = folds.get(str(k)) or {}
        bj = d / str(f.get("dir", f"fold_{k}")) / "bundle.json"
        if not bj.is_file() or sha256_file(bj) != f.get("bundle_sha256"):
            problems.append(f"fold {k} bundle.json missing or not its bundle_sha256")
        tf = d / str(f.get("train_match_ids_file", ""))
        if not tf.is_file() or sha256_file(tf) != f.get("train_match_ids_sha256"):
            problems.append(f"fold {k} train_match_ids file missing or changed")
        fold_sha[str(k)] = f.get("bundle_sha256")
    if problems:
        raise DraftRefused(f"oof_v export does not satisfy ev4_04_labels.OOF_SPEC: {problems}")
    return {"status": "ok", "ok": True, "dir": str(d), "manifest_sha256": blk["manifest_sha256"],
            "kind": om.get("kind"), "hyperparameter": om.get("hyperparameter"), "fold_rule": om.get("fold_rule"),
            "fold_bundle_sha256": fold_sha, "smoke": bool(om.get("smoke")), "block_source": source,
            **({"sidecar_sha256": sha256_file(side)} if source == OOF_BLOCK_FILE else {})}


def prices_block(prices: Optional[Path], allow_smoke: bool) -> Dict[str, Any]:
    """prices_1514.json (ev4_04_labels prices): path, sha256, n_matches and the table.  Missing -> PENDING."""
    if prices is None or not Path(prices).is_file():
        return _pending(f"prices_1514.json not found ({prices}); run ev4_04_labels.py prices on the full 15.14 list")
    p = Path(prices)
    b = read_json(p)
    if b.get("format") != PRICES_FORMAT or b.get("patch") != TRAIN_PATCH or \
            list(b.get("source_patches") or []) != [TRAIN_PATCH]:
        raise DraftRefused(f"{p} is not an {PRICES_FORMAT} table estimated on {TRAIN_PATCH} only")
    if b.get("sample") and not allow_smoke:
        raise DraftRefused(f"{p} is a sample price fit; only --smoke may use it")
    return {"status": "ok", "ok": True, "path": str(p), "sha256": sha256_file(p), "sample": bool(b.get("sample")),
            "n_matches": (b.get("fit") or {}).get("n_matches"), "table": b.get("table"),
            "matches_source_sha256": (b.get("matches") or {}).get("source_sha256")}


# ================================================================== compliance scan
def _walk_patch_keys(v: Any, path: str, bad: List[str]) -> None:
    if isinstance(v, Mapping):
        for k, x in v.items():
            kp = f"{path}/{k}"
            if k == "patch" and isinstance(x, (str, int)) and x not in ("", None):
                try:
                    if normalize_patch(str(x)) not in SELECTION_PATCHES:
                        bad.append(f"{kp} is not a selection patch")
                except Exception:
                    pass
            elif k == "held_out" and x is True:
                bad.append(f"{kp} = true")
            elif k in ("record1", "record1_sha256") and x not in (None, ""):
                bad.append(f"{kp} set (a held-out patch was opened)")
            _walk_patch_keys(x, kp, bad)
    elif isinstance(v, list):
        for i, x in enumerate(v[:2000]):
            _walk_patch_keys(x, f"{path}/{i}", bad)


def compliance_scan(roots: Sequence[Path], json_max_bytes: int = 20_000_000) -> Dict[str, Any]:
    """Proof that the stage-2 outputs hold no held-out patch data: file / directory names, JSON 'patch' / 'held_out' /
    'record1' fields, and the 'patch' column of every parquet file.  Text files (logs, RUNBOOK) are only listed when
    they mention a held-out patch string (refusal tests and plans mention them); they are not a violation."""
    import pyarrow.parquet as pq
    violations, mentions, n_files, n_parquet, n_json = [], [], 0, 0, 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            violations.append(f"scan root missing: {root}")
            continue
        for p in sorted(root.rglob("*")):
            relp = str(p.relative_to(root)).replace("\\", "/")
            if HELD_OUT_PATCH_RE.search(relp):
                violations.append(f"{root.name}/{relp}: held-out patch in the path")
            if not p.is_file():
                continue
            n_files += 1
            suf = p.suffix.lower()
            if suf == ".parquet":
                n_parquet += 1
                try:
                    schema = pq.read_schema(p)
                    if "patch" in schema.names:
                        vals = set(pq.read_table(p, columns=["patch"]).column("patch").unique().to_pylist())
                        badv = sorted(str(v) for v in vals if v is not None
                                      and normalize_patch(str(v)) not in SELECTION_PATCHES)
                        if badv:
                            violations.append(f"{root.name}/{relp}: patch column holds a held-out patch")
                except Exception as e:  # unreadable parquet is a finding, not a pass
                    violations.append(f"{root.name}/{relp}: parquet unreadable ({type(e).__name__})")
            elif suf == ".json" and p.stat().st_size <= json_max_bytes:
                n_json += 1
                try:
                    blob = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                bad: List[str] = []
                _walk_patch_keys(blob, "", bad)
                violations.extend(f"{root.name}/{relp}:{b}" for b in bad)
            elif suf in TEXT_SUFFIXES:
                try:
                    n = sum(1 for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
                            if HELD_OUT_PATCH_RE.search(ln))
                except OSError:
                    n = 0
                if n:
                    mentions.append({"file": f"{root.name}/{relp}", "lines": n})
    return {"roots": [str(r) for r in roots], "n_files": n_files, "n_parquet": n_parquet, "n_json": n_json,
            "violations": violations, "text_mentions (informational)": mentions, "ok": not violations,
            "statement_ko": COMPLIANCE_KO if not violations else None}


# ================================================================== disclosures / deviations
def disclosures_block(records: Mapping[str, Any], record1a: Mapping[str, Any]) -> Dict[str, Any]:
    f = records["files"]

    def ref(prefix: str) -> Dict[str, Any]:
        k, _ = _latest(records, prefix)
        return {"record": k, "sha256": f[k]["sha256"] if k else None}

    k_pre, pre = _latest(records, "stage2_predecisions_")
    k_vr, vr = _latest(records, "v_revision_prespec_")
    k_e1, e1 = _latest(records, "e1_decision_")
    k_ps, ps = _latest(records, "e1_prespec_")
    k_st, st = _latest(records, "items_strict_rule_")
    k_cf, cf = _latest(records, "items_confirmations_")
    k_s1, s1 = _latest(records, "stage1_decisions_")
    k_iv, iv = _latest(records, "items_validation_decisions_")
    td = pre.get("technical_defaults") or {}
    e1_disc = [{"text": t, "source": "record 1A disclosures"} for t in (record1a.get("disclosures") or [])]
    for d in e1_disc:
        if "equal-weight" in d["text"]:
            d["detail"] = {**ref("e1_prespec_"), **(ps.get("disclosure") or {})}
    e1_disc += [
        {"text": "the 2-component mixture on log kill gaps is misspecified (converged crossing about 32 s); not used "
                 "as a cross-check for G and not reported in the thesis", "source": {"record": "record 1A evidence"}},
        {"text": "the participant-sharing signal is an operational definition of 'same engagement'",
         "source": {"record": "plan section 1"}},
    ] + [{"text": t, "source": ref("e1_decision_")} for t in (e1.get("reporting_notes") or [])]
    dev = [
        ({"id": "DEV-side-marker", "plan": "section 4: keep the blue / red marker",
          "done": vr.get("revision_1_side_marker"), "status": "reversed: the side marker is added (as in the plan)",
          "superseded": {"done": pre.get("side_marker"), **ref("stage2_predecisions_")}, **ref("v_revision_prespec_")}
         if k_vr else
         {"id": "DEV-side-marker", "plan": "section 4: keep the blue / red marker",
          "done": pre.get("side_marker"), **ref("stage2_predecisions_")}),
    ] + ([{"id": "DEV-V-conditional-recalibration", "plan": "(not in the plan)",
           "done": vr.get("revision_2_conditional_recalibration"),
           "note": "whether it was triggered / adopted is recorded in V.V_structure.recalibrated and V.v_revision",
           **ref("v_revision_prespec_")}] if k_vr else []) + [
        {"id": "DEV-flip-copies", "plan": "section 4: add team-flipped copies",
         "done": "in-place 50% team swap with target flip (memory); the MLP swaps 50% of each minibatch",
         **ref("stage2_predecisions_")},
        {"id": "DEV-E1-sequential", "plan": "section 1: joint (G, D) argmin of the weighted disagreement",
         "done": (e1.get("deviation_from_prespecification") or {}).get("fallback"),
         "why": (e1.get("deviation_from_prespecification") or {}).get("what_happened"), **ref("e1_decision_")},
        {"id": "DEV-items-strict-exception", "plan": "test 9: strict inventory agreement",
         "done": st.get("inventory_99pct_measure"), **ref("items_strict_rule_")},
        {"id": "DEV-R15-5-removed", "plan": "15.x item rule R15-5 (Triple Tonic)",
         "done": cf.get("R15-5_triple_tonic"), **ref("items_confirmations_")},
        {"id": "DEV-roles", "plan": "section 3: role inference (Smite / support item / 3-8 min position, Hungarian)",
         "done": s1.get("roles"), "also": [x for x in (iv.get("assistant_decisions_noted") or []) if "slot" in x],
         **ref("stage1_decisions_")},
        {"id": "DEV-V-rows", "plan": "section 4: one random ms per minute bucket",
         "done": td.get("V_rows"), **ref("stage2_predecisions_")},
        {"id": "DEV-remakes", "plan": "(not in the plan)", "done": pre.get("remakes"),
         **ref("stage2_predecisions_")},
        {"id": "DEV-martingale-b-p", "plan": "section 4 (b): joint F test",
         "done": pre.get("martingale_b_primary_p"), **ref("stage2_predecisions_")},
        {"id": "DEV-martingale-h", "plan": "section 4: h from the 15.14 engagement-length distribution",
         "done": td.get("martingale_h"), **ref("stage2_predecisions_")},
        {"id": "DEV-effect-flags", "plan": "section 3: four effect flags",
         "done": s1.get("effect_flags"), **ref("stage1_decisions_")},
        {"id": "DEV-validation-99", "plan": "test 9: stop below 99% for every validation report",
         "done": s1.get("validation_99pct_rule"), **ref("stage1_decisions_")},
        {"id": "DEV-executions", "plan": "(not in the plan)", "done": s1.get("executions"),
         **ref("stage1_decisions_")},
        {"id": "DEV-constant-objective-columns", "plan": "(not in the plan)",
         "done": s1.get("assistant_decision_noted"), **ref("stage1_decisions_")},
        {"id": "DEV-ACE", "plan": "(v3.3 ACE truncation)",
         "done": next((x for x in (iv.get("assistant_decisions_noted") or []) if "ACE" in x), None),
         **ref("items_validation_decisions_")},
        {"id": "DEV-V-technical-defaults", "plan": "(not specified)",
         "done": {k: td.get(k) for k in ("V_selection", "MLP", "final_fit", "row_weights", "martingale_overlap",
                                         "recent_deaths_window", "post_states")}, **ref("stage2_predecisions_")},
    ]
    return {"e1_disclosures": e1_disc, "deviations": dev}


def population_rules(yields: Mapping[str, Any], preset: Mapping[str, Any], records: Mapping[str, Any]) -> Dict[str, Any]:
    _k, s1 = _latest(records, "stage1_decisions_")
    params = {p: (yields[p]["params"] or {}).get("v4") for p in PATCHES}
    if params[TRAIN_PATCH] != params[SELECT_PATCH]:
        raise DraftRefused("15.14 and 15.15 detect runs used different v4 parameters")
    return {
        "boundaries": {"G_ms": preset["values"]["TF2_KILL_CLUSTER_GAP_MS"],
                       "D": preset["values"]["CLUSTER_MAX_DIAMETER"], "source": "record 1A"},
        "detector_params_v4": params[TRAIN_PATCH],
        "tau": "first kill - 15 s (TF2_ENGAGE_PRE_KILL_MS = 15,000)",
        "survival": "kill-event survival with the patch respawn formula; >= 2 alive per team at tau",
        "merge": "adjacent merge with the A6 fix; head counts recomputed from the union of kill sets",
        "isolation": "no kill other than the engagement's own and no other engagement's tau in [tau, last kill]",
        "clean": "tau frame age < 10,000 ms (9,999 included, 10,000 excluded)",
        "cohorts": {"T": "both teams >= 4 credited", "S": "both teams 2-3", "ASYM": "one team 2-3, the other >= 4",
                    "P": "one team < 2"},
        "remakes": "GAME_END < 300,000 ms excluded everywhere",
        "roles": s1.get("roles"),
    }


def _ids(prefix: str, texts: Sequence[str]) -> List[Dict[str, str]]:
    return [{"id": f"{prefix}{i:02d}", "text": t} for i, t in enumerate(texts, 1)]


def pending_items(v: Mapping[str, Any], bal: Mapping[str, Any], e2: Mapping[str, Any],
                  yields: Mapping[str, Any], drift: Optional[Mapping[str, Any]] = None,
                  oof: Optional[Mapping[str, Any]] = None, prices: Optional[Mapping[str, Any]] = None,
                  ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """(blocking, author).  Blocking items (id 'B..') make the draft not lockable: the lock command refuses while any
    exists; they are fixed by re-running the draft on complete inputs, never by the author's word.  Author items
    (id 'A..') need a written resolution from the author (lock --resolutions FILE)."""
    blocking: List[str] = []
    author = [
        "q grid details marked 'draft default' (LightGBM early-stopping rows and patience, max_bin; within-learner "
        "choice on 15.15) and the PT_flex grid (n_knots_p {4, 6, 8}, C 0.001-10 with the edge rule)",
        "wording of the prior-exposure and compliance statements",
        "the deviation list (complete? any item to move to the thesis limitations?)",
        f"gold at tau in the balance table: {(bal.get('gold_field') or {}).get('field', FRAME_GOLD_FIELD)} "
        f"(event-updated: {(bal.get('gold_field') or {}).get('event_updated', False)})",
    ]
    if v.get("usable") is None:
        blocking.append("V: fit-V has not been given; record 1 cannot be locked until the frozen V is added")
    if oof is not None and not oof.get("ok"):
        blocking.append(f"oof_v: {oof.get('status')}")
    if prices is not None and not prices.get("ok"):
        blocking.append(f"prices_1514: {prices.get('status')}")
    if bal.get("rule_triggered"):
        author.append("the SMD disclosure sentence (max |SMD| > 0.1)")
    if bal.get("max_abs_smd") is None:
        blocking.append("balance: no SMD could be computed (a group with < 2 rows)")
    for p in PATCHES:
        r2 = yields[p]["r2_repro"]
        if r2.get("within_3pct") is False:
            blocking.append(f"{p}: r2_repro per selected match is more than 3% from the R2 baseline 19.6 "
                            "(RUNBOOK: stop)")
        if not e2[p]["counts_comparable"]:
            blocking.append(f"{p}: detect run is a sample; E2 counts are not full-patch counts")
        if not e2[p].get("legacy_denominator_same_cache", True):
            author.append(f"{p}: detect cache differs from the p3 cache; legacy per-match rates use another "
                          "denominator")
    changed = sorted({f for src in (drift or {}).values() for f in src["changed"]})
    if changed:
        author.append(f"code changed since the outputs were written ({len(changed)} files, each listed with "
                      f"--allow-drift or a smoke draft: {changed}); the code hashes in this draft are not the code "
                      "that produced the outputs")
    missing = sorted({f for src in (drift or {}).values() for f in src.get("not_hashed_now", [])})
    if missing:
        author.append(f"files hashed by an output but not by this draft: {missing}")
    author += [f"V: {w}" for w in (v.get("warnings") or [])]
    return _ids("B", blocking), _ids("A", author)


# ================================================================== build
def build_record(detect_dir: Path, extract_dir: Path, fit_v: Optional[Path], p3_paths: Sequence[Path],
                 scan_roots: Sequence[Path], smoke: bool = False, verify_chunks: bool = True,
                 records_dir: Path = RECORDS_DIR, record1a: Path = RECORD1A,
                 argv: Optional[Sequence[str]] = None, prices: Optional[Path] = None,
                 allow_drift: Sequence[str] = ()) -> Tuple[Dict[str, Any], HeldOutValues]:
    if not verify_chunks and not smoke:
        raise DraftRefused("--no-verify-chunks is a smoke option: a real draft verifies every 15.14 extract chunk")
    EC.predecisions_info()
    preset = preset_block(record1a)
    rec1a = read_json(record1a)
    records = records_block(records_dir)

    diags, diag_shas, yields = {}, {}, {}
    for p in PATCHES:
        d, s = load_diag(detect_dir, p)
        diags[p], diag_shas[p] = d, s
        yields[p] = detect_yields(d)
        if (d.get("predecisions") or {}).get("sha256") != EC.PREDECISIONS_SHA256:
            raise DraftRefused(f"detect diag {p} does not carry the pinned pre-decision record")
        if (d.get("preset") or {}).get("preset_values_sha256") != preset["values_sha256"]:
            raise DraftRefused(f"detect diag {p} was run with other v4-exact preset values")
    ex_man, ex_sha = {}, {}
    for p in PATCHES:
        ex_man[p], ex_sha[p] = load_extract_manifest(extract_dir, p)
    sample_inputs = [f"detect {p}" for p in PATCHES if not yields[p]["full_run"]]
    sample_inputs += [f"extract {p}" for p in PATCHES if ex_man[p].get("sample")]
    if sample_inputs and not smoke:
        raise DraftRefused(f"sample inputs ({sample_inputs}) need --smoke; a real record 1 draft needs full runs")
    for p in PATCHES:
        det_sha = (ex_man[p].get("match_list") or {}).get("sha256")
        want = ((diags[p].get("outputs") or {}).get(f"matches_{p}.parquet") or {}).get("sha256")
        if det_sha != want:
            raise DraftRefused(f"extract {p} was not built from this detect run's match list")

    v = v_block(fit_v, allow_smoke=smoke, extract_shas=ex_sha)
    if v.get("smoke") and not smoke:
        raise DraftRefused("smoke V needs --smoke")

    oof = oof_block(fit_v, v, allow_smoke=smoke)
    pr = prices_block(prices, allow_smoke=smoke)

    p3, held = read_p3(p3_paths)
    e2 = e2_table(yields, p3)
    eng, cov = load_balance_rows(extract_dir, ex_man[TRAIN_PATCH], verify_chunks=verify_chunks)
    bal = balance_table(eng, cov, gold_info=gold_field(STATE_V3_COLUMNS))
    comp = compliance_scan(scan_roots)
    if not comp["ok"]:
        raise DraftRefused(f"compliance scan found held-out data: {comp['violations'][:20]}", code=3)

    code_now = code_block()
    drift = code_drift(code_now, {
        **{f"detect_diag_{p}": ((diags[p].get("code") or {}).get("files_sha256") or {}) for p in PATCHES},
        **{f"extract_manifest_{p}": (ex_man[p].get("code_sha256") or {}) for p in PATCHES},
        **({"fit_v_manifest": v.get("code_sha256") or {}} if v.get("usable") else {}),
    })
    drift_allowed = drift_policy(drift, allow_drift, smoke)
    blocking, author = pending_items(v, bal, e2, yields, drift, oof, pr)
    disc = disclosures_block(records, rec1a)
    record = {
        "record": "1 DRAFT - not locked; the author reviews and signs off, and the locked record 1 is written "
                  "separately",
        "status": "DRAFT", "is_final_lock": False, "usable_as_record1": False, "smoke": bool(smoke),
        "lockable": not blocking,
        "V_frozen_bundle_sha256": (v.get("V_frozen") or {}).get("bundle_sha256") if v.get("usable") else None,
        "created_utc": utc_stamp(),
        "builder": {"script": rel(HERE), "sha256": sha256_file(HERE), "argv": list(argv or [])},
        "plan": {"path": str(PLAN), "sha256": sha256_file(PLAN) if PLAN.is_file() else None},
        "preset": preset,
        "record1a": {"path": str(record1a), "sha256": sha256_file(record1a), "locked": rec1a.get("locked"),
                     "evidence": rec1a.get("evidence"), "sensitivity_rows_for_E3": rec1a.get("sensitivity_rows_for_E3")},
        "records": records,
        "git": git_block(), "code_sha256": code_now, "code_drift_vs_outputs": drift, "code_drift_allowed": drift_allowed,
        "state_columns": state_block(ex_man), "game_rules": game_rules_block(),
        "population_rules": population_rules(yields, preset, records),
        "detect": {p: {"diag": str(Path(detect_dir) / f"diag_{p}.json"), "diag_sha256": diag_shas[p], **yields[p]}
                   for p in PATCHES},
        "extract": {p: extract_summary(ex_man[p], ex_sha[p]) for p in PATCHES},
        "e2_cumulative": {"p3": {k: p3[k] for k in ("files", "arms", "note")}, "tables": e2},
        "balance_15_14": bal,
        "V": v, "oof_v": oof, "prices_1514": pr,
        "q_candidate_grids": Q_CANDIDATE_GRIDS, "pt_candidate_grids": PT_CANDIDATE_GRIDS,
        "prior_exposure": PRIOR_EXPOSURE,
        "compliance_no_heldout_before_record1": comp,
        "e1_disclosures": disc["e1_disclosures"], "deviations": disc["deviations"],
        "pending_blocking": blocking, "pending_author_confirmation": author,
        "author_signoff": {"signed": False, "signed_by": None, "signed_utc": None,
                           "note": "the locked record 1 is a new file (not this draft) written after sign-off"},
    }
    return record, held


# ================================================================== Korean summary
def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "예" if x else "아니오"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_md(r: Mapping[str, Any]) -> str:
    L: List[str] = []
    a = L.append
    a(f"# 기록 1 초안{' (스모크)' if r['smoke'] else ''} — 잠금 아님")
    a("")
    a(f"- 상태: **{r['status']}**. 저자가 검토하고 서명한 뒤 별도 파일로 기록 1을 잠근다. 이 파일은 기록 1로 쓸 수 없다.")
    a(f"- 생성: {r['created_utc']} (UTC), 스크립트 `{r['builder']['script']}` sha256 `{r['builder']['sha256'][:12]}`")
    g = r["git"]
    a(f"- git HEAD `{str(g.get('head'))[:10]}`, 추적 파일 변경 {_f(g.get('dirty_tracked'))}"
      f" ({len(g.get('dirty_tracked_paths') or [])}개), 코드 폴더의 추적 안 된 파일 {len(g.get('untracked_in_code_dirs') or [])}개")
    if r["smoke"]:
        a("- **스모크 입력**으로 만든 초안이다. 수치는 표본 경기에서 나온 것이며 기록 1에 쓸 수 없다.")
    a("")
    a("## 1. 잠그는 설정과 해시")
    pr = r["preset"]
    a(f"- preset `{pr['name']}` 값 sha256 `{pr['values_sha256'][:16]}`; G = {_f(pr['values']['TF2_KILL_CLUSTER_GAP_MS'], 0)} ms, "
      f"D = {_f(pr['values']['CLUSTER_MAX_DIAMETER'], 0)}; 기록 1A 잠금값과 일치")
    a(f"- 기록 1A `{Path(r['record1a']['path']).name}` sha256 `{r['record1a']['sha256'][:16]}`")
    sc = r["state_columns"]
    a(f"- StateV3 이름 해시 `{sc['STATE_V3_NAME_HASH'][:16]}` ({sc['n_state_columns']}열), 위치 정보군 해시 "
      f"`{sc['SETUP_NAME_HASH'][:16]}` ({sc['n_setup_columns']}열)")
    gr = r["game_rules"]
    a(f"- Data Dragon v2 index sha256 `{gr['index_sha256'][:16]}`, 표 {len(gr['tables_sha256'])}개(묶음 해시 "
      f"`{gr['tables_set_sha256'][:16]}`), 아이템 효과 플래그 `{str(gr['item_effect_flags_sha256'])[:16]}`")
    a(f"- 기록 폴더 파일 {len(r['records']['files'])}개의 sha256과 결정 기록 {len(r['records']['decision_records'])}개의 본문을 담았다.")
    a("")
    a("## 2. 15.14·15.15 수율 (검출)")
    for p in PATCHES:
        y = r["detect"][p]
        sel = y["selection"]
        a(f"### {p}")
        a(f"- 경기: 목록 {_f(sel.get('n_list_patch'))}, 선택 {_f(sel.get('n_selected'))}, ok {_f(y['n_ok'])}"
          f"{' (표본)' if not y['full_run'] else ''}; 재경기 제외 {_f(y['remakes'].get('n_excluded'))}"
          f" (목록 단계 {_f(y['remakes'].get('n_excluded_list_stage'))}, 사건 단계 {_f(y['remakes'].get('n_excluded_event_stage'))})")
        a(f"- v4 교전 {_f(y['v4']['n'])} (ok 경기당 {_f(y['v4']['per_ok_match'])})")
        a("")
        a("| 경우 | 비클린·비고립 | 비클린·고립 | 클린·비고립 | 클린·고립 | 클린·고립 / 경기 |")
        a("|---|---:|---:|---:|---:|---:|")
        cells = {(c["cohort"], c["clean"], c["isolated"]): c["n"] for c in y["v4"]["cells"]}
        for c in COHORTS_V4:
            a(f"| {c} | {_f(cells[(c, 0, 0)])} | {_f(cells[(c, 0, 1)])} | {_f(cells[(c, 1, 0)])} | {_f(cells[(c, 1, 1)])} "
              f"| {_f(y['v4']['clean_isolated_per_ok_match'][c], 3)} |")
        r2 = y["r2_repro"]
        a("")
        a(f"- r2_repro {_f(r2['n'])}건: ok 경기당 {_f(r2['per_ok_match'])}, 선택 경기당 {_f(r2['per_selected_match'])} "
          f"(R2 기준 19.6과 {_f(100 * r2['rel_diff_per_selected'] if r2['rel_diff_per_selected'] is not None else None, 1)}% 차이, "
          f"3% 안 {_f(r2['within_3pct'])})")
        ex = r["extract"][p]
        rpm = ex.get("rows_per_match") or {}
        a(f"- 추출: 경기 {_f(ex.get('n_matches'))}, 행/경기 교전 {_f(rpm.get('eng'))}, 클린 위치 {_f(rpm.get('eng_clean_setup'))}, "
          f"V {_f(rpm.get('v'))}, 마팅게일 {_f(rpm.get('mart'))}; 모르는 아이템 비율 {_f((ex.get('item_check') or {}).get('rate'), 4)}")
        a("")
    a("## 3. E2 누적 변화표 (v3.3 대비)")
    a("v3.3·evr의 경우(T/S/P)는 존재 게이트 기반 규모 계급이고, v4의 경우(T/S/ASYM/P)는 킬 기록 참여로 센 것이다. "
      "경우 열은 정의가 같지 않으므로 합계와 경기당 비율을 비교한다.")
    for p in PATCHES:
        t = r["e2_cumulative"]["tables"][p]
        a("")
        a(f"### {p}{' (v4 행은 표본)' if t['v4_is_sample'] else ''}")
        a("| 단계 | T | S | ASYM | P | 합계 | 경기당 | 앞 단계 대비 |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in t["rows"]:
            bc = row["by_cohort"]
            a(f"| {row['stage']} | {_f(bc['T'])} | {_f(bc['S'])} | {_f(bc['ASYM'])} | {_f(bc['P'])} | {_f(row['total'])} "
              f"| {_f(row['per_match_total'])} | {_f(row.get('delta_per_match_vs_previous'))} |")
        a("")
        a(f"- 경기당 분모: v3.3·evr은 검출 목록의 패치 경기 수(p3는 같은 캐시의 모든 경기를 돌렸다; 같은 캐시 "
          f"{_f(t.get('legacy_denominator_same_cache'))}, 재경기 포함), v4·r2는 ok 경기 수(재경기 제외).")
        if t["v4_is_sample"]:
            a("- **검출이 표본 실행이다.** v4·r2 행은 표본 경기의 건수이고 v3.3·evr 행은 패치 전체 건수이므로 경기당 비율만 비교한다.")
    a("")
    a("## 4. 클린 대 비클린 균형 (15.14 고립 교전, SMD)")
    b = r["balance_15_14"]
    a("| 공변량 | " + " | ".join(list(COHORTS_V4) + ["전체"]) + " |")
    a("|---|" + "---:|" * (len(COHORTS_V4) + 1))
    for name, _d in BALANCE_COVARIATES:
        vals = [b["by_cohort"][g]["covariates"][name]["smd"] for g in (*COHORTS_V4, "all")]
        a(f"| {name} | " + " | ".join(_f(v, 3) for v in vals) + " |")
    a("| n (클린 / 비클린) | " + " | ".join(f"{_f(b['by_cohort'][g]['n_clean'])} / {_f(b['by_cohort'][g]['n_nonclean'])}"
                                         for g in (*COHORTS_V4, "all")) + " |")
    a("")
    mx = b["max_at"]
    a(f"- 최대 |SMD| = {_f(b['max_abs_smd'], 3)} ({mx['cohort']}, {mx['covariate']}). 규칙: 0.1을 넘으면 공개 문장을 넣는다 → "
      f"{'**넣는다**' if b['rule_triggered'] else '넣지 않는다'}.")
    if b["rule_triggered"]:
        a(f"- 공개 문장: {b['disclosure_ko']}")
    gf = b.get("gold_field") or {}
    a(f"- 골드: `{gf.get('field')}` (사건으로 갱신된 값 {_f(gf.get('event_updated'))}). {gf.get('source', '')}")
    a("- 웨이브 위상: (tau − 65,000) mod 30,000 ms(첫 웨이브 1:05, 이후 30초마다), 초반 지표: tau < 14:00.")
    a("")
    a("## 5. 기본 승률 예측기 V")
    v = r["V"]
    if v.get("usable"):
        sel = v.get("selection") or {}
        st = v.get("stop_rule") or {}
        a(f"- 선택: **{v['chosen']}** (V_SELECT 원 log loss {_f(sel.get('best_loss'), 4)}; 0.0005 안 단순 모형 채택 "
          f"{_f(sel.get('simpler_taken'))}). 후보 손실: " + ", ".join(f"{k} {_f(x, 4)}" for k, x in (sel.get('losses') or {}).items()))
        a(f"- 골드 차 기준선 대비: 선택 V {_f(st.get('chosen_raw_select_logloss'), 4)} 대 골드 {_f(st.get('gold_raw_select_logloss'), 4)}, "
          f"이김 {_f(st.get('beats_gold'))}, 기록 1 전 정지 {_f(st.get('stop_before_record1'))}")
        a(f"- 동결 V 묶음 sha256 `{str((v.get('V_frozen') or {}).get('bundle_sha256'))[:16]}`, frozen_manifest sha256 "
          f"`{v['frozen_manifest']['sha256'][:16]}`{' — 스모크 V' if v.get('smoke') else ''}")
        vs = v.get("V_structure") or {}
        rc = ((v.get("v_revision") or {}).get("recalibration")) or {}
        a(f"- V 구조: 묶음 형식 `{vs.get('bundle_format')}`, 진영 표지 {_f(vs.get('side_marker'))}, 재보정 적용 "
          f"{_f(vs.get('recalibrated'))} (재보정 조건 충족 {_f(rc.get('triggered'))}, 적합 {_f(rc.get('fitted'))}, "
          f"채택 {_f(rc.get('adopted'))})")
    else:
        a(f"- {v.get('status')}")
    vg = VM.GRIDS
    a(f"- V 후보 격자: `ev4_v_models.GRIDS`(로지스틱 C {vg['logistic']['C']}; LightGBM num_leaves {vg['lgbm']['num_leaves']}; "
      f"MLP weight decay {vg['mlp']['weight_decay']}).")
    for w in v.get("warnings") or []:
        a(f"- 경고: {w}")
    a(f"- 기록 1 최상위 `V_frozen_bundle_sha256`: `{r.get('V_frozen_bundle_sha256') or '—'}`")
    oof = r.get("oof_v") or {}
    if oof.get("ok"):
        a(f"- OOF V(15.14 학습 라벨용): oof_manifest sha256 `{oof['manifest_sha256'][:16]}`, 겹 묶음 sha256 "
          + ", ".join(f"{k} `{str(s)[:12]}`" for k, s in sorted((oof.get("fold_bundle_sha256") or {}).items())))
    else:
        a(f"- OOF V: {oof.get('status')}")
    pr = r.get("prices_1514") or {}
    if pr.get("ok"):
        a(f"- 15.14 사건 가격표 `{Path(pr['path']).name}` sha256 `{pr['sha256'][:16]}` (경기 {_f(pr.get('n_matches'))}"
          f"{', 표본' if pr.get('sample') else ''})")
    else:
        a(f"- 15.14 사건 가격표: {pr.get('status')}")
    a("")
    a("## 6. q와 PT 후보 격자 (여기서 고정)")
    q = r["q_candidate_grids"]["learners"]
    a(f"- 로지스틱: 표준화 입력, L2, C ∈ {q['logistic']['C']}")
    a(f"- LightGBM: num_leaves ∈ {q['lgbm']['num_leaves']}, learning_rate 0.05, min_data_in_leaf 200, feature_fraction 0.8, "
      "조기 종료, 최대 2,000회")
    a("- 선택: 경우마다 15.15 Brier로 고른다. 로지스틱과의 차이가 0.0005 이하이면 로지스틱.")
    pf = r["pt_candidate_grids"]["PT_flex"]
    a("- PT_linear: [p_pre, 경기 시각] 표준화 로지스틱, C = 1 (기존 정의).")
    a(f"- PT_flex (A8): 스플라인은 평균만 뺀다(표준화하지 않음). n_knots_p ∈ {pf['n_knots_p']}, n_knots_t = 4, "
      f"C ∈ {pf['C']}; 끝 값이 뽑히면 반 자릿수씩 최대 3번 넓힌다.")
    a("- '초안 기본값'으로 표시한 세부(조기 종료 행, 격자 안 선택 방식)는 저자 확인이 필요하다.")
    a("")
    a("## 7. 사전 노출과 준수")
    a(f"- {r['prior_exposure']['statement_ko']}")
    c = r["compliance_no_heldout_before_record1"]
    a(f"- 점검: 파일 {_f(c['n_files'])}개(parquet {_f(c['n_parquet'])}, JSON {_f(c['n_json'])}), 위반 {len(c['violations'])}건.")
    if c.get("statement_ko"):
        a(f"- {c['statement_ko']}")
    a("")
    dr = r["code_drift_vs_outputs"]
    changed = sorted({f for src in dr.values() for f in src["changed"]})
    a(f"- 산출물 이후 바뀐 코드 파일: {len(changed)}개" + (f" ({', '.join(changed)})" if changed else ""))
    a("")
    a("## 8. E1 공개 사항과 편차")
    for d in r["e1_disclosures"]:
        a(f"- E1: {d['text']}")
    for d in r["deviations"]:
        done = d.get("done")
        done = json.dumps(done, ensure_ascii=False) if isinstance(done, (dict, list)) else str(done)
        a(f"- {d['id']}: 계획 — {d.get('plan')}; 실행 — {done} (`{d.get('record')}`)")
    a("")
    a("## 9. 잠금 전 해결해야 할 항목")
    a(f"- 잠글 수 있는 초안: {_f(r.get('lockable'))}. 막는 항목(B)은 입력을 갖춰 초안을 다시 만들어야 풀린다. "
      "저자 항목(A)은 잠글 때 항목마다 저자의 서면 결정이 필요하다(lock --resolutions).")
    for it in r.get("pending_blocking") or []:
        a(f"- **{it['id']}** (막음): {it['text']}")
    for it in r["pending_author_confirmation"]:
        a(f"- {it['id']}: {it['text']}")
    a("")
    return "\n".join(L)


# ================================================================== write
def write_draft(record: Mapping[str, Any], held: HeldOutValues, out_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    out_dir = Path(out_dir)
    smoke = bool(record.get("smoke"))
    if smoke and out_dir.resolve() == RECORDS_DIR.resolve():
        raise DraftRefused("a smoke draft may not be written into records/")
    stem = f"{DRAFT_PREFIX}{'SMOKE_' if smoke else ''}{record['created_utc']}"
    jp, mp = out_dir / f"{stem}.json", out_dir / f"{stem}_ko.md"
    if jp.exists() or mp.exists():
        raise DraftRefused(f"refusing to overwrite {jp}")
    text = json.dumps(record, ensure_ascii=False, indent=2, default=_json_default, allow_nan=True)
    md = render_md(record)
    leak = leak_check([text, md], held)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp.write_text(text + "\n", encoding="utf-8")
    mp.write_text(md, encoding="utf-8")
    return jp, mp, leak


# ================================================================== lock (author only)
SIGNOFF_RE = re.compile(r"^\s*(?P<name>[^/]*\S[^/]*?)\s*/\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$")


def _interactive_confirm(draft_sha: str, stdin: Any = None, prompt: Any = input) -> None:
    """A real (non-smoke) lock must be typed by the author at an interactive terminal: 'LOCK <first 12 hex of the
    draft sha256>'.  Agents and scripts run without a TTY and are refused here."""
    stdin = sys.stdin if stdin is None else stdin
    try:
        tty = bool(stdin.isatty())
    except (AttributeError, ValueError):
        tty = False
    if not tty:
        raise DraftRefused("the real lock must be run by the author in an interactive terminal (no TTY: refused; an "
                           "agent or script never locks record 1)")
    want = f"LOCK {draft_sha[:12]}"
    got = prompt(f"Type '{want}' to lock record 1 from this draft: ")
    if str(got).strip() != want:
        raise DraftRefused("lock confirmation not typed; nothing written")


def lock_record(draft: Path, draft_sha256: str, author_signoff: str, resolutions: Path,
                records_dir: Path = RECORDS_DIR, smoke: bool = False, predecisions_sha256: Optional[str] = None,
                confirm: Any = _interactive_confirm) -> Tuple[Path, str]:
    """Turn an author-approved DRAFT into the final record1_<ts>.json in records_dir.  Refuses unless: the draft is
    an unmodified record1_DRAFT_*.json with this sha256 and status DRAFT; it is lockable (no blocking pending item;
    V bundle, oof_v and prices_1514 recorded); every author item has a written resolution in `resolutions`
    ({id: text}, no unknown ids); --author-signoff is 'Name / YYYY-MM-DD'; the embedded record 1A and pre-decision
    sha256 match the files in records_dir; no locked record 1 exists yet.  A smoke draft locks only with smoke=True
    into a directory other than records/ (and the result stays smoke, which assert_record1_locked refuses).  The
    real lock also needs the author's typed confirmation at a TTY (`confirm`)."""
    rd = Path(records_dir)
    real_rd = rd.resolve() == RECORDS_DIR.resolve()
    if smoke and real_rd:
        raise DraftRefused("a smoke lock may not be written into records/")
    m = SIGNOFF_RE.match(str(author_signoff or ""))
    if not m:
        raise DraftRefused("--author-signoff must be 'Name / YYYY-MM-DD'")
    dp = Path(draft)
    if not dp.is_file() or not dp.name.startswith(DRAFT_PREFIX) or dp.suffix != ".json":
        raise DraftRefused(f"{dp} is not a record1_DRAFT_*.json draft")
    dsha = sha256_file(dp)
    if not draft_sha256 or dsha != str(draft_sha256).strip().lower():
        raise DraftRefused(f"draft sha256 {dsha} is not the approved {draft_sha256}")
    d = read_json(dp)
    if d.get("status") != "DRAFT" or d.get("is_final_lock") is not False or \
            not str(d.get("record", "")).startswith("1 DRAFT"):
        raise DraftRefused("the file is not an unlocked record 1 DRAFT")
    if bool(d.get("smoke")) != bool(smoke):
        raise DraftRefused("a smoke draft locks only with --smoke, and a real draft only without it")
    if d.get("lockable") is not True or d.get("pending_blocking"):
        raise DraftRefused(f"the draft is not lockable; blocking items: {[b.get('id') for b in d.get('pending_blocking') or []]}")
    if not RL.HEX64.match(str(d.get("V_frozen_bundle_sha256") or "")):
        raise DraftRefused("the draft has no top-level V_frozen_bundle_sha256")
    for k in ("oof_v", "prices_1514"):
        if not (d.get(k) or {}).get("ok"):
            raise DraftRefused(f"the draft's {k} block is not recorded ({(d.get(k) or {}).get('status')})")
    res = read_json(Path(resolutions))
    ids = [it["id"] for it in d.get("pending_author_confirmation") or []]
    unresolved = [i for i in ids if not str(res.get(i) or "").strip()]
    unknown = sorted(set(res) - set(ids))
    if unresolved or unknown:
        raise DraftRefused(f"author items without a resolution {unresolved}; unknown ids {unknown}")
    try:
        RL.check_embedded_hashes(d, rd, predecisions_sha256)
    except RL.RecordNotLocked as e:
        raise DraftRefused(f"embedded record hashes: {e}") from e
    existing = sorted(p.name for p in rd.glob(f"{FINAL_PREFIX}*.json")
                      if not p.name.lower().startswith(DRAFT_PREFIX.lower()))
    if existing:
        raise DraftRefused(f"a locked record 1 already exists in {rd}: {existing}")
    if not smoke:
        confirm(dsha)
    ts = utc_stamp()
    rec = dict(d)
    rec.update({
        "record": "1", "status": "LOCKED", "is_final_lock": True, "usable_as_record1": True,
        "locked_utc": ts, "lockable": True, "pending_blocking": [],
        "author_signoff": {"signed": True, "signed_by": m.group("name").strip(), "signed_date": m.group("date"),
                           "signoff_text": str(author_signoff).strip(), "signed_utc": ts},
        "author_resolutions": {i: str(res[i]).strip() for i in ids},
        "locked_from_draft": {"path": str(dp), "sha256": dsha, "created_utc": d.get("created_utc")},
        "lock": {"script": rel(HERE), "sha256": sha256_file(HERE), "resolutions_file": str(resolutions),
                 "resolutions_sha256": sha256_file(Path(resolutions))},
    })
    try:
        RL.check_locked_fields(rec, allow_smoke=smoke)
    except RL.RecordNotLocked as e:
        raise DraftRefused(f"the locked record would not pass assert_record1_locked: {e}") from e
    out = rd / f"{FINAL_PREFIX}{'SMOKE_' if smoke else ''}{ts}.json"
    if out.exists():
        raise DraftRefused(f"refusing to overwrite {out}")
    rd.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rec, ensure_ascii=False, indent=2, default=_json_default, allow_nan=True) + "\n"
    out.write_text(text, encoding="utf-8")
    sha = sha256_file(out)
    if not smoke:
        RL.assert_record1_locked(out, sha, records_dir=rd, predecisions_sha256=predecisions_sha256)
    return out, sha


def main_lock(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="ev4_r1_record.py lock",
                                 description="AUTHOR ONLY: lock an approved DRAFT as records/record1_<ts>.json")
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--draft-sha256", required=True)
    ap.add_argument("--author-signoff", required=True, help="'Name / YYYY-MM-DD'")
    ap.add_argument("--resolutions", type=Path, required=True, help="JSON {author item id: resolution text}")
    ap.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    ap.add_argument("--smoke", action="store_true", help="test a smoke draft into a directory other than records/")
    a = ap.parse_args(list(argv))
    try:
        out, sha = lock_record(a.draft, a.draft_sha256, a.author_signoff, a.resolutions, a.records_dir, a.smoke)
    except DraftRefused as e:
        print(f"REFUSED (exit {e.code}): {e}", file=sys.stderr)
        return e.code
    print(f"locked record 1: {out}\nsha256: {sha}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["lock"]:
        return main_lock(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--detect", type=Path, default=STAGE2 / "detect")
    ap.add_argument("--extract", type=Path, default=STAGE2 / "extract")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit-v", type=Path, help="ev4_03 output dir (or its frozen_manifest.json)")
    g.add_argument("--no-fit-v", action="store_true", help="draft with the V block pending")
    ap.add_argument("--p3", type=Path, action="append", default=None)
    ap.add_argument("--scan-root", type=Path, action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=RECORDS_DIR)
    ap.add_argument("--smoke", action="store_true", help="allow sample inputs / a smoke V (never into records/)")
    ap.add_argument("--no-verify-chunks", action="store_true",
                    help="skip sha256 of the 15.14 extract chunks (--smoke only)")
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="prices_1514.json (missing -> pending)")
    ap.add_argument("--allow-drift", action="append", default=[], metavar="FILE",
                    help="a code file (as in code_sha256) whose change since the outputs was checked; repeat per file")
    a = ap.parse_args(argv)
    try:
        if a.smoke and a.out_dir.resolve() == RECORDS_DIR.resolve():
            raise DraftRefused("--smoke needs --out-dir outside records/")
        if a.no_verify_chunks and not a.smoke:
            raise DraftRefused("--no-verify-chunks is a smoke option (--smoke)")
        rec, held = build_record(a.detect, a.extract, None if a.no_fit_v else a.fit_v,
                                 a.p3 or list(P3_DEFAULT), a.scan_root or [STAGE2], smoke=a.smoke,
                                 verify_chunks=not a.no_verify_chunks, argv=argv, prices=a.prices,
                                 allow_drift=a.allow_drift)
        jp, mp, leak = write_draft(rec, held, a.out_dir)
    except DraftRefused as e:
        print(f"REFUSED (exit {e.code}): {e}", file=sys.stderr)
        return e.code
    b = rec["balance_15_14"]
    mx = b["max_abs_smd"]
    smd_txt = (f"max |SMD| {mx:.3f} ({b['max_at']['cohort']}, {b['max_at']['covariate']})" if mx is not None
               else "max |SMD| not computable")
    print(f"draft: {jp}\nsummary: {mp}\nsha256: {sha256_file(jp)}")
    print(f"V: {rec['V']['status']}; {smd_txt} -> disclosure {'added' if b['rule_triggered'] else 'not needed'}; "
          f"compliance ok, {leak['n_held_out_values_checked']} held-out values checked, 0 written")
    print(f"lockable: {rec['lockable']}; blocking: {[x['id'] + ' ' + x['text'][:70] for x in rec['pending_blocking']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
