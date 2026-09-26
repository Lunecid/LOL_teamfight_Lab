"""v4-exact stage 2, R6 labels: SVI (h = 60 / 90 / 120, primary 90) and the E5 outcome labels, per patch.

Plan: outputs/diag_survival_dbscan_20260925/docs/REESTIMATION_PLAN_V4_EXACT_20260925.md, section 5 ("라벨"),
"단계 2" (ev4_04_labels.py: SVI, training labels by cross-fitting, labels for E5 / E6), E5, E6.  Preset 'v4-exact'
(G = 14,000 ms, D = 4,300), locked in record 1A.  Author pre-decisions: outputs/reest_exact_v4_20260925/records/
stage2_predecisions_20260925T151437Z.json (pinned in ev4_common; 'post_states: computed at the label stage after V is
frozen'; remakes excluded from labels).

Two sub-commands:
  prices   OLS event-price estimation on 15.14 ONLY (function estimate_prices_1514) -> prices_1514.json
  labels   labels_<patch>.parquet (+ labels_<patch>.json) for one patch's ev4_02 extract + ev4_01 detect table

Implementation decisions made BEFORE any label was computed (2026-09-26, recorded here, in DECISIONS and in every
labels manifest):
  L1  Rows = the ev4_02 engagement rows (ISOLATED engagements, all cohorts, remakes already excluded; asserted again:
      no row with game_end < 300,000).  Keys match_id, eng_idx (ev4_01), tau; cohort / clean / isolated / t5 / counts
      are passed through and cross-checked against the ev4_01 detect table (join on match_id + tau, exactly one row).
  L2  p_pre = V(eng_X row) = V(StateV3 at tau - 1) from the stored extract matrix.  Pre-state spot check: the first
      engagement of the first 20 matches of every chunk is rebuilt from the cache pack (StateBuilderV3.at(tau - 1),
      float32) and must equal eng_X exactly (NaN == NaN); any mismatch halts.
  L3  e_h = gameplay.labels_exact.label_endpoint(L, events, h_s=h, next_start, game_end) for h in (60, 90, 120) (the DR
      rule scripts/engagement_labels_v3_rules.py imported unchanged), next_start = the extract's next_start (-1 ->
      absent), game_end = the extract's game_end (re-read from the events and compared).  The recomputed e90 must equal
      the extract's e90.
  L4  Endpoint validity (DR rule, as scripts/fc20260915_extract.py): endpoint_validity(e_h, L, tau - 1, support_start
      = max(0, first frame), last_frame = last frame) all true, and NOT next_start <= L
      ('same_match_overlap_next_start_le_L'); the post state must build with snapshot <= e_h.  An invalid endpoint
      gives p_post / dv NaN and svi <NA> (excluded, never imputed); the reasons are stored per h.
  L5  Post state = StateBuilderV3(pack, patch).at(e_h), cast to float32 like the extract, computed on the fly and NOT
      stored; p_post_h = V(post state).
  L6  svi_h = DR label_from_delta(p_post_h - p_pre) = 1[dv > 0]; an exact zero is 0 (flag svi_zero_h = 1).  Blue
      perspective (V = P(blue wins)); primary h = 90 ('svi', 'valid' are the h = 90 columns).
  L7  V source.  15.14 (the TRAIN patch of V and of q): OUT-OF-FOLD V only - match m is labelled with the fold model
      k = ev4_v_models.cv_fold(m) (sha256('<m>:ev4_03:cv5') mod 5) that never saw fold k (spec OOF_SPEC below; the
      script REFUSES 15.14 until ev4_03 exports it).  15.15 and every held-out patch: the frozen V
      (ev4_v_models.assert_v_usable + load_v with the manifest's bundle sha256).  15.14 / 15.15: the V manifest's
      train / select extract manifest sha256 must equal the extract being labelled's.  Held-out patches: the LOCKED
      record 1 (ev4_record_lock.assert_record1_locked) must name the frozen V bundle sha256 in its top-level
      V_frozen_bundle_sha256 (check_record1_v; no other key is read).
  L8  E5 outcome labels (NOT for training): kill_diff_label (own kills = ev4_01 kill_idx; an execution, killerId 0,
      counts against the victim's team); exchange_label_h (labels_exact.exchange_outcome: prices from
      prices_1514.json ONLY, radius D = the preset CLUSTER_MAX_DIAMETER, dead zone 300 g, ties excluded); next
      objective label_h (first elite objective in (e_h, e_h + 180 s], despawns skipped).  Computed for every h with
      e_h >= L, independently of the SVI validity.  1 = blue, 0 = red, <NA> = excluded.
  L9  Prices (estimate_prices_1514): per team and minute-frame interval starting >= 2:00, OLS of the team gold change
      on the team's event counts in the interval (analysis.event_prices.match_rows design: kill bounty, kills, assists,
      plates, towers by tier, inhibitor, first tower, dragon, elder, baron, herald, voidgrubs, atakhan, ward kills,
      lane / jungle CS change) with intercept; fitted from per-match sufficient statistics; match-cluster (CR1)
      standard errors and a match bootstrap (B = 200, seed 20260925); the table is
      analysis.event_prices.price_table(fit) (rounded to 5 g, monsters + one jungle CS, min count 200, first-tower
      bonus dropped).  Matches: the 'ok' rows of the ev4_01 15.14 match list (sha256 order), remakes (GAME_END <
      300,000) skipped; the pack meta patch must be 15.14.  15.15 / 15.16 are never read for prices.
  L10 Guards: ev4_common.predecisions_info() first; split_guard (EX.check_patch_access: held-out patches need --record1
      PATH --record1-sha256 HEX of the FULL record 1, never record 1A) BEFORE any data; for a held-out patch
      held_out_guard runs right after it and before any held-out manifest is read: ev4_record_lock.
      assert_record1_locked (a LOCKED, signed, non-smoke record1_<ts>.json inside records/ with the given sha256,
      embedded record 1A / pre-decision sha256 matching records/) and check_record1_v; preset lock
      (EX.load_params: apply_preset 'v4-exact', require_locked, record-1A G / D, core/presets.py sha256) and the extract
      manifest params must equal it; every chunk worker runs inside grid_guard.forbid_grid(); extract chunk files and
      the detect table are verified by sha256; extract code_sha256 must equal ev4_02_extract.code_hashes() of the
      current files (the post state is built with the current code) unless --allow-code-drift (logged); sample
      extracts / smoke V / sample prices / smoke OOF sets only with --smoke.

OOF_SPEC (what ev4_03_fit_v must export before 15.14 training labels can be written; see OOF_SPEC in the code).

Outputs (<out>, default outputs/reest_exact_v4_20260925/stage2/labels, or .../labels_smoke with --smoke):
  labels_<patch>.parquet   one row per extract engagement row, columns LABEL_COLUMNS
  labels_<patch>.json      manifest: inputs + sha256, guards, decisions, V source, counts, spot checks, runtimes
  parts_<patch>/           per-chunk parquet parts + sidecars (resume: a part whose plan hash and sha256 match is kept)

CLI
  python scripts/exact_v4/ev4_04_labels.py prices --matches <detect>/matches_15.14.parquet [--workers 4]
         [--out <stage2>/prices_1514.json]     smoke: --smoke --limit N --out FILE (the canonical path is refused)
  python scripts/exact_v4/ev4_04_labels.py labels --patch 15.15 --extract <extract>/15.15 --v-manifest
         <fit_v>/frozen_manifest.json [--detect PARQUET] [--prices FILE] [--out DIR] [--workers 4]
         [--allow-code-drift] [--smoke] [--limit-chunks N (smoke)]
  held-out patches additionally: --record1 PATH --record1-sha256 HEX
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys  # noqa: E402

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
for _p in (str(WT), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_common as EC  # noqa: E402
import ev4_record_lock as RL  # noqa: E402
import ev4_v_models as VM  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("ev4_02_extract", HERE.parent / "ev4_02_extract.py")
EX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EX)

OUT_BASE = EX.OUT_BASE
STAGE2 = OUT_BASE / "stage2"
DEFAULT_OUT = STAGE2 / "labels"
DEFAULT_OUT_SMOKE = STAGE2 / "labels_smoke"
DEFAULT_PRICES = STAGE2 / "prices_1514.json"
DEFAULT_PRICE_MATCHES = STAGE2 / "detect" / "matches_15.14.parquet"
CACHE = EX.CACHE
TRAIN_PATCH, SELECT_PATCH = "15.14", "15.15"
PRICE_PATCH = "15.14"
HORIZONS_S = (60, 90, 120)
PRIMARY_H = 90
SPOT_CHECK_N = 20
MAX_WORKERS = 4
PRICE_BOOT = 200
PRICE_SEED = 20260925
PRICES_FORMAT = "ev4_prices_1"
OOF_FORMAT = "ev4_oof_v_1"
OOF_FOLD_RULE = "ev4_v_models.cv_fold(match_id) = int(sha256('<match_id>:ev4_03:cv5').hexdigest()[:16], 16) mod 5"
N_FOLDS = 5
LABELS_FORMAT = "ev4_labels_1"
OOF_BLOCK_FILE = "oof_v_block.json"            # sidecar next to frozen_manifest.json (written by ev4_03b_oof_v.py)
OOF_BLOCK_FORMAT = "ev4_oof_v_block_1"
OOF_MANIFEST_KEYS = ("format", "train_patch", "kind", "hyperparameter", "n_folds", "fold_rule", "calibration",
                     "frozen_bundle_sha256", "smoke", "pilot", "folds")

OOF_SPEC = f"""\
ev4_03_fit_v must export out-of-fold (OOF) V predictors for the 15.14 training labels (plan: 'SVI(학습은 교차 적합)').
Required (ev4_04_labels refuses 15.14 until all of it exists and verifies):
  <fit_v out>/oof_v/oof_manifest.json   {{
      "format": "{OOF_FORMAT}", "train_patch": "15.14", "kind": <the CHOSEN kind (== V_frozen.kind)>,
      "hyperparameter": <the chosen grid value, e.g. {{"C": 0.01}}>, "n_folds": {N_FOLDS},
      "fold_rule": "{OOF_FOLD_RULE}",
      "calibration": "per fold PositiveSlopeSigmoid fitted on V_CAL (15.15) with unit weights",
      "frozen_bundle_sha256": <V_frozen bundle sha256 of the same run>, "smoke": <bool>, "pilot": <bool>,
      "folds": {{"0": {{"dir": "fold_0", "bundle_sha256": <sha256 of fold_0/bundle.json>, "heldout_fold": 0,
                      "train_match_ids_file": "fold_0/train_match_ids.txt",
                      "train_match_ids_sha256": <sha256 of that file>, "n_train_matches": <int>}}, ... "4": {{...}}}}}}
  <fit_v out>/oof_v/fold_k/             an ev4_v_models.save_bundle bundle: the chosen kind with the chosen
                                        hyperparameter, trained exactly like the final fit but on the 15.14 matches
                                        with cv_fold != k (logistic: all their rows; lgbm / mlp: those matches minus the
                                        inner 10 %, early-stopped on them; same 50 % swap rule; mlp with the FINAL seeds
                                        0, 1, 2), calibrated with its own PositiveSlopeSigmoid on V_CAL; bundle extra
                                        {{"oof_heldout_fold": k}}
  <fit_v out>/oof_v/fold_k/train_match_ids.txt   the training match ids, one per line, sorted (every id has
                                        cv_fold != k)
  frozen_manifest.json "oof_v": {{"dir": <path of oof_v>, "manifest_sha256": <sha256 of oof_manifest.json>,
                                 "kind": <chosen kind>}}
  or, since ev4_03 itself does not write that block, the sidecar <fit_v out>/{OOF_BLOCK_FILE} written by
  scripts/exact_v4/ev4_03b_oof_v.py (frozen_manifest.json is left untouched):
      {{"format": "{OOF_BLOCK_FORMAT}", "dir_rel": "oof_v", "manifest_sha256": <sha256 of oof_manifest.json>,
       "kind": <chosen kind>, "frozen_manifest_sha256": <sha256 of the frozen_manifest.json next to it>, ...}}
  It is read only when frozen_manifest.json has no 'oof_v' block; its oof_v dir is resolved next to the sidecar.
For logistic and lgbm the CV fits of the chosen family at the chosen grid value ARE these fold models (ev4_03b refits
them with the same rows, swap mask, seeds and budgets), each with one calibrator fitted on V_CAL.  For mlp the CV fits
use one seed (MLP_SEEDS_CV), so ev4_03b fits five 3-seed models.  write_oof_manifest in this script writes
oof_manifest.json and the id lists.
Every 15.14 match with V rows must be in the training lists of the 4 folds other than its own (labels check this:
coverage >= 99 % of the labelled matches outside --smoke).
"""

DECISIONS = {
    "L1_rows": "ev4_02 engagement rows (isolated, all cohorts); no game_end < 300000; joined to ev4_01 on match_id+tau",
    "L2_p_pre": "V(eng_X) = V(StateV3(tau-1)) stored; spot check first engagement of first 20 matches per chunk "
                "rebuilt from the pack == eng_X",
    "L3_endpoint": "labels_exact.label_endpoint(L, events, h_s in (60,90,120), next_start (extract), game_end) (DR rule "
                   "unchanged); recomputed e90 == extract e90",
    "L4_validity": "DR endpoint_validity(e_h, L, tau-1, max(0,first frame), last frame) and not next_start <= L; post "
                   "snapshot <= e_h; invalid -> p_post NaN, svi <NA> (excluded)",
    "L5_post_state": "StateBuilderV3.at(e_h) float32, on the fly, not stored",
    "L6_svi": "svi_h = DR label_from_delta(p_post_h - p_pre) = 1[dv > 0]; exact zero -> 0 (svi_zero_h); primary h=90",
    "L7_v_source": "15.14: out-of-fold V (fold cv_fold(match)), refused until ev4_03 exports OOF_SPEC; 15.15 / held-out: "
                   "frozen V (assert_v_usable + load_v sha); 15.14/15.15 V inputs must be the labelled extracts; "
                   "held-out: the locked record 1 (assert_record1_locked) must name the V bundle sha256 in its "
                   "top-level V_frozen_bundle_sha256; V bundle formats ev4_v_bundle_1 / _2 (side marker, optional "
                   "recalibration applied inside V.predict on the StateV3 rows, side = +1); the frozen V's structure "
                   "(side_marker, recalibrated) is recorded in inputs.v.v_structure and every OOF fold must share it",
    "L8_E5": "kill_diff_label (executions against the victim team); exchange_label_h (prices_1514 only, radius D, "
             "dead zone 300, ties excluded); next_objective_label_h (180 s, despawn skipped); for every h with e_h >= L; "
             "not for training",
    "L9_prices": "OLS on 15.14 ok matches (remakes skipped) of team gold change per frame interval (>= 2:00) on "
                 "analysis.event_prices regressors; sufficient statistics; CR1 match-cluster SE; match bootstrap "
                 f"B={PRICE_BOOT} seed {PRICE_SEED}; table = analysis.event_prices.price_table(fit)",
    "L10_guards": "predecisions first; split guard before data; held-out: assert_record1_locked + check_record1_v "
                  "right after the split guard, before any held-out manifest; preset lock; forbid_grid in workers; "
                  "sha256 of chunk files / detect table; code drift refused unless --allow-code-drift; smoke inputs only with --smoke",
}


class OOFNotAvailable(RuntimeError):
    """15.14 training labels need out-of-fold V predictors that ev4_03 does not (yet) export."""


# ============================================================================ small helpers
def log(msg: str) -> None:
    print(f"[ev4_04 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_load(p: Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _nan_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def _vec32(st) -> np.ndarray:
    return np.fromiter(st.values.values(), dtype=np.float64, count=len(st.values)).astype(np.float32)


def code_hashes() -> Dict[str, str]:
    """This script + the modules it uses for labels (the extract's own list is EX.code_hashes())."""
    out = {"scripts/exact_v4/ev4_04_labels.py": VM.sha256_file(HERE),
           "scripts/exact_v4/ev4_v_models.py": VM.sha256_file(HERE.parent / "ev4_v_models.py"),
           "scripts/exact_v4/ev4_record_lock.py": VM.sha256_file(HERE.parent / "ev4_record_lock.py"),
           "analysis/event_prices.py": VM.sha256_file(WT / "analysis" / "event_prices.py"),
           "gameplay/labels.py": VM.sha256_file(WT / "gameplay" / "labels.py")}
    out.update(EX.code_hashes())
    return out


def dr_rules():
    from gameplay.labels_exact import load_dr_rules
    R = load_dr_rules()
    if tuple(R.HORIZONS_S) != HORIZONS_S or int(R.PRIMARY_HORIZON_S) != PRIMARY_H:
        raise RuntimeError(f"DR rule horizons {R.HORIZONS_S} / primary {R.PRIMARY_HORIZON_S} differ from {HORIZONS_S} / "
                           f"{PRIMARY_H}")
    return R


# ============================================================================ prices (15.14 only)
def load_price_pack(mid: str, cache: Path = CACHE, patch: str = PRICE_PATCH) -> Dict[str, Any]:
    """meta, events, minute_ts, node_minute, gold_team_minute of one 15.14 match (never xy_raw_minute)."""
    meta = json.loads((cache / f"{mid}.meta.json").read_text(encoding="utf-8"))
    if str(meta.get("patch")) != patch:
        raise RuntimeError(f"{mid}: meta patch {meta.get('patch')!r} != {patch} (prices use {PRICE_PATCH} only)")
    meta = dict(meta)
    meta["team_map"] = {int(k): int(v) for k, v in (meta.get("team_map") or {}).items()}
    ev = json.loads((cache / f"{mid}.events.json").read_text(encoding="utf-8"))
    ev = [e for e in ev if isinstance(e, dict)] if isinstance(ev, list) else []
    with np.load(cache / f"{mid}.npz", allow_pickle=False) as z:
        pack = {"minute_ts": np.asarray(z["minute_ts"]).astype(np.int64), "node_minute": np.asarray(z["node_minute"]),
                "gold_team_minute": np.asarray(z["gold_team_minute"]).astype(np.float64)}
    pack.update(events=ev, meta=meta)
    return pack


def match_price_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    """Per-match OLS sufficient statistics with an intercept column: A'A, A'y, y'y, n, sum y."""
    A = np.hstack([np.ones((len(Y), 1)), np.asarray(X, dtype=np.float64)])
    y = np.asarray(Y, dtype=np.float64)
    return {"AtA": A.T @ A, "Aty": A.T @ y, "yty": float(y @ y), "n": int(len(y)), "sy": float(y.sum())}


def fit_prices_from_stats(stats: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """OLS from per-match sufficient statistics; OLS and match-cluster CR1 standard errors, R^2, counts.
    The coefficients equal a pooled OLS (np.linalg.lstsq) on the stacked rows."""
    AtA = np.sum([s["AtA"] for s in stats], axis=0)
    Aty = np.sum([s["Aty"] for s in stats], axis=0)
    yty = float(sum(s["yty"] for s in stats))
    n = int(sum(s["n"] for s in stats))
    sy = float(sum(s["sy"] for s in stats))
    p = AtA.shape[0]
    beta = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
    ssr = max(0.0, yty - 2.0 * float(beta @ Aty) + float(beta @ AtA @ beta))
    sst = yty - sy * sy / max(1, n)
    G = len(stats)
    try:
        Binv = np.linalg.inv(AtA)
        se_ols = np.sqrt(np.clip(np.diag(Binv) * ssr / max(1, n - p), 0, None))
        S = np.stack([s["Aty"] - s["AtA"] @ beta for s in stats])          # per-match scores A_g' e_g
        meat = S.T @ S
        c = (G / max(1, G - 1)) * ((n - 1) / max(1, n - p))
        se_cl = np.sqrt(np.clip(np.diag(c * Binv @ meat @ Binv), 0, None))
    except np.linalg.LinAlgError:
        se_ols = se_cl = np.full(p, np.nan)
    return {"n_intervals": n, "n_matches": G, "r2": 1.0 - ssr / sst if sst > 0 else float("nan"),
            "intercept": float(beta[0]), "intercept_se": float(se_ols[0]), "intercept_se_cluster": float(se_cl[0]),
            "coef": {k: float(beta[i + 1]) for i, k in enumerate(names)},
            "se": {k: float(se_ols[i + 1]) for i, k in enumerate(names)},
            "se_cluster": {k: float(se_cl[i + 1]) for i, k in enumerate(names)},
            "count": {k: float(AtA[0, i + 1]) for i, k in enumerate(names)}}


def bootstrap_prices(stats: Sequence[Mapping[str, Any]], names: Sequence[str], n_boot: int = PRICE_BOOT,
                     seed: int = PRICE_SEED) -> Dict[str, Dict[str, float]]:
    """Match bootstrap of the coefficients from the sufficient statistics (multinomial match weights)."""
    G = len(stats)
    if n_boot <= 0 or G < 2:
        return {}
    AtA = np.stack([s["AtA"] for s in stats])
    Aty = np.stack([s["Aty"] for s in stats])
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(int(n_boot)):
        w = rng.multinomial(G, np.full(G, 1.0 / G)).astype(np.float64)
        b = np.linalg.lstsq(np.tensordot(w, AtA, axes=1), w @ Aty, rcond=None)[0]
        coefs.append(b[1:])
    C = np.asarray(coefs)
    return {k: {"p2.5": float(np.percentile(C[:, i], 2.5)), "p50": float(np.percentile(C[:, i], 50)),
                "p97.5": float(np.percentile(C[:, i], 97.5)), "sd": float(C[:, i].std(ddof=1))}
            for i, k in enumerate(names)}


def _price_worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    from analysis.event_prices import match_rows
    from gameplay.grid_guard import forbid_grid
    out = {"stats": [], "status": Counter(), "remakes": []}
    with forbid_grid():
        for mid in task["match_ids"]:
            try:
                pack = load_price_pack(mid, Path(task["cache"]))
            except (OSError, ValueError, KeyError) as exc:
                out["status"][f"unreadable:{type(exc).__name__}"] += 1
                continue
            ge = EX.game_end_of(pack["events"])
            if EC.is_remake(ge):
                out["status"][EC.REMAKE_STATUS] += 1
                out["remakes"].append(mid)
                continue
            X, Y = match_rows(pack)
            if X is None or not len(Y):
                out["status"]["no_intervals"] += 1
                continue
            out["stats"].append(match_price_stats(X, Y))
            out["status"]["ok"] += 1
    out["status"] = dict(out["status"])
    return out


def price_match_ids(matches_file: Path, limit: Optional[int] = None) -> List[str]:
    """'ok' match ids of the ev4_01 15.14 match list (sha256 order), first `limit`."""
    return EX.match_universe(PRICE_PATCH, limit, Path(matches_file))


def estimate_prices_1514(match_ids: Sequence[str], cache: Path = CACHE, workers: int = 1, n_boot: int = PRICE_BOOT,
                         seed: int = PRICE_SEED, chunk: int = 500) -> Dict[str, Any]:
    """OLS event prices on 15.14 matches only (decision L9).  Returns {'fit', 'table', 'bootstrap', ...}."""
    from analysis.event_prices import REGRESSORS, price_table
    ids = list(match_ids)
    tasks = [{"match_ids": ids[i:i + chunk], "cache": str(cache)} for i in range(0, len(ids), chunk)]
    stats: List[Dict[str, Any]] = []
    status: Counter = Counter()
    remakes: List[str] = []
    results: Dict[int, Dict[str, Any]] = {}
    workers = max(1, min(MAX_WORKERS, int(workers)))
    if workers == 1 or len(tasks) <= 1:
        for i, t in enumerate(tasks):
            results[i] = _price_worker(t)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_price_worker, t): i for i, t in enumerate(tasks)}
            for f in as_completed(futs):
                results[futs[f]] = f.result()
    for i in sorted(results):                                # deterministic order regardless of workers
        stats.extend(results[i]["stats"])
        status.update(results[i]["status"])
        remakes.extend(results[i]["remakes"])
    if not stats:
        raise RuntimeError("no 15.14 match gave price intervals")
    fit = fit_prices_from_stats(stats, REGRESSORS)
    return {"fit": fit, "table": price_table(fit), "bootstrap": bootstrap_prices(stats, REGRESSORS, n_boot, seed),
            "regressors": list(REGRESSORS), "match_status": dict(status), "remakes_skipped": sorted(remakes),
            "n_boot": int(n_boot), "seed": int(seed)}


def run_prices(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    predec = EC.predecisions_info()
    access = EX.check_patch_access(PRICE_PATCH)                    # 15.14 is a selection patch; documents the guard
    _, params, guard = EX.load_params()
    if args.limit is not None and not args.smoke:
        raise SystemExit("--limit is a smoke option: pass --smoke (and --out FILE)")
    out = Path(args.out) if args.out is not None else DEFAULT_PRICES
    canonical = out.resolve() == DEFAULT_PRICES.resolve()
    if args.limit is not None and canonical:
        raise SystemExit(f"a sample price fit must not be written to the canonical {DEFAULT_PRICES}; pass --out")
    ids = price_match_ids(args.matches, args.limit)
    sample = bool(args.limit is not None or len(ids) < EX.FULL_RUN_MIN_MATCHES)   # same rule as ev4_02 'sample'
    if sample and not args.smoke:
        raise SystemExit(f"{len(ids)} matches (< {EX.FULL_RUN_MIN_MATCHES}) is a sample: only --smoke, with --out")
    if sample and canonical:
        raise SystemExit(f"a sample price fit must not be written to the canonical {DEFAULT_PRICES}; pass --out")
    log(f"prices: {len(ids)} 15.14 matches from {args.matches} (sample={sample}), workers {args.workers}")
    res = estimate_prices_1514(ids, Path(args.cache), workers=args.workers, n_boot=args.n_boot)
    blob = {"format": PRICES_FORMAT, "patch": PRICE_PATCH, "source_patches": [PRICE_PATCH],
            "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "sample": sample, "sample_limit": args.limit,
            "unit": "regression-estimated average team gold per event (local + global, killer + assisters), from "
                    "frame gold; not the rule payout",
            "note": "kill gold is read from the event (bounty + shutdownBounty); 'assists' is gold per assist beyond "
                    "the bounty",
            "table": res["table"], "fit": res["fit"], "bootstrap": res["bootstrap"], "regressors": res["regressors"],
            "n_boot": res["n_boot"], "seed": res["seed"],
            "matches": {"source": str(args.matches), "source_sha256": VM.sha256_file(Path(args.matches)),
                        "n_listed": len(ids), "match_ids_sha256": EX.sha256_text("\n".join(ids)),
                        "order": "sha256(match_id)", "status": res["match_status"],
                        "remakes_skipped": res["remakes_skipped"]},
            "remakes": EC.remake_rule(), "access": access, "guards": {"params": guard["params"],
                                                                     "record1a_sha256": guard["record1a_sha256"]},
            "decision": DECISIONS["L9_prices"],
            "predecisions": {k: v for k, v in predec.items() if k != "content"},
            "code_sha256": {k: v for k, v in code_hashes().items()
                            if k in ("scripts/exact_v4/ev4_04_labels.py", "analysis/event_prices.py",
                                     "scripts/exact_v4/ev4_common.py", "core/presets.py")},
            "git_head": EX.git_head(), "seconds": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    EX.write_json_atomic(out, blob)
    log(f"wrote {out}: table {res['table']}; R2 {res['fit']['r2']:.3f}; n_intervals {res['fit']['n_intervals']}")
    return blob


def load_prices(path: Path, smoke: bool) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The 15.14 price table (refuses another patch / format, a sample fit without --smoke)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"price table missing: {p} (run: ev4_04_labels.py prices ...)")
    blob = _json_load(p)
    if blob.get("format") != PRICES_FORMAT or blob.get("patch") != PRICE_PATCH or \
            list(blob.get("source_patches") or []) != [PRICE_PATCH]:
        raise RuntimeError(f"{p} is not an {PRICES_FORMAT} table estimated on {PRICE_PATCH} only")
    if blob.get("sample") and not smoke:
        raise SystemExit(f"{p} is a sample price fit; only --smoke may use it")
    table = {str(k): float(v) for k, v in blob["table"].items()}
    if "kills" not in table or "assists" not in table:
        raise RuntimeError(f"{p}: price table lacks kills / assists")
    return table, {"path": str(p), "sha256": VM.sha256_file(p), "sample": bool(blob.get("sample")), "table": table,
                   "n_matches": blob.get("fit", {}).get("n_matches")}


# ============================================================================ V sources
def write_oof_manifest(oof_dir: Path, kind: str, hyperparameter: Mapping[str, Any],
                       fold_bundles: Mapping[int, Mapping[str, Any]], fold_train_ids: Mapping[int, Sequence[str]],
                       frozen_bundle_sha256: str, smoke: bool, pilot: bool = False,
                       extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write oof_v/fold_k/train_match_ids.txt and oof_v/oof_manifest.json in the OOF_SPEC format.

    fold_bundles[k] = the ev4_v_models.save_bundle result of fold k (saved under oof_dir/fold_k).  extra = further
    top-level manifest keys (disclosures, provenance); it may not replace a key of the spec.  Returns the
    frozen_manifest 'oof_v' block ({'dir', 'manifest_sha256', 'kind'})."""
    oof_dir = Path(oof_dir)
    clash = sorted(set(extra or {}) & set(OOF_MANIFEST_KEYS))
    if clash:
        raise RuntimeError(f"extra manifest keys may not replace spec keys: {clash}")
    folds = {}
    for k in range(N_FOLDS):
        ids = sorted(str(m) for m in fold_train_ids[k])
        bad = [m for m in ids if VM.cv_fold(m) == k]
        if bad:
            raise RuntimeError(f"fold {k}: {len(bad)} training matches belong to the held-out fold ({bad[:3]})")
        fd = oof_dir / f"fold_{k}"
        if Path(fold_bundles[k]["dir"]).resolve() != fd.resolve():
            raise RuntimeError(f"fold {k} bundle must be saved under {fd}")
        tf = fd / "train_match_ids.txt"
        tf.write_text("\n".join(ids) + "\n", encoding="utf-8")
        folds[str(k)] = {"dir": f"fold_{k}", "bundle_sha256": fold_bundles[k]["bundle_sha256"], "heldout_fold": k,
                         "train_match_ids_file": f"fold_{k}/train_match_ids.txt",
                         "train_match_ids_sha256": VM.sha256_file(tf), "n_train_matches": len(ids)}
    man = {"format": OOF_FORMAT, "train_patch": TRAIN_PATCH, "kind": kind, "hyperparameter": dict(hyperparameter),
           "n_folds": N_FOLDS, "fold_rule": OOF_FOLD_RULE,
           "calibration": "per fold PositiveSlopeSigmoid fitted on V_CAL (15.15) with unit weights",
           "frozen_bundle_sha256": frozen_bundle_sha256, "smoke": bool(smoke), "pilot": bool(pilot), "folds": folds}
    man.update(dict(extra or {}))
    mp = oof_dir / "oof_manifest.json"
    EX.write_json_atomic(mp, man)
    return {"dir": str(oof_dir), "manifest_sha256": VM.sha256_file(mp), "kind": kind}


def read_v_manifest(path: Path) -> Tuple[Dict[str, Any], Path]:
    p = Path(path)
    if p.is_dir():
        p = p / "frozen_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"V frozen manifest missing: {p}")
    return _json_load(p), p


def read_oof_block_sidecar(vpath: Path) -> Optional[Dict[str, Any]]:
    """The <fit_v out>/oof_v_block.json sidecar of ev4_03b_oof_v.py, as an 'oof_v' block ({'dir', 'manifest_sha256',
    'kind', ...}), or None when there is none.  It must name the sha256 of the frozen_manifest.json next to it (so it
    cannot be moved to another fit-V run); its oof_v dir is resolved next to the sidecar (never its absolute path)."""
    vpath = Path(vpath)
    sp = vpath.parent / OOF_BLOCK_FILE
    if not sp.is_file():
        return None
    blk = _json_load(sp)
    if blk.get("format") != OOF_BLOCK_FORMAT:
        raise OOFNotAvailable(f"{sp}: format {blk.get('format')!r} is not {OOF_BLOCK_FORMAT}")
    if blk.get("frozen_manifest_sha256") != VM.sha256_file(vpath):
        raise OOFNotAvailable(f"{sp} belongs to another frozen_manifest.json (frozen_manifest_sha256 mismatch)")
    rel = str(blk.get("dir_rel") or "")
    if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise OOFNotAvailable(f"{sp}: dir_rel {rel!r} must be a relative path inside the fit-V directory")
    return {**blk, "dir": str(sp.parent / rel), "sidecar": str(sp), "sidecar_sha256": VM.sha256_file(sp)}


def check_oof(vman: Mapping[str, Any], vf: Mapping[str, Any], smoke: bool,
              vpath: Optional[Path] = None) -> Dict[str, Any]:
    """Validate the OOF export of an ev4_03 run (OOF_SPEC) WITHOUT loading models.  Raises OOFNotAvailable.

    The 'oof_v' block comes from frozen_manifest.json, or (when it has none and vpath, the frozen_manifest.json path,
    is given) from the ev4_03b_oof_v.py sidecar next to it (read_oof_block_sidecar)."""
    blk = vman.get("oof_v")
    source = "frozen_manifest"
    if not isinstance(blk, Mapping) and vpath is not None:
        blk = read_oof_block_sidecar(vpath)
        source = OOF_BLOCK_FILE
    if not isinstance(blk, Mapping) or not blk.get("dir") or not blk.get("manifest_sha256"):
        raise OOFNotAvailable("15.14 training labels need out-of-fold V, and this ev4_03 run exports none "
                              f"(frozen_manifest.json has no 'oof_v' block and there is no {OOF_BLOCK_FILE} next to "
                              "it).  Refusing: an in-sample V would leak the 15.14 outcomes into the SVI training "
                              "labels.\n" + OOF_SPEC)
    d = Path(blk["dir"])
    mp = d / "oof_manifest.json"
    if not mp.is_file() or VM.sha256_file(mp) != blk["manifest_sha256"]:
        raise OOFNotAvailable(f"{mp} is missing or does not match frozen_manifest oof_v.manifest_sha256\n" + OOF_SPEC)
    om = _json_load(mp)
    problems = []
    if om.get("format") != OOF_FORMAT:
        problems.append(f"format {om.get('format')!r}")
    if om.get("train_patch") != TRAIN_PATCH:
        problems.append(f"train_patch {om.get('train_patch')!r}")
    if om.get("kind") != vf.get("kind") or blk.get("kind") != vf.get("kind"):
        problems.append(f"kind {om.get('kind')!r} / {blk.get('kind')!r} != chosen {vf.get('kind')!r}")
    if int(om.get("n_folds", -1)) != N_FOLDS or om.get("fold_rule") != OOF_FOLD_RULE:
        problems.append("n_folds / fold_rule differ from the spec")
    if om.get("frozen_bundle_sha256") != vf.get("bundle_sha256"):
        problems.append("frozen_bundle_sha256 is not this run's V_frozen")
    if om.get("pilot"):
        problems.append("pilot run")
    if om.get("smoke") and not smoke:
        problems.append("smoke OOF set without --smoke")
    folds = om.get("folds") or {}
    if sorted(folds) != [str(k) for k in range(N_FOLDS)]:
        problems.append(f"folds {sorted(folds)}")
    if problems:
        raise OOFNotAvailable(f"OOF export {mp} does not satisfy the spec: {problems}\n" + OOF_SPEC)
    out = {"dir": str(d), "manifest_sha256": blk["manifest_sha256"], "kind": om["kind"], "folds": {},
           "block_source": source, "sidecar_sha256": blk.get("sidecar_sha256")}
    for k in range(N_FOLDS):
        f = folds[str(k)]
        if int(f.get("heldout_fold", -1)) != k:
            raise OOFNotAvailable(f"fold {k}: heldout_fold {f.get('heldout_fold')}")
        tf = d / f["train_match_ids_file"]
        if not tf.is_file() or VM.sha256_file(tf) != f["train_match_ids_sha256"]:
            raise OOFNotAvailable(f"fold {k}: {tf} missing or sha256 mismatch")
        out["folds"][k] = {"dir": str(d / f["dir"]), "bundle_sha256": f["bundle_sha256"], "train_ids_file": str(tf),
                           "n_train_matches": f.get("n_train_matches")}
    return out


def fold_train_ids(oof: Mapping[str, Any]) -> Dict[int, set]:
    """{k: set(training match ids of fold k)}; every id must have cv_fold != k."""
    out = {}
    for k, f in oof["folds"].items():
        ids = {ln.strip() for ln in Path(f["train_ids_file"]).read_text(encoding="utf-8").splitlines() if ln.strip()}
        bad = [m for m in ids if VM.cv_fold(m) == int(k)]
        if bad:
            raise OOFNotAvailable(f"fold {k} trained on {len(bad)} matches of its own held-out fold ({sorted(bad)[:3]})")
        out[int(k)] = ids
    return out


def check_record1_v(record: Mapping[str, Any], bundle_sha256: str) -> Dict[str, Any]:
    """Held-out patches: the LOCKED record 1 (the dict returned by ev4_record_lock.assert_record1_locked) must name
    the frozen V bundle in its top-level 'V_frozen_bundle_sha256'.  No other key is read (a nested V block, a
    'V_frozen' block or a draft is never accepted)."""
    from gameplay.split_guard import SplitViolation
    if not isinstance(record, Mapping):
        raise SplitViolation("check_record1_v needs the parsed locked record 1 (assert_record1_locked), not a path")
    RL.check_locked_fields(record)                           # a draft / smoke / unsigned record is refused here too
    named = record.get("V_frozen_bundle_sha256")
    if not isinstance(named, str) or not named:
        raise SplitViolation("record 1 does not name the frozen V bundle (top-level 'V_frozen_bundle_sha256'); "
                             "held-out labels need it")
    if named != str(bundle_sha256).strip().lower():
        raise SplitViolation(f"record 1 names V bundle {named}, the V manifest has {bundle_sha256}")
    return {"record1_v_bundle_sha256": named, "ok": True}


def held_out_guard(patch: str, record1: Optional[str], record1_sha256: Optional[str], v_manifest: Path, smoke: bool,
                   records_dir: Optional[Path] = None, predecisions_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Decision L10 for a held-out patch, run right after EX.check_patch_access and BEFORE any held-out manifest or
    data is read: record 1 must be the author-locked record (ev4_record_lock.assert_record1_locked), and its top-level
    V_frozen_bundle_sha256 must be the frozen V of --v-manifest (a 15.14 / 15.15 artefact).  Returns the block that
    resolve_v and the labels manifest carry."""
    rec = RL.assert_record1_locked(record1, record1_sha256, records_dir=records_dir,
                                   predecisions_sha256=predecisions_sha256)
    vman, _vpath = read_v_manifest(v_manifest)
    vf = VM.assert_v_usable(vman, allow_smoke=smoke)
    chk = check_record1_v(rec, vf["bundle_sha256"])
    return {"record1": str(record1), "record1_sha256": str(record1_sha256).strip().lower(), "locked": True,
            "signed_by": (rec.get("author_signoff") or {}).get("signed_by"), **chk}


def resolve_v(patch: str, v_manifest: Path, extract_manifest_sha: str, smoke: bool,
              record1_v: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The V source for `patch` (decision L7) as a picklable spec for the workers; nothing is loaded here except
    manifests and (for 15.14) the fold training id lists.  Held-out patches need record1_v = held_out_guard(...)."""
    from gameplay.split_guard import SELECTION_PATCHES, SplitViolation
    from gameplay.state_value_v3 import STATE_V3_NAME_HASH
    vman, vpath = read_v_manifest(v_manifest)
    vf = VM.assert_v_usable(vman, allow_smoke=smoke)
    if vman.get("state_v3_name_hash") != STATE_V3_NAME_HASH:
        raise RuntimeError(f"V was fitted on StateV3 {vman.get('state_v3_name_hash')}, code has {STATE_V3_NAME_HASH}")
    structure = VM.read_bundle_structure(vf["dir"], vf["bundle_sha256"])      # format 1 or 2; side / recalibration
    for flag in ("side_marker", "recalibrated"):
        if flag in vf and bool(vf[flag]) != bool(structure[flag]):
            raise RuntimeError(f"V manifest V_frozen.{flag} = {vf[flag]}, V_frozen/bundle.json has {structure[flag]}")
    spec = {"manifest": str(vpath), "manifest_sha256": VM.sha256_file(vpath), "frozen": dict(vf),
            "smoke": bool(vman.get("smoke")), "chosen": vman.get("chosen"),
            "v_structure": {"bundle_format": structure["bundle_format"], "side_marker": structure["side_marker"],
                            "recalibrated": structure["recalibrated"],
                            "predict": "V.predict(StateV3 rows): side = +1 (blue perspective)"
                                       + ("; recalibration from snapshot_age_s and time_minutes"
                                          if structure["recalibrated"] else "")}}
    if patch in (TRAIN_PATCH, SELECT_PATCH):
        which = "train" if patch == TRAIN_PATCH else "select"
        want = ((vman.get("inputs") or {}).get(which) or {}).get("manifest_sha256")
        if want != extract_manifest_sha:
            raise RuntimeError(f"V was fitted with a {patch} extract manifest {want}, labelling extract manifest "
                               f"{extract_manifest_sha}: labels must use the extract V was built on")
    else:
        if patch in SELECTION_PATCHES or not isinstance(record1_v, Mapping) or not record1_v.get("ok") or \
                record1_v.get("record1_v_bundle_sha256") != vf["bundle_sha256"]:
            raise SplitViolation(f"held-out patch {patch}: V needs the locked record 1 check (held_out_guard) naming "
                                 f"this V bundle")
        spec["record1_v"] = dict(record1_v)
    if patch == TRAIN_PATCH:
        oof = check_oof(vman, vf, smoke, vpath=vpath)
        for k, f in oof["folds"].items():                  # the OOF folds must reproduce the frozen V's structure
            fs = VM.read_bundle_structure(f["dir"], f["bundle_sha256"])
            if fs["side_marker"] != structure["side_marker"] or fs["recalibrated"] != structure["recalibrated"]:
                raise OOFNotAvailable(f"OOF fold {k} bundle has side_marker={fs['side_marker']} / recalibrated="
                                      f"{fs['recalibrated']}, the frozen V has {structure['side_marker']} / "
                                      f"{structure['recalibrated']}")
        spec.update(source="oof", oof=oof)
    else:
        spec.update(source="frozen")
    return spec


class VPredictor:
    """Worker-side V: the frozen V, or the 5 OOF fold models keyed by cv_fold(match)."""

    def __init__(self, spec: Mapping[str, Any], threads: int = 1):
        from gameplay.state_value_v3 import STATE_V3_NAME_HASH
        self.source = spec["source"]
        if self.source == "frozen":
            vf = spec["frozen"]
            self.frozen = VM.load_v(vf["dir"], expected_sha256=vf["bundle_sha256"],
                                    expected_columns_hash=STATE_V3_NAME_HASH)
            self.folds = None
        else:
            self.frozen = None
            self.folds = {int(k): VM.load_v(f["dir"], expected_sha256=f["bundle_sha256"],
                                            expected_columns_hash=STATE_V3_NAME_HASH)
                          for k, f in spec["oof"]["folds"].items()}
            kinds = {v.kind for v in self.folds.values()}
            if kinds != {spec["oof"]["kind"]}:
                raise OOFNotAvailable(f"fold bundle kinds {kinds} != {spec['oof']['kind']}")
            self.train_ids = fold_train_ids(spec["oof"])
        for v in ([self.frozen] if self.frozen is not None else list(self.folds.values())):
            if hasattr(v.model, "threads"):                 # lgbm / mlp: threads per worker (4 CPUs in total)
                v.model.threads = max(1, min(VM.MAX_THREADS, int(threads)))

    def coverage(self, mids: Sequence[str]) -> Dict[str, int]:
        """OOF only: labelled matches that are in the training lists of all 4 other folds (full 15.14 fold models)."""
        if self.source == "frozen":
            return {"n": 0, "full": 0}
        u = sorted(set(mids))
        full = sum(1 for m in u if all(m in self.train_ids[k] for k in self.train_ids if k != VM.cv_fold(m)))
        return {"n": len(u), "full": int(full)}

    def fold_of(self, mids: Sequence[str]) -> np.ndarray:
        if self.source == "frozen":
            return np.full(len(mids), -1, dtype=np.int8)
        return VM.per_match(list(mids), VM.cv_fold).astype(np.int8)

    def predict(self, X: np.ndarray, mids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """(P(blue wins), fold used (-1 = frozen)) for the rows of X."""
        folds = self.fold_of(mids)
        if not len(X):
            return np.zeros(0), folds
        if self.source == "frozen":
            return self.frozen.predict(X), folds
        out = np.full(len(X), np.nan)
        for k in np.unique(folds):
            sel = folds == k
            leaked = sorted(set(np.asarray(mids)[sel]) & self.train_ids[int(k)])
            if leaked:
                raise OOFNotAvailable(f"fold {k} model was trained on labelled matches {leaked[:3]}")
            out[sel] = self.folds[int(k)].predict(X[sel])
        return out, folds


# ============================================================================ inputs
def read_extract_manifest(d: Path) -> Dict[str, Any]:
    p = Path(d) / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"extract manifest missing: {p}")
    return _json_load(p)


def check_extract(man: Mapping[str, Any], patch: str, guard: Mapping[str, Any], smoke: bool,
                  allow_code_drift: bool) -> Dict[str, Any]:
    """Patch, StateV3 hash, sample flag, remake rule, preset params, code drift (decision L10)."""
    from gameplay.split_guard import normalize_patch
    from gameplay.state_value_v3 import STATE_V3_NAME_HASH, STATE_VERSION
    if normalize_patch(str(man.get("patch"))) != patch:
        raise RuntimeError(f"extract is patch {man.get('patch')}, --patch is {patch}")
    if man.get("STATE_V3_NAME_HASH") != STATE_V3_NAME_HASH or man.get("state_version") != STATE_VERSION:
        raise RuntimeError("extract StateV3 hash / version differ from the code")
    if man.get("sample") and not smoke:
        raise SystemExit("extract is a sample run (manifest sample=true); only --smoke may use it")
    r = man.get("remakes") or {}
    if r.get("threshold_ms") != EC.REMAKE_MAX_GAME_END_MS:
        raise RuntimeError(f"extract was not built with the remake rule: {r or None}")
    want = json.loads(json.dumps(guard["params"], default=EX.json_default))
    if (man.get("guards") or {}).get("params") != want or \
            (man.get("guards") or {}).get("record1a_sha256") != guard["record1a_sha256"]:
        raise RuntimeError("extract was built with other params / record 1A than the locked preset")
    cur = EX.code_hashes()
    got = man.get("code_sha256") or {}
    drift = {f: {"extract": got.get(f), "current": cur.get(f)} for f in sorted(set(got) | set(cur))
             if got.get(f) != cur.get(f)}
    if drift and not allow_code_drift:
        raise RuntimeError(f"extract code differs from the current code (the post states would be built with other "
                           f"code than the pre states): {json.dumps(drift)[:600]}; re-extract, or pass "
                           "--allow-code-drift after checking the change cannot affect the states")
    if drift:
        log(f"WARNING --allow-code-drift: {sorted(drift)}")
    det = man.get("detect_input") or {}
    if det.get("mode") != "parquet":
        raise RuntimeError("extract was built with --inline-detect; labels need the ev4_01 detect table (kill_idx)")
    return {"drift": drift, "drift_found": bool(drift), "allow_code_drift": bool(allow_code_drift),
            "detect_input": {k: det.get(k) for k in ("path", "sha256", "rows")}}


DETECT_COLUMNS = ("match_id", "eng_idx", "tau", "first_kill_ts", "last_kill_ts", "kill_idx", "kill_ts", "cohort",
                  "clean", "isolated", "centroid_x", "centroid_y")


def resolve_detect(man: Mapping[str, Any], detect: Optional[Path]) -> Tuple[Path, Dict[str, str]]:
    """The ev4_01 detect table the extract was built from (sha256 must equal the extract manifest's)."""
    det = man.get("detect_input") or {}
    want = det.get("sha256") or {}
    p = Path(detect) if detect is not None else Path(str(det.get("path")))
    got = EX.detect_files_sha256(p)
    if sorted(got.values()) != sorted(want.values()):
        raise RuntimeError(f"detect table {p} does not hash to the extract manifest's detect_input.sha256")
    return p, got


# ============================================================================ per chunk
def _read_detect_chunk(path: Path, patch: str, mids: Sequence[str]) -> Dict[Tuple[str, int], List[dict]]:
    import pyarrow.parquet as pq
    names = set(pq.ParquetDataset(path).schema.names)
    miss = [c for c in DETECT_COLUMNS if c not in names]
    if miss:
        raise KeyError(f"detect table lacks columns {miss}")
    cols = list(DETECT_COLUMNS) + (["patch"] if "patch" in names else [])
    tab = pq.read_table(path, columns=cols, filters=[("match_id", "in", list(set(mids)))]).to_pandas()
    if "patch" in tab.columns and set(tab["patch"].astype(str)) - {patch}:
        raise RuntimeError("detect table has rows of other patches")
    out: Dict[Tuple[str, int], List[dict]] = {}
    for r in tab.to_dict("records"):
        out.setdefault((str(r["match_id"]), int(r["tau"])), []).append(r)
    return out


EXCHANGE_EVENT_TYPES = ("CHAMPION_KILL", "LEVEL_UP", "TURRET_PLATE_DESTROYED", "BUILDING_KILL", "ELITE_MONSTER_KILL",
                        "WARD_KILL")


def event_views(events: Sequence[Mapping]) -> Dict[str, List[Mapping]]:
    """Per-match event subsets that give the labels_exact functions exactly the events they read (speed only; the
    full-event results are compared on the spot-check engagements, decision L2):
      kills     raw CHAMPION_KILL events in labels_exact.raw_kill_events order (kill_idx indexes this list;
                label_endpoint reads only these)
      exchange  CHAMPION_KILL, LEVEL_UP (death timers), TURRET_PLATE_DESTROYED, BUILDING_KILL, ELITE_MONSTER_KILL,
                WARD_KILL (exchange_outcome, first_tower_ts, event_survival.death_intervals)
      elite     ELITE_MONSTER_KILL (next_objective_outcome with game_end given)"""
    from gameplay.labels_exact import raw_kill_events
    ev = [e for e in events if isinstance(e, Mapping)]
    return {"kills": raw_kill_events(ev),
            "exchange": [e for e in ev if str(e.get("type", e.get("eventType", ""))).upper() in EXCHANGE_EVENT_TYPES],
            "elite": [e for e in ev if str(e.get("type", e.get("eventType", ""))).upper() == "ELITE_MONSTER_KILL"]}


def full_event_views(events: Sequence[Mapping]) -> Dict[str, List[Mapping]]:
    """The unreduced views (every function sees every event): the reference for the spot check."""
    from gameplay.labels_exact import raw_kill_events
    ev = list(events)
    return {"kills": raw_kill_events(ev), "exchange": ev, "elite": ev}


def label_engagement(row: Mapping[str, Any], det: Mapping[str, Any], views: Mapping[str, List[Mapping]],
                     ts: np.ndarray, builder, tm: Mapping[int, int], patch: str, prices: Mapping[str, float],
                     radius: float, R,
                     post_cache: Dict[int, Tuple[Optional[np.ndarray], int, str]]) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    """Endpoints, validity, post states and E5 labels of one engagement (no V here).  Returns (row dict, {h: post
    float32 vector} for valid endpoints)."""
    from gameplay.labels_exact import exchange_outcome, kill_difference, label_endpoint, next_objective_outcome
    mid, tau, L = str(row["match_id"]), int(row["tau"]), int(row["last_kill_ts"])
    ge = int(row["game_end"])
    ns = None if int(row["next_start"]) < 0 else int(row["next_start"])
    kill_idx = sorted(int(i) for i in det["kill_idx"])
    allk = views["kills"]
    if not kill_idx or kill_idx[-1] >= len(allk):
        raise RuntimeError(f"{mid}@{tau}: kill_idx {kill_idx} outside the match's {len(allk)} kills")
    own = [allk[i] for i in kill_idx]                       # == labels_exact.own_kill_events(kill_idx)
    if [int(k["timestamp"]) for k in own] != [int(x) for x in det["kill_ts"]]:
        raise RuntimeError(f"{mid}@{tau}: kill_idx does not reproduce the detect kill_ts")
    if int(own[-1]["timestamp"]) != L or int(own[0]["timestamp"]) != int(row["first_kill_ts"]):
        raise RuntimeError(f"{mid}@{tau}: own kills disagree with first / last kill")
    eng = {"tau": tau, "last_kill_ts": L, "kills": own}
    cx = det.get("centroid_x")
    if cx is not None and not (isinstance(cx, float) and math.isnan(cx)):
        eng.update(centroid_x=float(cx), centroid_y=float(det["centroid_y"]))
    support_start, last_frame = max(0, int(ts[0])), int(ts[-1])
    kd = kill_difference(own, tm)
    out: Dict[str, Any] = {"eng_idx": int(det["eng_idx"]), "kill_diff": int(kd), "n_own_kills": len(own),
                           "n_executions": int(sum(1 for k in own if int(k.get("killerId", 0) or 0) == 0)),
                           "kill_diff_label": 1 if kd > 0 else 0 if kd < 0 else None}
    posts: Dict[int, np.ndarray] = {}
    for h in HORIZONS_S:
        e, reasons = label_endpoint(L, views["kills"], h_s=h, next_start=ns, game_end=ge)
        if h == PRIMARY_H and int(e) != int(row["e90"]):
            raise RuntimeError(f"{mid}@{tau}: recomputed e90 {e} != extract e90 {row['e90']}")
        val = R.endpoint_validity(e, L, tau - 1, support_start, last_frame)
        inval = [k for k, ok in val.items() if not ok]
        if ns is not None and ns <= L:
            inval.insert(0, "same_match_overlap_next_start_le_L")
        snap = age = -1
        if not inval:
            hit = post_cache.get(int(e))
            if hit is None:
                try:
                    st = builder.at(int(e))
                    if st.snapshot_ms > e:
                        raise ValueError("future snapshot")
                    hit = (_vec32(st), int(st.snapshot_ms), "")
                except Exception as exc:  # noqa: BLE001 - a failed post state blocks the row, never silently
                    hit = (None, -1, f"post_state_error:{type(exc).__name__}:{exc}"[:160])
                post_cache[int(e)] = hit
            vec, snap, err = hit
            if err:
                inval.append(err)
            else:
                posts[h] = vec
                age = int(e) - snap
        out.update({f"e_h{h}": int(e), f"e_h{h}_reasons": "|".join(reasons), f"valid_h{h}": int(not inval),
                    f"invalid_h{h}": "|".join(inval), f"post_snapshot_ms_h{h}": int(snap),
                    f"post_frame_age_ms_h{h}": int(age)})
        # E5 outcome labels (event-only; independent of the SVI validity; need e_h >= L)
        if e >= L:
            xo = exchange_outcome(eng, views["exchange"], prices, int(e), radius=radius, team_map=tm, patch=patch)
            out.update({f"exchange_label_h{h}": xo["label"], f"exchange_decided_by_h{h}": xo["decided_by"],
                        f"exchange_gold_h{h}": float(xo["gold_diff"])})
            no = next_objective_outcome(views["elite"], int(e), team_map=tm, game_end=ge)
        else:
            out.update({f"exchange_label_h{h}": None, f"exchange_decided_by_h{h}": "endpoint_before_L",
                        f"exchange_gold_h{h}": float("nan")})
            no = "endpoint_before_L"
        out.update({f"next_obj_outcome_h{h}": no,
                    f"next_obj_label_h{h}": 1 if no == "Blue" else 0 if no == "Red" else None})
    return out, posts


def _same_labels(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


INT8_NULLABLE = tuple([f"svi_h{h}" for h in HORIZONS_S] + ["svi", "kill_diff_label"]
                      + [f"exchange_label_h{h}" for h in HORIZONS_S] + [f"next_obj_label_h{h}" for h in HORIZONS_S])
KEY_COLUMNS = ("match_id", "eng_idx", "tau", "first_kill_ts", "last_kill_ts", "patch", "extract_chunk", "extract_row")
PASS_COLUMNS = ("cohort", "cohort_code", "clean", "isolated", "t5", "n_kills", "n_blue", "n_red", "n_min",
                "frame_age_ms", "game_end", "next_start", "e90")


def label_columns() -> List[str]:
    cols = list(KEY_COLUMNS) + list(PASS_COLUMNS) + ["v_source", "v_fold", "p_pre"]
    for h in HORIZONS_S:
        cols += [f"e_h{h}", f"e_h{h}_reasons", f"valid_h{h}", f"invalid_h{h}", f"post_snapshot_ms_h{h}",
                 f"post_frame_age_ms_h{h}", f"p_post_h{h}", f"dv_h{h}", f"svi_h{h}", f"svi_zero_h{h}"]
    cols += ["svi", "valid", "kill_diff", "n_own_kills", "n_executions", "kill_diff_label"]
    for h in HORIZONS_S:
        cols += [f"exchange_label_h{h}", f"exchange_decided_by_h{h}", f"exchange_gold_h{h}",
                 f"next_obj_outcome_h{h}", f"next_obj_label_h{h}"]
    return cols


LABEL_COLUMNS = tuple(label_columns())


def attach_v(rows: List[Dict[str, Any]], eng_X: np.ndarray, post_X: Sequence[np.ndarray],
             post_ref: Sequence[Tuple[int, int]], V, source: str, R) -> None:
    """Decisions L2 / L5 / L6 in place: p_pre = V(eng_X[extract_row]), p_post_h = V(post state), dv_h, svi_h (DR
    label_from_delta: 1[dv > 0], exact zero -> 0 with svi_zero_h = 1); invalid endpoints keep NaN / <NA>.
    V.predict(X, mids) -> (p, fold); post_ref[j] = (row position, h) of post_X[j]."""
    mids = [r["match_id"] for r in rows]
    Xpre = eng_X[[r["extract_row"] for r in rows]] if rows else np.zeros((0, eng_X.shape[1]), np.float32)
    p_pre, folds = V.predict(Xpre, mids)
    pp = V.predict(np.stack(post_X), [mids[i] for i, _ in post_ref])[0] if len(post_X) else np.zeros(0)
    if len(pp) != len(post_ref) or sum(int(r[f"valid_h{h}"]) for r in rows for h in HORIZONS_S) != len(post_ref):
        raise RuntimeError("post states do not match the valid endpoints")
    for i, r in enumerate(rows):
        r.update(v_source=source, v_fold=int(folds[i]), p_pre=float(p_pre[i]))
        for h in HORIZONS_S:
            r.update({f"p_post_h{h}": float("nan"), f"dv_h{h}": float("nan"), f"svi_h{h}": None, f"svi_zero_h{h}": 0})
    for (i, h), p1 in zip(post_ref, pp):
        if not rows[i][f"valid_h{h}"]:
            raise RuntimeError("post state for an invalid endpoint")
        dv = float(p1) - rows[i]["p_pre"]
        rows[i].update({f"p_post_h{h}": float(p1), f"dv_h{h}": dv, f"svi_h{h}": int(R.label_from_delta(dv)),
                        f"svi_zero_h{h}": int(dv == 0.0)})
    for r in rows:
        r["svi"], r["valid"] = r[f"svi_h{PRIMARY_H}"], r[f"valid_h{PRIMARY_H}"]


def label_chunk(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Worker entry: one extract chunk -> one parts parquet + sidecar; everything inside forbid_grid()."""
    from gameplay.grid_guard import forbid_grid
    with forbid_grid():
        return _label_chunk_inner(task)


def _label_chunk_inner(task: Mapping[str, Any]) -> Dict[str, Any]:
    import pandas as pd
    from gameplay.champion_attributes import load_champion_table_v2
    from gameplay.item_state import load_item_table_v2
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, StateBuilderV3
    t0 = time.time()
    patch, cid = task["patch"], task["chunk"]
    ed = Path(task["extract_dir"])
    stem = f"chunk_{int(cid):05d}"
    for fn, f in task["files"].items():
        if VM.sha256_file(ed / fn) != f["sha256"]:
            raise RuntimeError(f"{ed / fn} does not match its sha256")
    params = EX.verify_params(task["params"])
    radius = float(params.diameter)
    R = dr_rules()
    V = VPredictor(task["v_spec"], threads=int(task.get("v_threads", 1)))
    items_table, champion_table = load_item_table_v2(patch), load_champion_table_v2(patch)
    prices = dict(task["prices"])
    with np.load(ed / f"{stem}.npz") as z:
        if tuple(z["state_columns"].tolist()) != STATE_V3_COLUMNS:
            raise RuntimeError(f"{stem}: state columns differ from STATE_V3_COLUMNS")
        eng_X = np.asarray(z["eng_X"])
    eng = pd.read_parquet(ed / f"{stem}_eng.parquet")
    if len(eng) != len(eng_X) or not np.array_equal(eng["row"].to_numpy(), np.arange(len(eng))):
        raise RuntimeError(f"{stem}: engagement index and eng_X disagree")
    if len(eng) and (eng["game_end"].to_numpy() < EC.REMAKE_MAX_GAME_END_MS).any():
        raise RuntimeError(f"{stem}: engagement rows of remake matches (GAME_END < {EC.REMAKE_MAX_GAME_END_MS})")
    if len(eng) and (eng["isolated"].to_numpy() != 1).any():
        raise RuntimeError(f"{stem}: non-isolated engagement rows in the extract")
    mids = list(dict.fromkeys(eng["match_id"].astype(str)))
    det = _read_detect_chunk(Path(task["detect_path"]), patch, mids) if mids else {}
    rows: List[Dict[str, Any]] = []
    post_X: List[np.ndarray] = []
    post_ref: List[Tuple[int, int]] = []                     # (row position, h)
    spots: List[Dict[str, Any]] = []
    status = Counter()
    for mid, g in eng.groupby(eng["match_id"].astype(str), sort=False):
        pack = EX.load_pack(mid, patch, Path(task["cache"]))
        events = pack["events"]
        ge = EX.game_end_of(events)
        if ge is None or EC.is_remake(ge):
            raise RuntimeError(f"{mid}: GAME_END {ge} (remake or missing) in the extract")
        b = StateBuilderV3(pack, patch, items_table=items_table, champion_table=champion_table)
        tm = b.tm
        ts = pack["minute_ts"]
        views = event_views(events)
        post_cache: Dict[int, Tuple[Optional[np.ndarray], int, str]] = {}
        for j, r in enumerate(g.to_dict("records")):
            if int(r["game_end"]) != int(ge):
                raise RuntimeError(f"{mid}: extract game_end {r['game_end']} != events GAME_END {ge}")
            hits = det.get((mid, int(r["tau"])), [])
            if len(hits) != 1:
                raise RuntimeError(f"{mid}@{r['tau']}: {len(hits)} detect rows for the extract engagement")
            d = hits[0]
            for c in ("first_kill_ts", "last_kill_ts", "clean", "isolated"):
                if int(d[c]) != int(r[c]):
                    raise RuntimeError(f"{mid}@{r['tau']}: detect {c} {d[c]} != extract {r[c]}")
            if str(d["cohort"]) != str(r["cohort"]):
                raise RuntimeError(f"{mid}@{r['tau']}: detect cohort {d['cohort']} != extract {r['cohort']}")
            lab, posts = label_engagement(r, d, views, ts, b, tm, patch, prices, radius, R, post_cache)
            if j == 0 and len(spots) < SPOT_CHECK_N:
                # decision L2: pre state rebuilt == stored eng_X; reduced event views == full events
                pre = _vec32(b.at(int(r["tau"]) - 1))
                ref, ref_posts = label_engagement(r, d, full_event_views(events), ts, b, tm, patch, prices, radius, R,
                                                  {})
                spots.append({"match_id": mid, "tau": int(r["tau"]),
                              "equal": _nan_equal(pre, eng_X[int(r["row"])].astype(np.float32)),
                              "views_equal": _same_labels(lab, ref) and sorted(posts) == sorted(ref_posts) and all(
                                  _nan_equal(posts[h], ref_posts[h]) for h in posts)})
            out = {"match_id": mid, "tau": int(r["tau"]), "first_kill_ts": int(r["first_kill_ts"]),
                   "last_kill_ts": int(r["last_kill_ts"]), "patch": patch, "extract_chunk": int(cid),
                   "extract_row": int(r["row"])}
            for c in PASS_COLUMNS:
                if c in r and r[c] is not None and not (isinstance(r[c], float) and math.isnan(r[c])):
                    out[c] = r[c]
            out.update(lab)
            pos = len(rows)
            rows.append(out)
            for h, vec in posts.items():
                post_X.append(vec)
                post_ref.append((pos, h))
            status["rows"] += 1
        del pack, b
    bad = [s for s in spots if not (s["equal"] and s["views_equal"])]
    if bad:
        raise RuntimeError(f"spot check failed (rebuilt StateV3(tau-1) != eng_X, or reduced event views != full "
                           f"events): {bad[:3]}")

    attach_v(rows, eng_X, post_X, post_ref, V, task["v_spec"]["source"], R)
    df = pd.DataFrame(rows, columns=list(LABEL_COLUMNS)) if rows else pd.DataFrame({c: [] for c in LABEL_COLUMNS})
    for c in INT8_NULLABLE:
        df[c] = pd.array(df[c].tolist(), dtype="Int8")
    out_dir = Path(task["parts_dir"])
    part = out_dir / f"part_{int(cid):05d}.parquet"
    tmp = part.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, part)
    side = {"chunk": int(cid), "plan_hash": task["plan_hash"], "rows": int(len(df)), "n_matches": len(mids),
            "oof_coverage": V.coverage(mids),
            "file": part.name, "sha256": VM.sha256_file(part), "spot_check": {"n": len(spots), "failures": len(bad),
                                                                               "cases": spots},
            "n_post_states": int(len(post_X)), "seconds": round(time.time() - t0, 2)}
    EX.write_json_atomic(out_dir / f"part_{int(cid):05d}.json", side)
    return side


def part_is_done(parts_dir: Path, cid: int, plan_hash: str) -> Optional[Dict[str, Any]]:
    sp = parts_dir / f"part_{int(cid):05d}.json"
    if not sp.is_file():
        return None
    side = _json_load(sp)
    p = parts_dir / str(side.get("file", ""))
    if side.get("plan_hash") != plan_hash or not p.is_file() or VM.sha256_file(p) != side.get("sha256"):
        return None
    return side


# ============================================================================ summary
def summarise(df) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df)), "matches": int(df["match_id"].nunique()) if len(df) else 0}
    by: Dict[str, Any] = {}
    for coh, g in list(df.groupby("cohort")) + [("ALL", df)]:
        ent: Dict[str, Any] = {"rows": int(len(g)), "clean": int(g["clean"].sum())}
        for h in HORIZONS_S:
            v = g[f"valid_h{h}"] == 1
            s = g.loc[v, f"svi_h{h}"].astype("float")
            ent[f"h{h}"] = {"valid": int(v.sum()), "svi_mean": float(s.mean()) if len(s) else None,
                            "svi_zero": int(g[f"svi_zero_h{h}"].sum()),
                            "exchange_labelled": int(g[f"exchange_label_h{h}"].notna().sum()),
                            "next_obj_labelled": int(g[f"next_obj_label_h{h}"].notna().sum())}
            if len(s):
                sv, ex = g.loc[v, f"svi_h{h}"], g.loc[v, f"exchange_label_h{h}"]
                both = sv.notna() & ex.notna()
                ent[f"h{h}"]["svi_eq_exchange"] = float((sv[both] == ex[both]).mean()) if both.any() else None
                kd = g.loc[v, "kill_diff_label"]
                both = sv.notna() & kd.notna()
                ent[f"h{h}"]["svi_eq_kill_diff"] = float((sv[both] == kd[both]).mean()) if both.any() else None
        ent["kill_diff_labelled"] = int(g["kill_diff_label"].notna().sum())
        by[str(coh)] = ent
    out["by_cohort"] = by
    out["invalid_reasons"] = {f"h{h}": dict(Counter(r for s in df[f"invalid_h{h}"].astype(str) if s
                                                    for r in s.split("|"))) for h in HORIZONS_S}
    out["exchange_decided_by"] = {f"h{h}": {str(k): int(v) for k, v in
                                            df[f"exchange_decided_by_h{h}"].value_counts().items()} for h in HORIZONS_S}
    out["next_obj_outcome"] = {f"h{h}": {str(k): int(v) for k, v in
                                         df[f"next_obj_outcome_h{h}"].value_counts().items()} for h in HORIZONS_S}
    out["v_fold_counts"] = {str(k): int(v) for k, v in df["v_fold"].value_counts().sort_index().items()}
    out["p_pre_quantiles"] = ({str(q): float(df["p_pre"].quantile(q)) for q in (0.05, 0.5, 0.95)} if len(df) else {})
    return out


# ============================================================================ labels main
def run_labels(args: argparse.Namespace) -> Dict[str, Any]:
    t_start = time.time()
    predec = EC.predecisions_info()                         # refuse a missing / edited author record first
    access = EX.check_patch_access(args.patch, args.record1, args.record1_sha256)   # split guard before any data
    patch = access["patch"]
    record1_v = None
    if not access.get("selection_patch"):
        # held out: the locked record 1 and its V bundle, before any held-out manifest is read
        record1_v = held_out_guard(patch, args.record1, args.record1_sha256, args.v_manifest, args.smoke)
        access["record1_locked"] = record1_v
    if args.limit_chunks is not None and not args.smoke:
        raise SystemExit("--limit-chunks is a smoke option (--smoke)")
    workers = max(1, min(MAX_WORKERS, int(args.workers)))
    _, params, guard = EX.load_params()                     # preset lock
    dr_rules()                                              # DR horizons == (60, 90, 120), primary 90
    ed = Path(args.extract)
    man = read_extract_manifest(ed)
    xcheck = check_extract(man, patch, guard, args.smoke, args.allow_code_drift)
    man_sha = VM.sha256_file(ed / "manifest.json")
    v_spec = resolve_v(patch, args.v_manifest, man_sha, args.smoke, record1_v)
    prices, price_info = load_prices(args.prices, args.smoke)
    detect_path, detect_sha = resolve_detect(man, args.detect)
    out = Path(args.out) if args.out is not None else (DEFAULT_OUT_SMOKE if args.smoke else DEFAULT_OUT)
    parts_dir = out / f"parts_{patch}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    cids = sorted((man.get("chunks") or {}).keys())
    if args.limit_chunks is not None:
        cids = cids[: int(args.limit_chunks)]
    vsha = ({"frozen": v_spec["frozen"]["bundle_sha256"]} if v_spec["source"] == "frozen" else
            {str(k): f["bundle_sha256"] for k, f in v_spec["oof"]["folds"].items()})
    plan = {"patch": patch, "extract_manifest_sha256": man_sha, "detect_sha256": detect_sha, "v_source": v_spec["source"],
            "v_bundles": vsha, "prices_sha256": price_info["sha256"], "code": code_hashes(), "decisions": DECISIONS,
            "horizons": HORIZONS_S, "remake_threshold_ms": EC.REMAKE_MAX_GAME_END_MS}
    plan_hash = EX.sha256_text(json.dumps(plan, sort_keys=True, default=EX.json_default))
    tasks, sides = [], {}
    for c in cids:
        done = part_is_done(parts_dir, int(c), plan_hash)
        if done is not None:
            sides[int(c)] = done
            continue
        tasks.append({"patch": patch, "chunk": int(c), "extract_dir": str(ed), "files": man["chunks"][c]["files"],
                      "params": guard["params"], "v_spec": v_spec, "prices": prices, "detect_path": str(detect_path),
                      "cache": str(args.cache), "parts_dir": str(parts_dir), "plan_hash": plan_hash,
                      "v_threads": max(1, VM.MAX_THREADS // workers)})
    log(f"patch {patch}: {len(cids)} chunks ({len(tasks)} to run), V source {v_spec['source']}, workers {workers}, "
        f"out {out}")
    t_run = time.time()
    if workers == 1 or len(tasks) <= 1:
        for tk in tasks:
            s = label_chunk(tk)
            sides[s["chunk"]] = s
            log(f"chunk {s['chunk']}: {s['rows']} rows, {s['seconds']} s")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(label_chunk, tk): tk["chunk"] for tk in tasks}
            for f in as_completed(futs):
                try:
                    s = f.result()
                except Exception:
                    for g in futs:
                        g.cancel()
                    log(f"chunk {futs[f]} FAILED")
                    raise
                sides[s["chunk"]] = s
                log(f"chunk {s['chunk']}: {s['rows']} rows, {s['seconds']} s")
    wall = time.time() - t_run
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = [pq.read_table(parts_dir / sides[int(c)]["file"]) for c in cids]
    tab = pa.concat_tables(tables) if tables else pa.table({c: [] for c in LABEL_COLUMNS})
    final = out / f"labels_{patch}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    pq.write_table(tab, tmp)
    os.replace(tmp, final)
    df = pd.read_parquet(final)
    n_expected = sum(int(man["chunks"][c]["rows"]["eng"]) for c in cids)
    if len(df) != n_expected:
        raise RuntimeError(f"labels rows {len(df)} != extract engagement rows {n_expected}")
    if df.duplicated(["match_id", "tau"]).any():
        raise RuntimeError("duplicate (match_id, tau) keys")
    if len(df) and (df["game_end"] < EC.REMAKE_MAX_GAME_END_MS).any():
        raise RuntimeError("remake rows in the labels")
    cov = {"n": sum((s.get("oof_coverage") or {}).get("n", 0) for s in sides.values()),
           "full": sum((s.get("oof_coverage") or {}).get("full", 0) for s in sides.values())}
    if v_spec["source"] == "oof" and not args.smoke and cov["full"] < 0.99 * cov["n"]:
        raise OOFNotAvailable(f"only {cov['full']} of {cov['n']} labelled 15.14 matches are in the training lists of "
                              "all 4 other folds: the OOF fold models were not fitted on the full 15.14 V rows")
    spot = {"n": sum(s["spot_check"]["n"] for s in sides.values()),
            "failures": sum(s["spot_check"]["failures"] for s in sides.values())}
    use = "train (q fit; OOF V)" if patch == TRAIN_PATCH else ("select (q selection)" if patch == SELECT_PATCH
                                                                 else "score (held out)")
    manifest = {
        "format": LABELS_FORMAT, "script": "scripts/exact_v4/ev4_04_labels.py",
        "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "patch": patch, "use": use,
        "smoke": bool(args.smoke), "limit_chunks": args.limit_chunks, "access": access,
        "outputs": {final.name: {"sha256": VM.sha256_file(final), "rows": int(len(df))}},
        "columns": list(LABEL_COLUMNS), "primary_h": PRIMARY_H, "horizons": list(HORIZONS_S),
        "inputs": {"extract": {"dir": str(ed), "manifest_sha256": man_sha, "sample": man.get("sample"),
                               "n_matches": man.get("n_matches"), "rows_eng": n_expected},
                   "detect": {"path": str(detect_path), "sha256": detect_sha},
                   "v": {k: v for k, v in v_spec.items() if k != "oof"} | (
                       {"oof": {"dir": v_spec["oof"]["dir"], "manifest_sha256": v_spec["oof"]["manifest_sha256"],
                                "kind": v_spec["oof"]["kind"], "bundles": vsha,
                                "block_source": v_spec["oof"].get("block_source"),
                                "sidecar_sha256": v_spec["oof"].get("sidecar_sha256")}}
                       if v_spec["source"] == "oof" else {}),
                   "prices": {k: v for k, v in price_info.items()}},
        "input_checks": {"extract": xcheck, "remake_rule": EC.remake_rule()},
        "guards": {"apply_preset": EX.PRESET, "require_locked": True, "forbid_grid": "every chunk worker",
                   "split_guard": "EX.check_patch_access before data", "params": guard["params"],
                   "record1a_sha256": guard["record1a_sha256"], "presets_py_sha256": guard["presets_py_sha256"]},
        "decisions": DECISIONS, "oof_spec": OOF_SPEC,
        "predecisions": {k: v for k, v in predec.items() if k != "content"},
        "code_sha256": code_hashes(), "git_head": EX.git_head(),
        "plan_hash": plan_hash, "spot_check_pre_state": spot, "oof_coverage": cov, "summary": summarise(df),
        "parts": {f"{c:05d}": {"file": s["file"], "sha256": s["sha256"], "rows": s["rows"], "seconds": s["seconds"]}
                  for c, s in sorted(sides.items())},
        "seconds_wall_this_invocation": round(wall, 1), "seconds_total": round(time.time() - t_start, 1),
        "worker_seconds_per_match": (sum(s["seconds"] for s in sides.values())
                                     / max(1, sum(s["n_matches"] for s in sides.values()))),
    }
    EX.write_json_atomic(out / f"labels_{patch}.json", manifest)
    log(f"wrote {final} ({len(df)} rows); summary ALL {manifest['summary']['by_cohort'].get('ALL')}")
    return manifest


# ============================================================================ CLI
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prices", help="OLS event prices on 15.14 only -> prices_1514.json")
    pr.add_argument("--matches", type=Path, default=DEFAULT_PRICE_MATCHES,
                    help="ev4_01 matches_15.14.parquet (status 'ok' rows are used)")
    pr.add_argument("--limit", type=int, default=None, help="first N matches by sha256(match_id) (smoke)")
    pr.add_argument("--out", type=Path, default=None)
    pr.add_argument("--workers", type=int, default=1)
    pr.add_argument("--n-boot", type=int, default=PRICE_BOOT)
    pr.add_argument("--cache", type=Path, default=CACHE)
    pr.add_argument("--smoke", action="store_true")
    lb = sub.add_parser("labels", help="labels_<patch>.parquet")
    lb.add_argument("--patch", required=True)
    lb.add_argument("--extract", type=Path, required=True, help="ev4_02 extract directory of the patch")
    lb.add_argument("--v-manifest", type=Path, required=True, help="ev4_03 frozen_manifest.json (or its directory)")
    lb.add_argument("--detect", type=Path, default=None, help="ev4_01 engagements parquet (default: extract manifest)")
    lb.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    lb.add_argument("--out", type=Path, default=None)
    lb.add_argument("--workers", type=int, default=MAX_WORKERS)
    lb.add_argument("--cache", type=Path, default=CACHE)
    lb.add_argument("--smoke", action="store_true", help="allow sample extracts, a smoke V and sample prices")
    lb.add_argument("--limit-chunks", type=int, default=None, help="first N extract chunks (smoke)")
    lb.add_argument("--allow-code-drift", action="store_true")
    lb.add_argument("--record1", default=None)
    lb.add_argument("--record1-sha256", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "prices":
        return run_prices(args)
    return run_labels(args)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
