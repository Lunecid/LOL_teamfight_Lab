"""v4-exact stage 2, step R5 (E4): fit the three V candidates on 15.14, select once on 15.15, calibrate, freeze.

Plan: outputs/diag_survival_dbscan_20260925/docs/REESTIMATION_PLAN_V4_EXACT_20260925.md, section 4 ("기본 승률 예측기 V
강화 (E4)"), "단계 2" (ev4_03_fit_v.py), "실행 순서" R5 and record 1 (V grid and choice).  Preset 'v4-exact'
(G = 14,000 ms, D = 4,300), locked in record 1A (outputs/reest_exact_v4_20260925/records/record1a_boundaries_20260925T114606Z.json).

V REVISION (author pre-specification, 2026-09-25T23:39:53Z, after the first full fit-V's martingale diagnostics and
BEFORE any refit: outputs/reest_exact_v4_20260925/records/v_revision_prespec_20260925T233953Z.json, sha256 pinned in
V_REVISION_SHA256; the whole record is copied into report_e4.json / frozen_manifest.json 'v_revision'):
  R1  Side marker (REVERSES the stage-2 pre-decision 'side marker not added'): V input = the 996 StateV3 columns + a
      column 'side' (last): +1 on original (blue-perspective) rows, -1 on team-swapped rows (logistic / LightGBM /
      gold baseline: the in-place 50 % swap of F3 also flips the sign of 'side'; MLP: per minibatch).  The side column
      is standardised with moments pooled over the swap (mean 0, sd 1).  15.15 rows are scored with side = +1, and
      V.predict() on a StateV3 matrix appends side = +1 (the blue perspective, P(blue wins)).  Same grids, budgets,
      folds, seeds and selection rule as before; bundle format ev4_v_bundle_2 (format-1 bundles still load).
  R2  Conditional recalibration of the chosen (side-marker, calibrated) V: TRIGGERED only if it fails martingale (a) in
      any stratum (band +-0.002; +-0.005 for recent deaths >= 3) or (b) primary (wild bootstrap) p <= 0.05.  Then a
      logistic recalibration is fitted on V_CAL (ev4_v_models.LogitRecalibrator: logit V; frame-age bins 0-15, 15-30,
      30-45, 45-60 s of snapshot_age_s as intercepts and interactions with logit V; natural cubic spline of
      time_minutes with 4 knots at the 5/35/65/95 % V_CAL quantiles; unpenalised ML) and (a) / (b) are re-run for it
      (b with the same bootstrap replicates).  ADOPTED if its V_SELECT log loss is not worse than the side-marker V's
      (calibrated) by more than 0.0005; otherwise the side-marker V is kept.  Both are reported (report_e4.json
      v_revision.recalibration) regardless.  The frozen V is the side-marker V or its recalibrated version; a
      recalibrated bundle applies the recalibration inside predict() from the state's snapshot_age_s and
      time_minutes.  No other change is made because of martingale results.  Technical (not in the record): an age
      bin with < 20 rows of a class makes the recalibration unidentifiable -> not fitted (reason reported), side-marker
      V kept (cannot happen at full size).
  The freeze now happens after the martingale checks (the recalibration decision needs them).

Implementation decisions made BEFORE any result (2026-09-25; recorded here, in DECISIONS and in every report):
  F1  V training rows = the ev4_02 V rows: ONE uniform random ms per 2-minute bucket per match (not one per minute,
      memory; ev4_02 decision D1).  Post-engagement states are NOT stored at extraction; they are computed at the
      label stage (ev4_04) after this V is frozen (ev4_02 decision D2).
  F2  Data.  Train = all 15.14 V rows.  15.15 V rows split by match: sha256('<match_id>:ev4_03:vsplit') even ->
      V_SELECT (50 %), odd -> V_CAL (50 %).  Features = all 996 StateV3 columns in STATE_V3_COLUMNS order (the V-only
      snapshot_age_s included).  NaN (champion fields of an id unknown to the patch table) -> the 15.14 column mean
      (stored in every bundle; counts reported).
  F3  Team symmetry.  logistic / LightGBM / gold-difference baseline: a seeded random 50 % of the 15.14 rows
      (numpy default_rng(20260925), Bernoulli 0.5 per row) is team-swapped (every blue_* / obj_blue_* column
      exchanged with its red partner) and its target flipped, instead of adding copies.  MLP: a fresh random 50 % of
      every minibatch is swapped.  15.15 rows are always scored in their natural orientation.
  F4  Tuning only by 15.14 match-grouped 5-fold CV (fold = sha256('<match_id>:ev4_03:cv5') mod 5); the choice within
      a family is the lowest pooled out-of-fold log loss (ties -> the earlier grid value).  Early stopping
      ('inner fold') = the 10 % of the training matches with sha256('<match_id>:ev4_03:inner') mod 10 == 0, inside
      each CV training fold and for the final fit (final LightGBM / MLP = all 15.14 matches minus the inner 10 %,
      stopped on it; final logistic = all 15.14 rows).  The MLP CV uses seed 0 only; the final MLP uses seeds 0, 1, 2
      with logits averaged.
  F5  FIXED grids (ev4_v_models.GRIDS):
        logistic  standardised (moments pooled over blue/red swap pairs), L2, C in {0.001, 0.01, 0.1}
        lgbm      num_leaves in {31, 63}; learning_rate 0.05, min_data_in_leaf 200, feature_fraction 0.8, max_bin 255,
                  max 3,000 rounds, early stopping 100 rounds on the inner fold, deterministic, 4 threads
        mlp       shared player encoder 72 -> 64 -> 64 (GELU) applied to the 10 role-slot blocks, concatenated in
                  role-slot order with the 276 team / global columns -> 256 -> 128 -> 1; AdamW lr 1e-3, batch 1,024,
                  weight decay in {1e-4, 1e-3}, max 20 epochs, patience 3 on the inner fold, CPU, torch threads 4
  F6  Selection: lowest RAW (uncalibrated) log loss on V_SELECT; a simpler candidate (logistic < lgbm < mlp) within
      0.0005 of the best is taken instead.  Then every candidate (and the baseline) gets its own PositiveSlopeSigmoid
      (fc20260915_common, copied in ev4_v_models) fitted on V_CAL with unit row weights; the frozen V is the chosen
      candidate with its calibrator.
  F7  E4 metrics on V_SELECT and V_CAL (raw and calibrated): log loss, Brier, AUC, calibration slope / intercept
      (y ~ a + b logit p) and calibration-in-the-large, overall and by t < 15, 15-25, >= 25 min; paired log-loss
      differences to the chosen V with match-cluster SEs.  Baselines: gold-difference logistic (15.14; features
      sum blue totalGold_norm - sum red totalGold_norm and its product with time_minutes) and the frozen previous
      evaluator (A_MLP_expanded) - not scored, see PREV_EVALUATOR.
  F8  Martingale checks on the 15.15 intervals (t, t + h) from ev4_02 (one per match), dV = V(t + h) - V(t) with the
      frozen (calibrated) V:
        (a) mean dV in 4 strata (all; frame update in (t, t+h] yes / no; kills in (t - 60 s, t] >= 3) with
            Bonferroni simultaneous 90 % intervals (z = 2.241), match-cluster SE; pass = interval inside +-0.002
            (+-0.005 for the death stratum);
        (b) efficiency regression of dV on X_t = the q feature set (state_value_v3.q_columns(), 995 columns): OLS on
            the identifiable principal directions (reduced to floor(G / 10) leading directions when there are fewer
            than 10 matches per direction - smoke only, flagged).  PRIMARY p-value (author pre-decision, see below):
            the wild cluster (match) bootstrap p of the studentised CR1 joint Wald statistic, Rademacher weights,
            null imposed, 999 replicates in the full run (MART_BOOT; --smoke may pass --mart-boot N), for the chosen
            V.  SECONDARY: the asymptotic CR1 joint F test (oversized at this sample size: smoke F p = 7e-17 vs
            bootstrap p = 0.715), reported for every candidate.  Also out-of-sample R^2 (basis + OLS fitted on one
            half of the matches by sha256('<match_id>:ev4_03:mart_oos'), evaluated on the other half, both
            directions).
      The same checks are also reported for the other calibrated candidates (secondary; F test and R^2 only).
  F9  Guards: apply_preset(cfg, 'v4-exact') + ExactParams.from_cfg(cfg, 'v4', require_locked=True) + record-1A G / D
      and core/presets.py sha256 (ev4_02_extract.load_params); the extract manifests must carry the same params;
      split_guard.assert_selection_patches on the two input patches BEFORE any array is read (train must be 15.14,
      select 15.15; anything else is refused - this script never takes a held-out patch); every chunk file is
      verified against its sidecar sha256; the whole run is inside grid_guard.forbid_grid().  Sample extracts
      (manifest sample = true) are refused unless --smoke; budget overrides and candidate subsets need --smoke.
  F10 Remakes: both extract manifests must carry the remake rule of ev4_common (GAME_END < 300,000 ms excluded,
      ev4_02 decision D11); after loading, no V row (15.14 / 15.15) and no martingale row may have game_end
      < 300,000 (RuntimeError otherwise; census 'remake_rows_found' = 0).
  F11 Input linkage (check_inputs, before any array is read): the 15.15 manifest's h_source.sha256 must equal the
      15.14 manifest's h_distribution.sha256 and the file <train>/h_distribution.npy must hash to it; without
      --smoke h_source.source_sample must be false (the full, non-sample 15.14 h distribution) and its
      source_n_matches must equal the 15.14 n_matches.  Both manifests' code_sha256 must equal
      ev4_02_extract.code_hashes() of the current files (extract script, ev4_common, gameplay modules, DR rules);
      a mismatch is refused unless --allow-code-drift, which is logged and written to the report ('input_checks').
  F12 Stop rule (record key 'stop_rule'): if the chosen V does not beat the gold-difference logistic baseline on
      RAW V_SELECT log loss at full size, record 1 must not be written; the report carries 'stop_rule' with
      stop_before_record1 (always false under --smoke / --pilot-matches, where it is informational).  ENFORCED:
      when stop_before_record1 is true the candidates, predictions, report_e4.json and frozen_manifest.json
      (frozen = false, V_frozen = null) are still written for diagnosis, but V_frozen/ is NOT written (a stale one
      in <out> is removed), V_NOT_FROZEN.md explains the stop, and the script exits with code 3 (EXIT_STOP_RULE).
      Later stages must call ev4_v_models.assert_v_usable(frozen_manifest) before load_v; it raises on such a run.
  F13 Pilot (--pilot-matches N, diagnostic only): train / select extracts are subset to the first N matches of each
      extract by sha256(match_id) (ties by id; the ev4_02 universe order) among the matches with V rows; the
      martingale rows are those of the kept 15.15 matches.  Grids, budgets, CV, selection, calibration and the
      stop rule are unchanged (budget overrides still need --smoke).  A pilot is marked pilot = true, its stop rule
      is informational (as under --smoke), it never writes V_frozen/ (V_NOT_FROZEN.md instead) and it refuses an
      --out that already holds a V_frozen/.

Author pre-decisions (chat, before any full-run result), outputs/reest_exact_v4_20260925/records/
stage2_predecisions_20260925T151437Z.json, sha256 b724ed7973bab4ab5481092b73b9bd890103a8d32fd92d96be0eb1759b7be101
(pinned in ev4_common; the whole record is copied into report_e4.json / frozen_manifest.json 'decisions'
-> 'predecisions_record'):
  * side marker: NOT added (SUPERSEDED by the V revision R1 above: the side marker is added).  V was symmetrised by
    swapping blue/red blocks and flipping the target for a random 50 % of rows (F3); the blue-side advantage was
    absorbed only by the V_CAL calibration intercept.  The plan's "add flipped copies" is replaced by the in-place
    50 % swap (memory).
  * martingale (b): wild cluster bootstrap p primary, CR1 F secondary (F8).
  * remakes: GAME_END < 300,000 ms excluded from V training, detection outputs, labels and all evaluations (F10).
  * technical defaults:
      martingale_h          h drawn from the empirical e90 - tau distribution of 15.14 isolated engagements
      recent_deaths_window  kills in (t - 60 s, t]
      V_selection           raw (uncalibrated) V_SELECT log loss; tie within 0.0005 -> simpler (logistic < LightGBM
                            < MLP)
      MLP                   CV with one seed; final model averages three seeds
      final_fit             final LightGBM / MLP trained on 90 % of 15.14 matches with 10 % for early stopping; no
                            full refit
      row_weights           unit weights in V training, calibration and E4 metrics
      martingale_overlap    martingale intervals come from 15.15 matches that also contribute to V_CAL (accepted)
      V_rows                one random ms per 2-minute bucket per match
      post_states           computed at the label stage after V is frozen
  * stop rule: F12.

Outputs (<out>, default outputs/reest_exact_v4_20260925/stage2/fit_v, or .../fit_v_smoke with --smoke, or
.../fit_v_pilot with --pilot-matches):
  candidates/<kind>/         bundle per candidate (bundle.json lists the sha256 of every file; model files,
                             prep.npz with the NaN fill, calibrator fitted on V_CAL; side marker)
  recalibrated/<kind>/       only when R2 triggered and the recalibration was fitted: the chosen candidate's bundle
                             with the recalibration (adopted or not)
  V_frozen/                  byte copy of the frozen variant: candidates/<chosen> (side-marker V) or
                             recalibrated/<chosen> (adopted recalibration); load with ev4_v_models.load_v; NOT written
                             when the stop rule triggers or for a pilot
  V_NOT_FROZEN.md            only when V_frozen/ is not written: why (stop rule / pilot)
  report_e4.json             CV tables, selection, E4 metrics, baselines, martingale checks, runtimes, memory
  predictions_v_15.15.parquet  match_id, t, bucket, split, y_blue_win, p_raw_<kind>, p_cal_<kind>, p_*_gold
  predictions_mart_15.15.parquet  martingale rows with V(t), V(t + h) for every calibrated candidate
  frozen_manifest.json       chosen kind, bundle sha256 of V_frozen and of every candidate, inputs, code sha256,
                             guards, decisions, grids
  run.log

Predict API for later stages:
  import ev4_v_models as VM
  vf = VM.assert_v_usable(<out>/frozen_manifest.json)      # raises VNotUsable: stop rule / pilot / smoke / not frozen
  V = VM.load_v(vf["dir"], expected_sha256=vf["bundle_sha256"])
  p_blue = V.predict(X)   # X: (n, 996) StateV3 matrix in STATE_V3_COLUMNS order (or pass columns=names);
                          # side = +1 (blue perspective) is appended; a recalibrated V uses the rows'
                          # snapshot_age_s and time_minutes

CLI
  python scripts/exact_v4/ev4_03_fit_v.py --train <extract>/15.14 --select <extract>/15.15 [--out DIR] [--threads 4]
         [--allow-code-drift] [--pilot-matches N]
  smoke: ... --smoke [--lgbm-rounds N] [--lgbm-patience N] [--mlp-epochs N] [--logistic-maxiter N] [--mart-boot N]
               [--candidates logistic,lgbm,mlp]
  exit codes: 0 done; 3 stop rule triggered (full run; nothing frozen); 1 / other: refused or crashed
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + MKL OpenMP clash on Windows/conda (repo convention)

import sys  # noqa: E402

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
for _p in (str(WT), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_common as EC  # noqa: E402
import ev4_v_models as VM  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("ev4_02_extract", HERE.parent / "ev4_02_extract.py")
EX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EX)

OUT_BASE = EX.OUT_BASE
DEFAULT_OUT = OUT_BASE / "stage2" / "fit_v"
DEFAULT_OUT_SMOKE = OUT_BASE / "stage2" / "fit_v_smoke"
DEFAULT_OUT_PILOT = OUT_BASE / "stage2" / "fit_v_pilot"
EXIT_STOP_RULE = 3       # process exit code when the F12 stop rule triggers (full run): nothing is frozen
TRAIN_PATCH, SELECT_PATCH = "15.14", "15.15"
CANDIDATES = VM.SIMPLICITY_ORDER
MART_BOOT = 999          # wild cluster bootstrap replicates for martingale (b), full run (primary p); smoke may lower
MART_BANDS = {"all": 0.002, "frame_update_yes": 0.002, "frame_update_no": 0.002, "recent_deaths_ge3": 0.005}
V_REVISION_RECORD = EX.OUT_BASE / "records" / "v_revision_prespec_20260925T233953Z.json"
V_REVISION_SHA256 = "b09390d63c3fa4d879c6b126c65be9e5b9ff45e0cb6d546980063750e4a1d79a"


def v_revision_info() -> Dict[str, Any]:
    """{path, sha256, content} of the author's V revision pre-specification; refuses a missing or edited record."""
    return EC.predecisions_info(V_REVISION_RECORD, V_REVISION_SHA256)

DECISIONS = {
    "F1_rows": "ev4_02 V rows: one uniform ms per 2-minute bucket per match; post-engagement states computed at the "
               "label stage after V is frozen",
    "F2_data": "train 15.14 V rows; 15.15 V rows -> V_SELECT / V_CAL by sha256('<mid>:ev4_03:vsplit') parity; "
               "996 StateV3 columns; NaN -> 15.14 column mean",
    "F3_symmetry": "logistic/lgbm/gold: seeded 50% rows swapped in place (default_rng(20260925)); mlp: 50% of each "
                   "minibatch; 15.15 scored natural",
    "F4_tuning": "15.14 5-fold CV by sha256('<mid>:ev4_03:cv5'); pooled OOF log loss; inner fold = "
                 "sha256('<mid>:ev4_03:inner') mod 10 == 0; final lgbm/mlp = 15.14 minus inner, stopped on inner; "
                 "final logistic = all 15.14; MLP CV seed 0, final seeds 0,1,2",
    "F5_grids": "ev4_v_models.GRIDS (fixed)",
    "F6_selection": "lowest RAW V_SELECT log loss; simpler (logistic<lgbm<mlp) within 0.0005 wins; "
                    "PositiveSlopeSigmoid per candidate on V_CAL (unit weights)",
    "F7_metrics": "logloss, Brier, AUC, cal slope/intercept, CITL; bins <15, 15-25, >=25 min; paired deltas with "
                  "match-cluster SE; gold-diff logistic baseline",
    "F8_martingale": "frozen V; (a) 4 strata Bonferroni 90% vs +-0.002 (+-0.005 deaths); (b) OLS on identifiable PCs "
                     "of q_columns; PRIMARY p = wild cluster (match) bootstrap, Rademacher, null imposed, "
                     f"{MART_BOOT} reps (full run), chosen V; SECONDARY p = CR1 joint F (all candidates); OOS R^2 by "
                     "sha256('<mid>:ev4_03:mart_oos') halves",
    "F9_guards": "load_params (preset + require_locked + record 1A); assert_selection_patches before arrays; "
                 "sidecar sha256; forbid_grid; sample extracts only with --smoke",
    "F10_remakes": f"extract manifests carry the remake rule (GAME_END < {EC.REMAKE_MAX_GAME_END_MS} ms excluded); "
                   "no V / martingale row with game_end below it (asserted after loading)",
    "F11_input_linkage": "15.15 h_source.sha256 == 15.14 h_distribution.sha256 == sha256(<train>/h_distribution.npy); "
                         "non-smoke: h source not a sample and from all 15.14 matches; extract code_sha256 == current "
                         "files unless --allow-code-drift (logged)",
    "F12_stop_rule": "chosen V must beat the gold-difference logistic on RAW V_SELECT log loss at full size, else stop "
                     "before record 1; enforced: no V_frozen/ (V_NOT_FROZEN.md), report / candidates kept for "
                     f"diagnosis, exit code {EXIT_STOP_RULE}; later stages call ev4_v_models.assert_v_usable",
    "F13_pilot": "--pilot-matches N: first N matches by sha256(match_id) of each extract (with V rows), diagnostic "
                 "only; same grids / budgets / selection / stop rule (informational); never writes V_frozen/",
    "deviation_side_marker": "REVERSED by the V revision (R1): a +-1 side column is added to the 996 StateV3 "
                             "columns (+1 original rows, -1 swapped rows; predict on StateV3 uses +1), as plan "
                             "section 4 ('keep the blue/red marker'); the stage-2 pre-decision 'side marker not added' "
                             "is superseded",
    "R1_side_marker": "V input = StateV3 996 columns + 'side' (+1 original / -1 swapped; logistic/lgbm/gold: in-place "
                      "50% swap flips it; mlp: per minibatch); side standardised with swap-pooled moments (0, 1); "
                      "15.15 and predict() use side = +1; same grids / budgets / folds / seeds / selection rule",
    "R2_conditional_recalibration": "only if the side-marker V fails martingale (a) in any stratum or (b) primary p <= "
                                    "0.05: logistic on logit V + frame-age bins (0-15/15-30/30-45/45-60 s) intercepts "
                                    "and x logit V + natural spline of game minute (4 knots, V_CAL 5/35/65/95 % "
                                    "quantiles), fitted on V_CAL; (a)/(b) re-run; adopted if V_SELECT log loss not "
                                    "worse than the side-marker V by > 0.0005; both reported",
    "deviation_flip_copies": "plan 'add flipped copies' replaced by the in-place 50% swap (memory)",
}

PREV_EVALUATOR = {
    "name": "A_MLP_expanded (frozen previous V, StateV2 features)",
    "scored": False,
    "reason": ("Not scored inside fit-V. It is scored separately by scripts/exact_v4/ev4_03c_prev_v_compare.py on the "
               "same 15.15 V_SELECT rows (author permission to read only C:/Users/todtj/PycharmProjects/LOL_teamfight/"
               "outputs/v_redesign_wave4_corrected_20260919/evaluators/; record v_revision_prespec_20260925T233953Z); "
               "see stage2/prev_v_compare/compare_vs_fit_v2.json."),
}


# ============================================================================ logging / memory
class Log:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[ev4_03 {time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def memory_info() -> Dict[str, Optional[float]]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)
        return {"rss_gb": mi.rss / 1e9, "peak_gb": (peak / 1e9) if peak is not None else None}
    except Exception:  # noqa: BLE001
        return {"rss_gb": None, "peak_gb": None}


# ============================================================================ inputs
def read_manifest(d: Path) -> Dict[str, Any]:
    p = Path(d) / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"extract manifest missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def check_h_source(train_dir: Path, mt: Mapping[str, Any], ms: Mapping[str, Any], smoke: bool) -> Dict[str, Any]:
    """F11: the 15.15 martingale h came from THIS 15.14 extract's h_distribution.npy (sha256 linkage); without
    --smoke that distribution must be the full, non-sample one.  Hashes the .npy file bytes only."""
    hd = mt.get("h_distribution") or {}
    hs = ms.get("h_source") or {}
    want = hd.get("sha256")
    if not want:
        raise RuntimeError("15.14 extract manifest has no h_distribution.sha256")
    if hs.get("sha256") != want:
        raise RuntimeError(f"15.15 extract used h source {hs.get('sha256')} ({hs.get('path')}), but the 15.14 extract's "
                           f"h_distribution.sha256 is {want}")
    hp = Path(train_dir) / str(hd.get("file") or "h_distribution.npy")
    if not hp.is_file() or VM.sha256_file(hp) != want:
        raise RuntimeError(f"{hp} is missing or does not hash to the 15.14 manifest's h_distribution.sha256")
    if Path(str(hs.get("path", ""))).name != hp.name:
        raise RuntimeError(f"15.15 h source file {hs.get('path')} is not an {hp.name}")
    if not smoke:
        if hs.get("source_sample") is not False:
            raise RuntimeError(f"15.15 extract used a sample h distribution (source_sample={hs.get('source_sample')}); "
                               "the full run needs the full 15.14 h_distribution.npy")
        if mt.get("sample") is not False:
            raise RuntimeError("15.14 extract is a sample run")
        if hs.get("source_n_matches") != mt.get("n_matches"):
            raise RuntimeError(f"h source was built from {hs.get('source_n_matches')} matches, the 15.14 extract has "
                               f"{mt.get('n_matches')}")
    return {"h_sha256": want, "h_file": str(hp), "h_source_path": hs.get("path"),
            "h_source_sample": hs.get("source_sample"), "h_source_n_matches": hs.get("source_n_matches"),
            "train_n_matches": mt.get("n_matches"), "ok": True}


def check_code_drift(manifests: Mapping[str, Mapping[str, Any]], allow: bool) -> Dict[str, Any]:
    """F11: extract manifests' code_sha256 vs ev4_02_extract.code_hashes() of the current files."""
    cur = EX.code_hashes()
    drift: Dict[str, Any] = {}
    for name, m in manifests.items():
        got = m.get("code_sha256") or {}
        diff = {f: {"extract": got.get(f), "current": cur.get(f)} for f in sorted(set(got) | set(cur))
                if got.get(f) != cur.get(f)}
        if diff:
            drift[name] = diff
    if drift and not allow:
        raise RuntimeError(f"extract code differs from the current code: {json.dumps(drift)[:800]}; re-extract, or pass "
                           "--allow-code-drift only after checking that the change cannot affect the extracted rows")
    return {"current": cur, "drift": drift, "allow_code_drift": bool(allow), "drift_found": bool(drift)}


def check_remake_rule(manifests: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """F10: both extracts were built with the ev4_common remake rule."""
    out = {}
    for name, m in manifests.items():
        r = m.get("remakes") or {}
        if r.get("threshold_ms") != EC.REMAKE_MAX_GAME_END_MS:
            raise RuntimeError(f"{name} extract was not built with the remake rule (GAME_END < "
                               f"{EC.REMAKE_MAX_GAME_END_MS} ms excluded): manifest remakes = {r or None}")
        out[name] = {"threshold_ms": r["threshold_ms"], "n_excluded": r.get("n_excluded"),
                     "match_ids": r.get("match_ids")}
    return out


def assert_no_remakes(tables: Mapping[str, Any]) -> Dict[str, int]:
    """F10: no row with game_end < REMAKE_MAX_GAME_END_MS in the loaded V / martingale index tables."""
    out = {}
    for name, df in tables.items():
        if df is None or not len(df):
            out[name] = 0
            continue
        if "game_end" not in df.columns:
            raise RuntimeError(f"{name} rows carry no game_end; cannot check the remake rule")
        ge = df["game_end"].to_numpy()
        bad = ge < EC.REMAKE_MAX_GAME_END_MS
        if bad.any():
            ids = sorted(set(df.loc[bad, "match_id"].astype(str)))
            raise RuntimeError(f"{name}: {int(bad.sum())} rows of {len(ids)} remake matches (GAME_END < "
                               f"{EC.REMAKE_MAX_GAME_END_MS} ms) present: {ids[:10]}")
        out[name] = 0
    return out


def check_inputs(train_dir: Path, select_dir: Path, smoke: bool,
                 allow_code_drift: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Guards on the manifests only (no array is read here): split (train 15.14, select 15.15), StateV3 hash,
    sample runs, remake rule, h-source linkage and code drift.  Returns (train manifest, select manifest, checks)."""
    from gameplay.split_guard import SplitViolation, assert_selection_patches, normalize_patch
    from gameplay.state_value_v3 import STATE_V3_NAME_HASH, STATE_VERSION
    mt, ms = read_manifest(train_dir), read_manifest(select_dir)
    pt, ps = normalize_patch(str(mt.get("patch"))), normalize_patch(str(ms.get("patch")))
    assert_selection_patches([pt, ps])
    if pt != TRAIN_PATCH or ps != SELECT_PATCH:
        raise SplitViolation(f"V is trained on {TRAIN_PATCH} and selected on {SELECT_PATCH}; got train {pt}, select {ps}")
    for name, m in (("train", mt), ("select", ms)):
        if m.get("STATE_V3_NAME_HASH") != STATE_V3_NAME_HASH or m.get("state_version") != STATE_VERSION:
            raise RuntimeError(f"{name} extract has StateV3 {m.get('STATE_V3_NAME_HASH')} / {m.get('state_version')}, "
                               f"code has {STATE_V3_NAME_HASH} / {STATE_VERSION}")
        if m.get("sample") and not smoke:
            raise SystemExit(f"{name} extract is a sample run (manifest sample=true); only --smoke may use it")
    checks = {"remake_rule": check_remake_rule({"train": mt, "select": ms}),
              "h_source": check_h_source(train_dir, mt, ms, smoke),
              "code": check_code_drift({"train": mt, "select": ms}, allow_code_drift)}
    return mt, ms, checks


def check_params(manifests: Sequence[Mapping[str, Any]], guard: Mapping[str, Any]) -> None:
    want = json.loads(json.dumps(guard["params"], default=EX.json_default))
    for m in manifests:
        got = (m.get("guards") or {}).get("params")
        if got != want:
            raise RuntimeError(f"extract {m.get('patch')} was built with params {got}, preset gives {want}")
        if (m.get("guards") or {}).get("record1a_sha256") != guard["record1a_sha256"]:
            raise RuntimeError(f"extract {m.get('patch')} used another record 1A")


def chunk_ids(man: Mapping[str, Any]) -> List[str]:
    return sorted((man.get("chunks") or {}).keys())


def verify_files(d: Path, man: Mapping[str, Any], log: Log) -> Dict[str, Any]:
    """Every chunk file listed in the manifest matches its sha256 (and the sidecar agrees with the manifest)."""
    n = 0
    for cid in chunk_ids(man):
        ent = man["chunks"][cid]
        side = json.loads((Path(d) / f"chunk_{int(cid):05d}.json").read_text(encoding="utf-8"))
        if side.get("files") != ent.get("files"):
            raise RuntimeError(f"chunk {cid} sidecar and manifest disagree")
        for fn, f in ent["files"].items():
            if VM.sha256_file(Path(d) / fn) != f["sha256"]:
                raise RuntimeError(f"{d}/{fn} does not match its sha256")
            n += 1
    log(f"verified {n} files in {d}")
    return {"n_files": n, "manifest_sha256": VM.sha256_file(Path(d) / "manifest.json")}


def _kept_rows(d: Path, c: str, suffix: str, keep: Optional[set]) -> Optional[np.ndarray]:
    """Pilot filter: boolean mask over a chunk's index rows whose match_id is in `keep` (None = keep everything)."""
    if keep is None:
        return None
    import pandas as pd
    mids = pd.read_parquet(Path(d) / f"chunk_{int(c):05d}_{suffix}.parquet", columns=["match_id"])["match_id"]
    return mids.astype(str).isin(keep).to_numpy()


def load_v_rows(d: Path, man: Mapping[str, Any], keep: Optional[set] = None,
                side: bool = False) -> Tuple[np.ndarray, "Any"]:
    """All V rows of an extract (keep = None), or only the rows of the matches in `keep` (pilot, F13).  side=True
    appends the side-marker column (V revision R1) filled with +1 (natural, blue-perspective orientation)."""
    import pandas as pd
    from gameplay.state_value_v3 import STATE_V3_COLUMNS
    ids = chunk_ids(man)
    masks = {c: _kept_rows(d, c, "v", keep) for c in ids}
    if keep is None:
        n = sum(int(man["chunks"][c]["rows"]["v"]) for c in ids)
    else:
        n = sum(int(masks[c].sum()) for c in ids)
    ns = len(STATE_V3_COLUMNS)
    X = np.empty((n, ns + (1 if side else 0)), dtype=np.float32)
    if side:
        X[:, ns] = 1.0
    metas, a = [], 0
    for c in ids:
        sel = masks[c]
        if sel is not None and not sel.any():
            continue
        z = np.load(Path(d) / f"chunk_{int(c):05d}.npz")
        if tuple(z["state_columns"].tolist()) != STATE_V3_COLUMNS:
            raise RuntimeError(f"chunk {c}: state columns differ from STATE_V3_COLUMNS")
        xv = z["v_X"]
        mv = pd.read_parquet(Path(d) / f"chunk_{int(c):05d}_v.parquet")
        if len(mv) != len(xv) or not np.array_equal(mv["row"].to_numpy(), np.arange(len(xv))):
            raise RuntimeError(f"chunk {c}: V index and v_X disagree")
        if sel is not None:
            xv = xv[sel]
            mv = mv.loc[sel].reset_index(drop=True)
        X[a:a + len(xv), :ns] = xv
        mv["chunk"] = int(c)
        metas.append(mv)
        a += len(xv)
    if a != n:
        raise RuntimeError("V row count differs from the manifest")
    meta = pd.concat(metas, ignore_index=True)
    return X, meta


def load_mart_rows(d: Path, man: Mapping[str, Any], keep: Optional[set] = None,
                   side: bool = False) -> Tuple[np.ndarray, np.ndarray, "Any"]:
    """All martingale rows of an extract (keep = None), or only those of the matches in `keep` (pilot, F13).
    side=True appends the side-marker column (+1) to both state matrices."""
    import pandas as pd
    from gameplay.state_value_v3 import STATE_V3_COLUMNS
    ids = chunk_ids(man)
    masks = {c: _kept_rows(d, c, "mart", keep) for c in ids}
    if keep is None:
        n = sum(int(man["chunks"][c]["rows"]["mart"]) for c in ids)
    else:
        n = sum(int(masks[c].sum()) for c in ids)
    ns = len(STATE_V3_COLUMNS)
    X0 = np.empty((n, ns + (1 if side else 0)), dtype=np.float32)
    X1 = np.empty_like(X0)
    if side:
        X0[:, ns] = 1.0
        X1[:, ns] = 1.0
    metas, a = [], 0
    for c in ids:
        sel = masks[c]
        if sel is not None and not sel.any():
            continue
        z = np.load(Path(d) / f"chunk_{int(c):05d}.npz")
        x0, x1 = z["m_X0"], z["m_X1"]
        mm = pd.read_parquet(Path(d) / f"chunk_{int(c):05d}_mart.parquet")
        if len(mm) != len(x0) or len(x1) != len(x0):
            raise RuntimeError(f"chunk {c}: martingale index and arrays disagree")
        if sel is not None:
            x0, x1 = x0[sel], x1[sel]
            mm = mm.loc[sel].reset_index(drop=True)
        X0[a:a + len(x0), :ns] = x0
        X1[a:a + len(x1), :ns] = x1
        mm["chunk"] = int(c)
        metas.append(mm)
        a += len(x0)
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    return X0, X1, meta


def pilot_match_ids(d: Path, man: Mapping[str, Any], n: int) -> Tuple[List[str], int]:
    """F13: the first n matches of an extract by sha256(match_id) (ties by id; the ev4_02 universe order), among the
    matches with V rows.  Reads only the match_id column of the chunk V index tables.  Returns (ids, n_total)."""
    import pandas as pd
    mids: set = set()
    for c in chunk_ids(man):
        mids.update(pd.read_parquet(Path(d) / f"chunk_{int(c):05d}_v.parquet",
                                    columns=["match_id"])["match_id"].astype(str).tolist())
    order = sorted(mids, key=lambda m: (EX.sha256_text(m), m))
    return order[:int(n)], len(order)


# ============================================================================ CV helpers
def pick(cv: Mapping[Any, float]) -> Any:
    """Grid value with the lowest pooled OOF log loss (first in grid order on ties)."""
    keys = list(cv)
    return min(keys, key=lambda k: (cv[k], keys.index(k)))


def cv_logistic(X, y, folds, grid, perm, log: Log, maxiter: int, negate=None) -> Dict[str, Any]:
    groups = VM.swap_pairs(perm)
    oof = {C: np.full(len(y), np.nan) for C in grid}
    fits: List[Dict[str, Any]] = []
    for k in range(5):
        tr = np.flatnonzero(folds != k)
        te = np.flatnonzero(folds == k)
        mu, sd = VM.pooled_standardizer(X, tr, groups, negate=negate)
        for C in grid:
            m = VM.LogisticModel(C).fit(X, y, tr, mu, sd, maxiter=maxiter)
            oof[C][te] = m.predict_proba(X[te])
            fits.append({"fold": k, "C": C, **m.fit_info, "fold_logloss": VM.log_loss(y[te], oof[C][te])})
            log(f"  logistic fold {k} C={C:g}: {fits[-1]['fold_logloss']:.5f} (nit {m.fit_info['nit']}, "
                f"{m.fit_info['seconds']} s, {m.fit_info['message'][:40]})")
    table = {C: VM.log_loss(y, oof[C]) for C in grid}
    return {"grid": {"C": list(grid)}, "cv_logloss": {str(C): v for C, v in table.items()}, "chosen": pick(table),
            "fits": fits}


def cv_lgbm(ds, X, y, folds, inner, grid, rounds, patience, threads, log: Log) -> Dict[str, Any]:
    oof = {nl: np.full(len(y), np.nan) for nl in grid}
    fits: List[Dict[str, Any]] = []
    for k in range(5):
        tr = np.flatnonzero((folds != k) & ~inner)
        st = np.flatnonzero((folds != k) & inner)
        te = np.flatnonzero(folds == k)
        for nl in grid:
            m = VM.LGBMModel(nl, rounds, patience, threads).fit(ds, tr, st)
            oof[nl][te] = m.predict_proba(X[te])
            fits.append({"fold": k, **m.fit_info, "fold_logloss": VM.log_loss(y[te], oof[nl][te])})
            log(f"  lgbm fold {k} leaves={nl}: {fits[-1]['fold_logloss']:.5f} (best it {m.best_iteration}, "
                f"{m.fit_info['seconds']} s)")
    table = {nl: VM.log_loss(y, oof[nl]) for nl in grid}
    return {"grid": {"num_leaves": list(grid)}, "cv_logloss": {str(k): v for k, v in table.items()},
            "chosen": pick(table), "fits": fits}


def cv_mlp(X, y, folds, inner, grid, names, perm, epochs, patience, threads, log: Log) -> Dict[str, Any]:
    groups = VM.mlp_groups(names, perm)
    negate = VM.swap_negate_idx(names)
    oof = {wd: np.full(len(y), np.nan) for wd in grid}
    fits: List[Dict[str, Any]] = []
    for k in range(5):
        tr = np.flatnonzero((folds != k) & ~inner)
        st = np.flatnonzero((folds != k) & inner)
        te = np.flatnonzero(folds == k)
        mu, sd = VM.pooled_standardizer(X, tr, groups, negate=negate)
        for wd in grid:
            m = VM.MLPModel(wd, names, seeds=VM.MLP_SEEDS_CV, max_epochs=epochs, patience=patience, threads=threads)
            m.fit(X, y, tr, st, mu, sd)
            oof[wd][te] = m.predict_proba(X[te])
            fi = {k2: v for k2, v in m.fit_info.items() if k2 != "runs"}
            fi["runs"] = [{k3: v3 for k3, v3 in r.items() if k3 != "history"} for r in m.fit_info["runs"]]
            fits.append({"fold": k, **fi, "fold_logloss": VM.log_loss(y[te], oof[wd][te])})
            log(f"  mlp fold {k} wd={wd:g}: {fits[-1]['fold_logloss']:.5f} ({m.fit_info['seconds']} s, "
                f"epochs {[r['epochs_run'] for r in m.fit_info['runs']]})")
    table = {wd: VM.log_loss(y, oof[wd]) for wd in grid}
    return {"grid": {"weight_decay": list(grid)}, "cv_logloss": {str(k): v for k, v in table.items()},
            "chosen": pick(table), "fits": fits}


# ============================================================================ code provenance
def code_hashes() -> Dict[str, str]:
    out = {"scripts/exact_v4/ev4_03_fit_v.py": VM.sha256_file(HERE),
           "scripts/exact_v4/ev4_v_models.py": VM.sha256_file(HERE.parent / "ev4_v_models.py"),
           "scripts/exact_v4/ev4_common.py": VM.sha256_file(HERE.parent / "ev4_common.py"),
           "scripts/exact_v4/ev4_02_extract.py": VM.sha256_file(HERE.parent / "ev4_02_extract.py")}
    for m in ("gameplay/state_value_v3.py", "gameplay/split_guard.py", "gameplay/grid_guard.py", "core/presets.py",
              "gameplay/exact_population.py"):
        p = WT / m
        if p.exists():
            out[m] = VM.sha256_file(p)
    return out


def git_state() -> Dict[str, Any]:
    import subprocess
    head = EX.git_head()
    try:
        st = subprocess.run(["git", "-C", str(WT), "status", "--porcelain", "--", "scripts/exact_v4", "gameplay",
                             "core"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        st = ""
    return {"head": head, "dirty_paths": [ln[3:] for ln in st.splitlines() if ln.strip()]}


# ============================================================================ main
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=Path, required=True, help="ev4_02 extract directory of 15.14")
    ap.add_argument("--select", type=Path, required=True, help="ev4_02 extract directory of 15.15")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=VM.MAX_THREADS)
    ap.add_argument("--smoke", action="store_true", help="allow sample extracts and reduced budgets")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--lgbm-rounds", type=int, default=None)
    ap.add_argument("--lgbm-patience", type=int, default=None)
    ap.add_argument("--mlp-epochs", type=int, default=None)
    ap.add_argument("--logistic-maxiter", type=int, default=None)
    ap.add_argument("--mart-boot", type=int, default=None,
                    help=f"wild bootstrap replicates for (b), the primary p (default {MART_BOOT}; smoke only)")
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="accept extracts whose code_sha256 differs from the current code (logged in the report)")
    ap.add_argument("--pilot-matches", type=int, default=None, metavar="N",
                    help="diagnostic pilot: first N matches of each extract by sha256(match_id); same grids and "
                         "selection rule; marked pilot=true; never writes V_frozen/")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    t_start = time.time()
    predec = EC.predecisions_info()           # refuse a missing or edited author pre-decision record first
    vrev = v_revision_info()                  # and the V revision pre-specification (side marker, recalibration)
    pilot = args.pilot_matches is not None
    if pilot and args.pilot_matches < 1:
        raise SystemExit("--pilot-matches must be >= 1")
    threads = int(args.threads)
    if not 1 <= threads <= VM.MAX_THREADS:
        raise SystemExit(f"--threads must be in [1, {VM.MAX_THREADS}]")
    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    if not cands or any(c not in CANDIDATES for c in cands):
        raise SystemExit(f"--candidates must be a subset of {CANDIDATES}")
    overrides = {k: getattr(args, k) for k in ("lgbm_rounds", "lgbm_patience", "mlp_epochs", "logistic_maxiter",
                                               "mart_boot")
                 if getattr(args, k) is not None}
    if not args.smoke and (overrides or cands != list(CANDIDATES)):
        raise SystemExit("budget overrides and candidate subsets are allowed only with --smoke (the grid is fixed)")
    budget = {"lgbm_rounds": overrides.get("lgbm_rounds", VM.LGBM_MAX_ROUNDS),
              "lgbm_patience": overrides.get("lgbm_patience", VM.LGBM_PATIENCE),
              "mlp_epochs": overrides.get("mlp_epochs", VM.MLP_MAX_EPOCHS),
              "mlp_patience": VM.MLP_PATIENCE,
              "logistic_maxiter": overrides.get("logistic_maxiter", VM.GRIDS["logistic"]["fixed"]["maxiter"]),
              "mart_boot": overrides.get("mart_boot", MART_BOOT)}

    from gameplay.grid_guard import forbid_grid
    with forbid_grid():
        # ---- guards before any array is read
        mt, ms, checks = check_inputs(args.train, args.select, args.smoke, args.allow_code_drift)
        _, params, guard = EX.load_params()
        check_params([mt, ms], guard)
        if args.out is not None:
            out = Path(args.out)
        else:
            out = DEFAULT_OUT_SMOKE if args.smoke else (DEFAULT_OUT_PILOT if pilot else DEFAULT_OUT)
        if pilot and (out / "V_frozen").exists():
            raise SystemExit(f"--pilot-matches: {out} already holds a V_frozen/ (a frozen run); use another --out")
        out.mkdir(parents=True, exist_ok=True)
        log = Log(out / "run.log")
        log(f"out {out}; smoke={args.smoke}; pilot={args.pilot_matches if pilot else False}; candidates {cands}; "
            f"budget {budget}; threads {threads}")
        log(f"input checks: remakes {checks['remake_rule']}; h source {checks['h_source']}; "
            f"code drift {checks['code']['drift_found']} (allowed {checks['code']['allow_code_drift']})")
        if checks["code"]["drift_found"]:
            log(f"WARNING --allow-code-drift: extract code differs from current code: {checks['code']['drift']}")
        log(f"pre-decision record {predec['path']} sha256 {predec['sha256']}")
        log(f"V revision record {vrev['path']} sha256 {vrev['sha256']}")
        report = _run(args, out, log, mt, ms, guard, cands, budget, threads, t_start, checks, predec, vrev)
    if report["stop_rule"]["stop_before_record1"]:
        print(f"[ev4_03] STOP RULE: the chosen V does not beat the gold-difference baseline on raw V_SELECT log loss; "
              f"V_frozen/ was not written (see {out / 'V_NOT_FROZEN.md'}); do NOT write record 1. "
              f"Exit code {EXIT_STOP_RULE}.", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_STOP_RULE)
    return report


def decisions_block(predec: Mapping[str, Any], vrev: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The report / manifest 'decisions' block: this script's decisions plus the whole author records."""
    out = {**DECISIONS, "predecisions_record": {"path": predec["path"], "sha256": predec["sha256"],
                                                **predec["content"]}}
    if vrev is not None:
        out["v_revision_record"] = {"path": vrev["path"], "sha256": vrev["sha256"], **vrev["content"]}
    return out


def checks_for_report(checks: Mapping[str, Any]) -> Dict[str, Any]:
    """check_inputs results for the report (the current code hashes are already in frozen_manifest code_sha256)."""
    out = dict(checks)
    if "code" in out:
        out["code"] = {k: v for k, v in out["code"].items() if k != "current"}
    return out


def stop_rule(raw_select: Mapping[str, float], gold_raw_select: float, chosen: str, smoke: bool,
              pilot: bool = False) -> Dict[str, Any]:
    """F12: the chosen V must beat the gold-difference logistic on RAW V_SELECT log loss (full size).  Under --smoke
    or --pilot-matches the rule is informational (stop_before_record1 false; would_stop_at_full_size carries it)."""
    beats = bool(raw_select[chosen] < gold_raw_select)
    return {"chosen": chosen, "chosen_raw_select_logloss": float(raw_select[chosen]),
            "gold_raw_select_logloss": float(gold_raw_select), "beats_gold": beats,
            "would_stop_at_full_size": not beats,
            "stop_before_record1": bool((not beats) and not (smoke or pilot)), "smoke": bool(smoke),
            "pilot": bool(pilot),
            "rule": "if the selected V does not beat the gold-difference logistic baseline on V_SELECT log loss at "
                    "full size, stop before record 1 and report",
            "enforcement": f"stop_before_record1 true -> no V_frozen/, V_NOT_FROZEN.md, exit code {EXIT_STOP_RULE}; "
                           "ev4_v_models.assert_v_usable refuses the run"}


def write_not_frozen(out: Path, reason: str, stop: Mapping[str, Any], pilot_info: Optional[Mapping[str, Any]]) -> Path:
    """V_NOT_FROZEN.md: why this run wrote no V_frozen/ (stop rule or pilot)."""
    lines = ["# V not frozen", "",
             f"Run: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, `scripts/exact_v4/ev4_03_fit_v.py`, out `{out}`.",
             ""]
    if reason == "stop_rule":
        lines += ["**Reason: stop rule (F12).** The chosen V does not beat the gold-difference logistic baseline on "
                  "raw V_SELECT log loss at full size.", "",
                  f"- chosen: `{stop['chosen']}`, raw V_SELECT log loss {stop['chosen_raw_select_logloss']:.6f}",
                  f"- gold-difference logistic: {stop['gold_raw_select_logloss']:.6f}",
                  "", "Do NOT write record 1. Report and investigate (feature scaling, swap mapping, a leak or a "
                  "mismatch in the 15.15 rows). `candidates/`, the predictions and `report_e4.json` are kept for "
                  "diagnosis only.", f"The script exited with code {EXIT_STOP_RULE}."]
    else:
        lines += [f"**Reason: pilot run** (`--pilot-matches {(pilot_info or {}).get('n_matches_requested')}`). A pilot "
                  "is diagnostic only and never writes V_frozen/.", "",
                  f"- chosen: `{stop['chosen']}` (raw V_SELECT log loss {stop['chosen_raw_select_logloss']:.6f}; "
                  f"gold-difference logistic {stop['gold_raw_select_logloss']:.6f}; would stop at full size: "
                  f"{stop['would_stop_at_full_size']})"]
    lines += ["", "Later stages must call `ev4_v_models.assert_v_usable(frozen_manifest.json)`, which refuses this run.",
              ""]
    p = out / "V_NOT_FROZEN.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _run(args, out: Path, log: Log, mt, ms, guard, cands, budget, threads, t_start, checks=None,
         predec=None, vrev=None) -> Dict[str, Any]:
    checks = checks or {}
    predec = predec or EC.predecisions_info()
    vrev = vrev or v_revision_info()
    import pandas as pd
    import torch
    torch.set_num_threads(threads)
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH, q_columns

    state_names = list(STATE_V3_COLUMNS)
    ns = len(state_names)
    names = VM.with_side(state_names)          # R1: V input = StateV3 + side marker (last column)
    perm = VM.swap_permutation(names)
    neg = VM.swap_negate_idx(names)
    timings: Dict[str, float] = {}
    mem: Dict[str, Any] = {}

    def lap(name: str, t0: float) -> None:
        timings[name] = round(time.time() - t0, 2)
        mem[name] = memory_info()

    t0 = time.time()
    inputs = {"train": {"dir": str(args.train), **verify_files(args.train, mt, log), "n_matches": mt.get("n_matches"),
                        "sample": mt.get("sample")},
              "select": {"dir": str(args.select), **verify_files(args.select, ms, log),
                         "n_matches": ms.get("n_matches"), "sample": ms.get("sample")}}
    pilot_n = getattr(args, "pilot_matches", None)
    pilot = pilot_n is not None
    keep_tr = keep_se = None
    pilot_info = None
    if pilot:                                              # F13: diagnostic subset, same rules otherwise
        ids_tr, tot_tr = pilot_match_ids(args.train, mt, pilot_n)
        ids_se, tot_se = pilot_match_ids(args.select, ms, pilot_n)
        keep_tr, keep_se = set(ids_tr), set(ids_se)
        pilot_info = {"n_matches_requested": int(pilot_n),
                      "rule": "first N matches of each extract by sha256(match_id) (ties by id) among matches with "
                              "V rows; martingale rows of the kept 15.15 matches",
                      "train": {"n_matches": len(ids_tr), "n_matches_with_v_rows": tot_tr,
                                "match_ids_sha256": EX.sha256_text("\n".join(ids_tr))},
                      "select": {"n_matches": len(ids_se), "n_matches_with_v_rows": tot_se,
                                 "match_ids_sha256": EX.sha256_text("\n".join(ids_se))}}
        log(f"PILOT (diagnostic only, never frozen): {pilot_info}")
    Xtr, mtr = load_v_rows(args.train, mt, keep_tr, side=True)
    Xse, mse = load_v_rows(args.select, ms, keep_se, side=True)
    M0, M1, mmart = load_mart_rows(args.select, ms, keep_se, side=True)
    remake_rows = assert_no_remakes({"train_v": mtr, "select_v": mse, "martingale": mmart})
    lap("load", t0)
    log(f"loaded 15.14 V {Xtr.shape}, 15.15 V {Xse.shape}, 15.15 martingale {M0.shape}; {memory_info()}")

    # ---- NaN fill (15.14 means of the StateV3 columns; the side column is never NaN)
    fill, n_nan_tr = VM.nan_fill_values(Xtr[:, :ns])
    nan_counts = {"train": n_nan_tr, "select": VM.fill_nan_inplace(Xse[:, :ns], fill),
                  "mart0": VM.fill_nan_inplace(M0[:, :ns], fill), "mart1": VM.fill_nan_inplace(M1[:, :ns], fill)}
    VM.fill_nan_inplace(Xtr[:, :ns], fill)

    ytr = mtr["y_blue_win"].to_numpy().astype(np.float64)
    yse = mse["y_blue_win"].to_numpy().astype(np.float64)
    mids_tr = mtr["match_id"].astype(str).to_numpy()
    mids_se = mse["match_id"].astype(str).to_numpy()
    folds = VM.per_match(mids_tr, VM.cv_fold).astype(int)
    inner = VM.per_match(mids_tr, VM.inner_holdout).astype(bool)
    split = VM.per_match(mids_se, VM.v_split)
    sel_m, cal_m = split == "select", split == "cal"
    census = {"train_rows": int(len(ytr)), "train_matches": int(len(set(mids_tr))),
              "train_rows_inner": int(inner.sum()), "train_fold_rows": np.bincount(folds, minlength=5).tolist(),
              "select_rows": int(sel_m.sum()), "cal_rows": int(cal_m.sum()),
              "select_matches": int(len(set(mids_se[sel_m]))), "cal_matches": int(len(set(mids_se[cal_m]))),
              "mart_rows": int(len(M0)), "y_mean_train": float(ytr.mean()), "y_mean_select": float(yse[sel_m].mean()),
              "y_mean_cal": float(yse[cal_m].mean()), "nan_filled": nan_counts, "remake_rows_found": remake_rows}
    log(f"census {census}")

    # ---- team swap (in place) for logistic / lgbm / gold
    smask = VM.swap_mask(len(ytr))
    VM.apply_swap_inplace(Xtr, ytr, smask, perm, negate=neg)      # R1: swapped rows get side = -1
    census["swapped_rows"] = int(smask.sum())
    census["side_minus_rows"] = int((Xtr[:, ns] == -1).sum())
    models: Dict[str, Any] = {}
    cv: Dict[str, Any] = {}

    if "logistic" in cands:
        t0 = time.time()
        log("logistic CV")
        cv["logistic"] = cv_logistic(Xtr, ytr, folds, VM.GRIDS["logistic"]["C"], perm, log, budget["logistic_maxiter"],
                                     negate=neg)
        C = cv["logistic"]["chosen"]
        mu, sd = VM.pooled_standardizer(Xtr, None, VM.swap_pairs(perm), negate=neg)
        models["logistic"] = VM.LogisticModel(C).fit(Xtr, ytr, np.arange(len(ytr)), mu, sd,
                                                     maxiter=budget["logistic_maxiter"])
        log(f"logistic final C={C:g}: {models['logistic'].fit_info}")
        lap("logistic", t0)

    t0 = time.time()
    gold = VM.GoldDiffLogistic(names).fit(Xtr, ytr)
    lap("gold_baseline", t0)

    if "lgbm" in cands:
        t0 = time.time()
        log("lgbm dataset + CV")
        ds = VM.LGBMModel.dataset(Xtr, ytr)
        cv["lgbm"] = cv_lgbm(ds, Xtr, ytr, folds, inner, VM.GRIDS["lgbm"]["num_leaves"], budget["lgbm_rounds"],
                             budget["lgbm_patience"], threads, log)
        nl = cv["lgbm"]["chosen"]
        models["lgbm"] = VM.LGBMModel(nl, budget["lgbm_rounds"], budget["lgbm_patience"], threads).fit(
            ds, np.flatnonzero(~inner), np.flatnonzero(inner))
        log(f"lgbm final leaves={nl}: {models['lgbm'].fit_info}")
        del ds
        lap("lgbm", t0)

    VM.apply_swap_inplace(Xtr, ytr, smask, perm, negate=neg)   # undo: natural orientation (side +1) for the MLP
    if "mlp" in cands:
        t0 = time.time()
        log("mlp CV")
        cv["mlp"] = cv_mlp(Xtr, ytr, folds, inner, VM.GRIDS["mlp"]["weight_decay"], names, perm,
                           budget["mlp_epochs"], budget["mlp_patience"], threads, log)
        wd = cv["mlp"]["chosen"]
        tr = np.flatnonzero(~inner)
        mu, sd = VM.pooled_standardizer(Xtr, tr, VM.mlp_groups(names, perm), negate=neg)
        models["mlp"] = VM.MLPModel(wd, names, seeds=VM.MLP_SEEDS_FINAL, max_epochs=budget["mlp_epochs"],
                                    patience=budget["mlp_patience"], threads=threads)
        models["mlp"].fit(Xtr, ytr, tr, np.flatnonzero(inner), mu, sd, log=log)
        log(f"mlp final wd={wd:g}: {[(r['seed'], r['best_inner_logloss'], r['epochs_run']) for r in models['mlp'].fit_info['runs']]}")
        lap("mlp", t0)

    # ---- 15.15: raw predictions, selection, calibration
    t0 = time.time()
    p_raw = {k: m.predict_proba(Xse) for k, m in models.items()}
    p_raw["gold"] = gold.predict_proba(Xse)
    raw_select = {k: VM.log_loss(yse[sel_m], p_raw[k][sel_m]) for k in models}
    selection = VM.select_candidate(raw_select)
    chosen = selection["chosen"]
    log(f"selection {selection}")
    stop = stop_rule(raw_select, VM.log_loss(yse[sel_m], p_raw["gold"][sel_m]), chosen, bool(args.smoke), pilot)
    log(f"stop rule: {stop}")
    if stop["stop_before_record1"]:
        log("STOP RULE: the chosen V does not beat the gold-difference baseline on raw V_SELECT log loss; "
            "do NOT write record 1 - report and investigate (V_frozen/ will not be written)")
    not_frozen_reason = "stop_rule" if stop["stop_before_record1"] else ("pilot" if pilot else None)
    cals = {k: VM.PositiveSlopeSigmoid().fit(p_raw[k][cal_m], yse[cal_m], np.ones(int(cal_m.sum()))) for k in p_raw}
    p_cal = {k: cals[k].predict(p_raw[k]) for k in p_raw}
    t_se = mse["t"].to_numpy()
    e4: Dict[str, Any] = {}
    for k in p_raw:
        e4[k] = {}
        for which, pp in (("raw", p_raw[k]), ("cal", p_cal[k])):
            e4[k][which] = {sp: VM.metrics_by_bins(yse[m], pp[m], t_se[m]) for sp, m in (("select", sel_m), ("cal", cal_m))}
        e4[k]["calibrator"] = cals[k].describe()
    deltas = {}
    for k in p_raw:
        if k == chosen:
            continue
        deltas[k] = {w: VM.paired_logloss_delta(yse[sel_m], pp[k][sel_m], pp[chosen][sel_m], mids_se[sel_m])
                     for w, pp in (("raw", p_raw), ("cal", p_cal))}
    lap("select_calibrate", t0)

    # ---- candidate bundles (side marker)
    t0 = time.time()
    bundles: Dict[str, Any] = {}
    for d_old in (out / "candidates", out / "recalibrated"):
        if d_old.exists():                                 # a previous run's bundles never survive next to this run's
            shutil.rmtree(d_old)
    for k, m in models.items():
        extra = {"patch_train": TRAIN_PATCH, "patch_calibration": SELECT_PATCH, "calibration_rows": "V_CAL",
                 "cv": {kk: vv for kk, vv in cv[k].items() if kk != "fits"}, "smoke": bool(args.smoke),
                 "budget": budget, "state_v3_name_hash": STATE_V3_NAME_HASH,
                 "v_revision_record_sha256": vrev["sha256"]}
        bundles[k] = VM.save_bundle(out / "candidates" / k, m, cals[k], state_names, fill, extra, side_marker=True)
    lap("bundles", t0)

    # ---- martingale checks (15.15)
    t0 = time.time()
    mart: Dict[str, Any] = {"n": int(len(M0))}
    mart_pred: Dict[str, np.ndarray] = {}
    age_i, min_i = state_names.index(VM.AGE_COLUMN), state_names.index(VM.MINUTE_COLUMN)
    mids_m = oos = strata = None
    qidx = [state_names.index(c) for c in q_columns(state_names)]

    def mart_checks(p0: np.ndarray, p1: np.ndarray, n_boot: int) -> Dict[str, Any]:
        dv = p1 - p0
        eff = VM.efficiency_regression(dv, M0[:, qidx], mids_m, oos, n_boot=n_boot)
        return {"a_means": VM.martingale_means(dv, mids_m, strata, MART_BANDS), "b_efficiency": eff,
                "b_p_primary_wild_bootstrap": eff.get("p_value_primary"), "b_p_secondary_F": eff["p_value_F"]}

    def mart_log(name: str, r: Mapping[str, Any], n_boot: int) -> None:
        pb = r["b_p_primary_wild_bootstrap"]
        eff = r["b_efficiency"]
        log(f"martingale {name}: all-pass {r['a_means']['all_pass']}; "
            f"{ {s_: (round(v.get('mean', float('nan')), 5), v.get('pass')) for s_, v in r['a_means']['strata'].items()} }; "
            f"(b) primary wild-bootstrap p={'n/a' if pb is None else format(pb, '.3g')} "
            f"(B={n_boot}), secondary F p={eff['p_value_F']:.3g} k={eff['k_used']} R2oos={eff.get('r2_oos_mean')}")

    if len(M0):
        mids_m = mmart["match_id"].astype(str).to_numpy()
        strata = {"all": np.ones(len(M0), bool), "frame_update_yes": mmart["frame_update"].to_numpy() == 1,
                  "frame_update_no": mmart["frame_update"].to_numpy() == 0,
                  "recent_deaths_ge3": mmart["recent_deaths_ge3"].to_numpy() == 1}
        oos = VM.per_match(mids_m, VM.oos_half).astype(int)
        for k, m in models.items():
            p0 = cals[k].predict(m.predict_proba(M0))
            p1 = cals[k].predict(m.predict_proba(M1))
            mart_pred[f"v0_{k}"], mart_pred[f"v1_{k}"] = p0, p1
            nb = budget["mart_boot"] if k == chosen else 0
            mart[k] = {"primary": k == chosen, **mart_checks(p0, p1, nb)}
            mart_log(k, mart[k], nb)
        mart["strata_bands"] = MART_BANDS
        mart["q_columns"] = len(qidx)
        mart["b_primary_test"] = ("wild cluster (match) bootstrap p of the CR1 Wald statistic, chosen V only; "
                                  "the CR1 F test is secondary (author pre-decision)")
        mart["b_boot_replicates"] = int(budget["mart_boot"])
    lap("martingale", t0)

    # ---- R2: conditional recalibration of the chosen side-marker V (pre-specified trigger / model / adoption)
    t0 = time.time()
    ll_side_sel = VM.log_loss(yse[sel_m], p_cal[chosen][sel_m])
    recal: Dict[str, Any] = {"record": {"path": vrev["path"], "sha256": vrev["sha256"]}, "of": chosen,
                             "triggered": False, "fitted": False, "adopted": False,
                             "side_marker_v": {"select_logloss": ll_side_sel,
                                               "martingale": {kk: vv for kk, vv in (mart.get(chosen) or {}).items()
                                                              if kk != "primary"} or None}}
    if len(M0):
        trig = VM.recal_trigger(mart[chosen]["a_means"], mart[chosen]["b_p_primary_wild_bootstrap"])
    else:
        trig = {"triggered": False, "note": "no martingale rows: the trigger cannot be evaluated"}
    recal["trigger"] = trig
    recal["triggered"] = bool(trig["triggered"])
    log(f"recalibration trigger: {trig}")
    rc = None
    p_rc = None
    if trig["triggered"]:
        try:
            rc = VM.LogitRecalibrator().fit(p_cal[chosen][cal_m], Xse[cal_m, age_i], Xse[cal_m, min_i], yse[cal_m])
        except ValueError as e:                            # technical: unidentifiable (thin age bin; smoke only)
            recal["not_fitted_reason"] = str(e)
            log(f"recalibration NOT fitted: {e}")
    if rc is not None:
        recal["fitted"] = True
        recal["model"] = rc.to_dict()
        p_rc = rc.predict(p_cal[chosen], Xse[:, age_i], Xse[:, min_i])
        ll_rc_sel = VM.log_loss(yse[sel_m], p_rc[sel_m])
        adopt = VM.recal_adopt(ll_rc_sel, ll_side_sel)
        recal["adoption"] = adopt
        recal["adopted"] = bool(adopt["adopted"])
        recal["e4"] = {"side_marker": e4[chosen]["cal"],
                       "recalibrated": {sp: VM.metrics_by_bins(yse[mm], p_rc[mm], t_se[mm])
                                        for sp, mm in (("select", sel_m), ("cal", cal_m))},
                       "delta_select_recalibrated_minus_side_marker": VM.paired_logloss_delta(
                           yse[sel_m], p_rc[sel_m], p_cal[chosen][sel_m], mids_se[sel_m])}
        if len(M0):
            q0 = rc.predict(mart_pred[f"v0_{chosen}"], M0[:, age_i], M0[:, min_i])
            q1 = rc.predict(mart_pred[f"v1_{chosen}"], M1[:, age_i], M1[:, min_i])
            mart_pred[f"v0_{chosen}_recal"], mart_pred[f"v1_{chosen}_recal"] = q0, q1
            rmart = mart_checks(q0, q1, budget["mart_boot"])
            mart_log(f"{chosen} recalibrated", rmart, budget["mart_boot"])
            recal["recalibrated_v"] = {"select_logloss": ll_rc_sel, "martingale": rmart}
            again = VM.recal_trigger(rmart["a_means"], rmart["b_p_primary_wild_bootstrap"])
            recal["recalibrated_v"]["still_fails"] = bool(again["triggered"])
            recal["recalibrated_v"]["fails_detail"] = again
        else:
            recal["recalibrated_v"] = {"select_logloss": ll_rc_sel, "martingale": None}
        extra = {"patch_train": TRAIN_PATCH, "patch_calibration": SELECT_PATCH, "calibration_rows": "V_CAL",
                 "recalibration_rows": "V_CAL", "cv": {kk: vv for kk, vv in cv[chosen].items() if kk != "fits"},
                 "smoke": bool(args.smoke), "budget": budget, "state_v3_name_hash": STATE_V3_NAME_HASH,
                 "v_revision_record_sha256": vrev["sha256"], "recalibrated_from_bundle_sha256":
                     bundles[chosen]["bundle_sha256"], "recalibration_adopted": recal["adopted"]}
        rb = VM.save_bundle(out / "recalibrated" / chosen, models[chosen], cals[chosen], state_names, fill, extra,
                            side_marker=True, recalibrator=rc)
        recal["bundle"] = {"dir": str(out / "recalibrated" / chosen), "bundle_sha256": rb["bundle_sha256"]}
        log(f"recalibration: V_SELECT log loss {ll_rc_sel:.6f} vs side-marker {ll_side_sel:.6f}; adopted "
            f"{recal['adopted']}; bundle {rb['bundle_sha256']}")
    recal["no_further_tuning"] = vrev["content"].get("no_further_tuning")
    lap("recalibration", t0)

    # ---- freeze (after the martingale checks: the frozen variant depends on R2)
    t0 = time.time()
    variant = "side_marker_recalibrated" if recal["adopted"] else "side_marker"
    src_dir = (out / "recalibrated" / chosen) if recal["adopted"] else (out / "candidates" / chosen)
    frozen_sha = recal["bundle"]["bundle_sha256"] if recal["adopted"] else bundles[chosen]["bundle_sha256"]
    p_final = p_rc if recal["adopted"] else p_cal[chosen]
    frozen_dir = out / "V_frozen"
    # a previous run's frozen V / manifest / note in <out> must never survive next to this run's outputs
    if frozen_dir.exists():
        shutil.rmtree(frozen_dir)
    for stale in (out / "frozen_manifest.json", out / "V_NOT_FROZEN.md"):
        if stale.exists():
            stale.unlink()
    if not_frozen_reason is None:
        shutil.copytree(src_dir, frozen_dir)
        reload_dir = frozen_dir
    else:                                                  # F12 stop / F13 pilot: nothing is frozen
        write_not_frozen(out, not_frozen_reason, stop, pilot_info)
        reload_dir = src_dir
    V = VM.load_v(reload_dir, expected_sha256=frozen_sha, expected_columns_hash=STATE_V3_NAME_HASH)
    if not V.side_marker or V.recalibrated != recal["adopted"]:
        raise RuntimeError(f"frozen V structure {V.structure()} is not the {variant} V")
    nchk = min(5000, len(Xse))
    chk = V.predict(Xse[:nchk, :ns])                       # StateV3 columns only: side = +1 appended by predict()
    reload_diff = float(np.max(np.abs(chk - p_final[:nchk]))) if nchk else 0.0
    if reload_diff > 1e-6:
        raise RuntimeError(f"frozen V reload differs from the in-memory model by {reload_diff}")
    frozen_structure = {**V.structure(), "variant": variant}
    lap("freeze", t0)
    if not_frozen_reason is None:
        log(f"frozen {chosen} ({variant}): {frozen_sha} (reload max diff {reload_diff:.2e})")
    else:
        log(f"NOT frozen ({not_frozen_reason}): chosen {chosen} ({variant}) bundle {frozen_sha} kept for diagnosis "
            f"only (reload max diff {reload_diff:.2e}); see V_NOT_FROZEN.md")

    # ---- write predictions and reports
    t0 = time.time()
    pv = pd.DataFrame({"match_id": mids_se, "t": t_se, "bucket": mse["bucket"].to_numpy(), "split": split,
                       "y_blue_win": yse.astype(np.int8)})
    for k in p_raw:
        pv[f"p_raw_{k}"] = p_raw[k]
        pv[f"p_cal_{k}"] = p_cal[k]
    if p_rc is not None:
        pv[f"p_recal_{chosen}"] = p_rc
    pv["p_frozen_variant"] = p_final
    pv.to_parquet(out / "predictions_v_15.15.parquet", index=False)
    if len(M0):
        pm = mmart.drop(columns=["row"], errors="ignore").copy()
        for c, v in mart_pred.items():
            pm[c] = v
        pm.to_parquet(out / "predictions_mart_15.15.parquet", index=False)
    lap("write", t0)

    total = round(time.time() - t_start, 2)
    report = {"script": "scripts/exact_v4/ev4_03_fit_v.py", "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
              "smoke": bool(args.smoke), "pilot": pilot, "pilot_subset": pilot_info,
              "frozen": not_frozen_reason is None, "not_frozen_reason": not_frozen_reason,
              "budget": budget, "threads": threads, "candidates": cands,
              "decisions": decisions_block(predec, vrev), "grids": VM.GRIDS, "census": census, "inputs": inputs,
              "input_checks": checks_for_report(checks),
              "cv": cv, "final_fits": {k: m.fit_info for k, m in models.items()}, "selection": selection,
              "stop_rule": stop,
              "e4": e4, "deltas_vs_chosen_select": deltas, "gold_baseline": gold.describe(),
              "prev_evaluator": PREV_EVALUATOR, "martingale": mart, "bundles": bundles,
              "v_revision": {"record": {"path": vrev["path"], "sha256": vrev["sha256"], "content": vrev["content"]},
                             "side_marker": {"added": True, "column": VM.SIDE_COLUMN, "n_model_columns": len(names),
                                             "encoding": "+1 original rows, -1 team-swapped rows",
                                             "predict_default": VM.SIDE_PREDICT_DEFAULT,
                                             "bundle_format": VM.BUNDLE_FORMAT},
                             "recalibration": recal},
              "frozen_v": {"kind": chosen, "bundle_sha256": frozen_sha, **frozen_structure},
              "reload_max_abs_diff": reload_diff, "timings_s": timings, "memory": mem, "seconds_total": total,
              "peak_memory_gb": memory_info().get("peak_gb")}
    EX.write_json_atomic(out / "report_e4.json", report)
    files = {p.relative_to(out).as_posix(): VM.sha256_file(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name not in ("frozen_manifest.json", "run.log")}
    manifest = {"script": report["script"], "created_utc": report["created_utc"], "smoke": bool(args.smoke),
                "pilot": pilot, "pilot_subset": pilot_info,
                "frozen": not_frozen_reason is None, "not_frozen_reason": not_frozen_reason,
                "chosen": chosen, "selection": selection,
                "V_frozen": ({"dir": str(frozen_dir), "bundle_sha256": frozen_sha,
                              "kind": chosen, "variant": variant, "bundle_format": VM.BUNDLE_FORMAT,
                              "side_marker": True, "recalibrated": bool(recal["adopted"]),
                              "load": "ev4_v_models.assert_v_usable(frozen_manifest) then "
                                      "ev4_v_models.load_v(dir, expected_sha256=bundle_sha256)"}
                             if not_frozen_reason is None else None),
                "v_revision": {"record": {"path": vrev["path"], "sha256": vrev["sha256"]},
                               "side_marker": True, "variant": variant,
                               "chosen_bundle_sha256": frozen_sha,
                               "recalibration": {"triggered": recal["triggered"], "fitted": recal["fitted"],
                                                 "adopted": recal["adopted"], "trigger": recal["trigger"],
                                                 "adoption": recal.get("adoption"),
                                                 "not_fitted_reason": recal.get("not_fitted_reason"),
                                                 "bundle_sha256": (recal.get("bundle") or {}).get("bundle_sha256"),
                                                 "side_marker_bundle_sha256": bundles[chosen]["bundle_sha256"]}},
                "candidates": {k: b["bundle_sha256"] for k, b in bundles.items()},
                "state_v3_name_hash": STATE_V3_NAME_HASH, "inputs": inputs, "guards": guard,
                "decisions": decisions_block(predec, vrev), "input_checks": report["input_checks"], "stop_rule": stop,
                "grids": VM.GRIDS, "budget": budget, "code_sha256": code_hashes(), "git": git_state(),
                "files_sha256": files}
    EX.write_json_atomic(out / "frozen_manifest.json", manifest)
    log(f"done in {total} s; peak memory {report['peak_memory_gb']} GB")
    return report


if __name__ == "__main__":
    main()
