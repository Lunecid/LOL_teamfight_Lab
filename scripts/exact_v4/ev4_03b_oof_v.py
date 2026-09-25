"""v4-exact stage 2, R5b: out-of-fold (OOF) V for the 15.14 training labels (ev4_04_labels.OOF_SPEC).

Plan: outputs/diag_survival_dbscan_20260925/docs/REESTIMATION_PLAN_V4_EXACT_20260925.md, section 5 ("SVI(학습은 교차
적합)").  Runs AFTER a finished ev4_03_fit_v.py run (frozen_manifest.json present) and before ev4_04_labels.py
labels --patch 15.14.  ev4_03_fit_v.py / ev4_v_models.py are imported, never changed; frozen_manifest.json is NOT
modified.

Implementation decisions (2026-09-26, before any OOF result; recorded here, in DECISIONS and in oof_manifest.json):
  O1  Nothing is re-selected.  kind = frozen_manifest 'chosen' (== V_frozen.kind == V_frozen/bundle.json kind);
      hyperparameter = the V_frozen/bundle.json model spec (logistic C / lgbm num_leaves / mlp weight_decay), which
      must equal the bundle's extra.cv.chosen and report_e4.json cv[kind].chosen.  Budgets (logistic maxiter, lgbm
      rounds / patience, mlp epochs / patience) = frozen_manifest 'budget'; threads = report_e4.json 'threads'
      (LightGBM's deterministic mode depends on it) unless --threads (logged).
  O2  Rows.  The 15.14 / 15.15 extracts of the fit: <extract>/manifest.json sha256 == frozen_manifest
      inputs.{train,select}.manifest_sha256; every chunk file verified (ev4_03_fit_v.verify_files); rows loaded with
      ev4_03_fit_v.load_v_rows (same order as the fit).  NaN fill = 15.14 column means over ALL 15.14 rows, as in
      the fit and its CV; it must equal the fill stored in V_frozen/prep.npz.  The census (rows, matches, rows per
      fold, inner rows, V_SELECT / V_CAL rows, swapped rows) must equal report_e4.json census.
  O3  Folds = ev4_v_models.cv_fold(match) (fit-V's CV).  Fold model k = the chosen kind at the chosen value trained
      like fit-V's CV fit of fold k: logistic on all rows of the matches with fold != k (standardiser pooled over
      swap pairs on those rows); lgbm / mlp on those matches minus the inner 10 % (ev4_v_models.inner_holdout),
      early-stopped on the inner ones (lgbm Dataset built once on all 15.14 rows, as in the fit; its bin edges are
      label-free); mlp standardiser on the fit rows and the FINAL seeds 0, 1, 2 with logits averaged (fit-V's CV used
      seed 0 only, so these five 3-seed fits are new fits).
  O4  Team swap.  logistic / lgbm: the rows swapped in place by ev4_v_models.swap_mask(n) (default_rng(20260925); same
      row order, so the same mask as the fit); mlp: a fresh 50 % of each minibatch (ev4_v_models.MLPModel).
  O5  Calibration: one PositiveSlopeSigmoid per fold fitted on V_CAL (15.15) with unit weights; bundle =
      ev4_v_models.save_bundle with extra {"oof_heldout_fold": k, ...}; each bundle is reloaded by sha256 and must
      reproduce the in-memory predictions (<= 1e-6).
  O6  fold_k/train_match_ids.txt = every 15.14 match with fold != k: the fit matches AND the early-stopping matches
      (both informed the model), so the labels' coverage rule (a match is in the lists of its 4 other folds) holds.
  O7  Gates: no fold-k match in fold k's training ids; coverage of the 15.14 V-row matches = 100 % (each in the lists
      of its 4 other folds, none in its own); kind / hyperparameter / seeds of each saved bundle = the chosen ones.
      Reported, not gates: coverage of the 15.14 engagement-row matches (the rows the labels score); CV reproduction
      (logistic / lgbm: fold-k held-out log loss on the swapped rows and the lgbm best iteration / logistic
      iterations vs report_e4.json cv fits of the chosen value; mlp: vs the 1-seed CV value, informational); agreement
      of every fold model with the frozen V on a V_SELECT sample (Pearson r of p and of logit p, mean / max |dp|,
      log loss) - V_SELECT is used only for this report, never for a fit or a choice.
  O8  Disclosures (oof_manifest.json 'disclosures'): the hyperparameter was chosen by fit-V's 5-fold CV over ALL 15.14
      folds (pooled OOF log loss), so each fold model is out-of-fold for its weights but not for that one grid choice;
      the NaN fill means and the lgbm bin edges use all 15.14 rows (label-free); calibration uses V_CAL (15.15).
  O9  Guards: ev4_common.predecisions_info() first; grid_guard.forbid_grid(); ev4_03_fit_v.check_inputs (split guard
      15.14 / 15.15 before any array, StateV3, sample runs, remake rule, h linkage, code drift unless
      --allow-code-drift) and check_params; ev4_v_models.assert_v_usable (strict; with --smoke allow_smoke=True);
      a pilot fit-V run is always refused (its rows are a subset); a smoke fit-V run or sample extract needs --smoke
      and gives a smoke OOF set (ev4_04_labels accepts it only with --smoke); an existing oof_v/ or sidecar needs
      --force (both are removed before the new fit).

Outputs (inside the fit-V directory, the only place ev4_04_labels / ev4_r1_record look):
  <fit_v>/oof_v/fold_k/            V bundle of fold k (+ train_match_ids.txt), k = 0..4
  <fit_v>/oof_v/oof_manifest.json  ev4_04_labels.OOF_SPEC manifest (written by ev4_04_labels.write_oof_manifest) plus
                                   disclosures, provenance and the verification summary
  <fit_v>/oof_v/oof_report.json    per-fold fit info, CV reproduction, coverage, agreement, timings (sha256 in the
                                   manifest)
  <fit_v>/oof_v/run.log
  <fit_v>/oof_v_block.json         written LAST: the 'oof_v' block (manifest sha256, fold bundle / id-list sha256)
                                   bound to sha256(frozen_manifest.json); read by ev4_04_labels.check_oof and
                                   ev4_r1_record.oof_block when frozen_manifest.json has no 'oof_v' block

CLI
  python scripts/exact_v4/ev4_03b_oof_v.py --fit-v <stage2>/fit_v --train <extract>/15.14 --select <extract>/15.15
         [--threads N] [--agree-rows 20000] [--allow-code-drift] [--force] [--smoke]
  python scripts/exact_v4/ev4_03b_oof_v.py --fit-v <stage2>/fit_v --plan     runtime projection from report_e4.json
                                                                               (reads no array)
  exit codes: 0 done; 1 / other: refused or crashed
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"                                  # as ev4_03_fit_v (same BLAS threading as the fit)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FV = _load("ev4_03_fit_v")                                 # import only (running job's code; never modified here)
LB = _load("ev4_04_labels")
EX = FV.EX

TRAIN_PATCH, SELECT_PATCH = "15.14", "15.15"
N_FOLDS = LB.N_FOLDS
OOF_DIRNAME = "oof_v"
REPORT_NAME = "oof_report.json"
AGREE_ROWS = 20_000
RELOAD_TOL = 1e-6
REPRO_WARN = 1e-6
HP_KEY = {"logistic": "C", "lgbm": "num_leaves", "mlp": "weight_decay"}

DECISIONS = {
    "O1_no_reselection": "kind = frozen_manifest chosen; hyperparameter = V_frozen bundle model spec (== bundle "
                         "extra.cv.chosen == report_e4 cv[kind].chosen); budgets = frozen_manifest budget; threads = "
                         "report_e4 threads unless --threads",
    "O2_rows": "same 15.14/15.15 extracts (manifest sha256 == frozen_manifest inputs), chunk files verified, "
               "ev4_03_fit_v.load_v_rows order; NaN fill = all-15.14 column means == V_frozen prep.npz; census == "
               "report_e4 census",
    "O3_folds": "fold = ev4_v_models.cv_fold; fold k trained like fit-V's CV fit of fold k (logistic all rows of "
                "fold != k; lgbm/mlp fold != k minus inner 10 %, early-stopped on inner; lgbm Dataset on all 15.14 "
                "rows; mlp FINAL seeds 0,1,2)",
    "O4_swap": "logistic/lgbm: ev4_v_models.swap_mask(n) rows swapped in place (same mask as the fit); mlp: 50 % of "
               "each minibatch",
    "O5_calibration": "per fold PositiveSlopeSigmoid on V_CAL (15.15), unit weights; bundle extra oof_heldout_fold; "
                      f"reload check <= {RELOAD_TOL}",
    "O6_train_ids": "fold_k/train_match_ids.txt = all 15.14 matches with fold != k (fit + early-stopping matches)",
    "O7_gates": "no own-fold id in any list; 15.14 V-row match coverage 100 %; saved kind / hyperparameter / seeds == "
                "chosen.  Reported only: engagement-match coverage, CV reproduction, agreement with frozen V on a "
                "V_SELECT sample",
    "O8_disclosure": "hyperparameter chosen by fit-V CV over all 15.14 folds; NaN fill and lgbm bins from all 15.14 "
                     "rows (label-free); calibration on V_CAL",
    "O9_guards": "predecisions; forbid_grid; ev4_03 check_inputs + check_params; assert_v_usable (strict, --smoke: "
                 "allow_smoke); pilot fit-V refused; smoke inputs need --smoke; existing outputs need --force",
}


class Log(FV.Log):
    def __call__(self, msg: str) -> None:
        line = f"[ev4_03b {time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ============================================================================ fit-V run
def _hp_from_bundle(kind: str, bundle: Mapping[str, Any]) -> Dict[str, Any]:
    spec = bundle.get("model") or {}
    if kind == "logistic":
        return {"C": float(spec["C"])}
    if kind == "lgbm":
        return {"num_leaves": int(spec["num_leaves"])}
    if kind == "mlp":
        return {"weight_decay": float(spec["weight_decay"])}
    raise RuntimeError(f"unknown V kind {kind}")


def _same_value(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def read_fit_v(fit_v: Path, smoke: bool) -> Dict[str, Any]:
    """Frozen manifest, usable frozen V, report, chosen kind / hyperparameter / budget (decision O1).  Reads no array."""
    from gameplay.state_value_v3 import STATE_V3_NAME_HASH
    fit_v = Path(fit_v)
    vpath = fit_v / "frozen_manifest.json"
    if not vpath.is_file():
        raise SystemExit(f"{vpath} missing: run (or finish) ev4_03_fit_v.py first")
    vman = json.loads(vpath.read_text(encoding="utf-8"))
    if vman.get("pilot"):
        raise SystemExit("the fit-V run is a pilot (--pilot-matches): its rows are a subset; no OOF set is built")
    if vman.get("smoke") and not smoke:
        raise SystemExit("the fit-V run is a smoke run; only --smoke may build its OOF set")
    vf = VM.assert_v_usable(vman, allow_smoke=smoke)        # stop rule / not frozen / smoke without --smoke refuse
    if vman.get("state_v3_name_hash") != STATE_V3_NAME_HASH:
        raise RuntimeError(f"fit-V used StateV3 {vman.get('state_v3_name_hash')}, code has {STATE_V3_NAME_HASH}")
    kind = vman.get("chosen")
    if kind not in VM.SIMPLICITY_ORDER or vf.get("kind") != kind:
        raise RuntimeError(f"chosen kind {kind!r} / V_frozen kind {vf.get('kind')!r} disagree")
    frozen_dir = fit_v / "V_frozen"                        # by location (the manifest's absolute dir may be stale)
    bj = frozen_dir / "bundle.json"
    if not bj.is_file() or VM.sha256_file(bj) != vf["bundle_sha256"]:
        raise RuntimeError(f"{bj} missing or not the V_frozen bundle_sha256 of the manifest")
    bundle = json.loads(bj.read_text(encoding="utf-8"))
    if bundle.get("kind") != kind:
        raise RuntimeError(f"V_frozen bundle kind {bundle.get('kind')} != chosen {kind}")
    hp = _hp_from_bundle(kind, bundle)
    key = HP_KEY[kind]
    rpath = fit_v / "report_e4.json"
    if not rpath.is_file():
        raise RuntimeError(f"{rpath} missing (needed for the census, threads and the CV reproduction check)")
    report = json.loads(rpath.read_text(encoding="utf-8"))
    cv_b = (((bundle.get("extra") or {}).get("cv")) or {}).get("chosen")
    cv_r = ((report.get("cv") or {}).get(kind) or {}).get("chosen")
    if not (_same_value(cv_b, hp[key]) and _same_value(cv_r, hp[key])):
        raise RuntimeError(f"chosen {key}: bundle model {hp[key]}, bundle extra.cv {cv_b}, report_e4 cv {cv_r} differ")
    if kind == "mlp" and [int(s) for s in (bundle.get("model") or {}).get("seeds", [])] != list(VM.MLP_SEEDS_FINAL):
        raise RuntimeError(f"V_frozen mlp seeds {(bundle.get('model') or {}).get('seeds')} != {VM.MLP_SEEDS_FINAL}")
    budget = dict(vman.get("budget") or {})
    need = ("lgbm_rounds", "lgbm_patience", "mlp_epochs", "mlp_patience", "logistic_maxiter")
    if any(k not in budget for k in need):
        raise RuntimeError(f"frozen_manifest budget lacks {[k for k in need if k not in budget]}")
    if not smoke:
        full = {"lgbm_rounds": VM.LGBM_MAX_ROUNDS, "lgbm_patience": VM.LGBM_PATIENCE, "mlp_epochs": VM.MLP_MAX_EPOCHS,
                "mlp_patience": VM.MLP_PATIENCE, "logistic_maxiter": VM.GRIDS["logistic"]["fixed"]["maxiter"]}
        if {k: budget[k] for k in need} != full:
            raise RuntimeError(f"non-smoke fit-V budget {budget} differs from the fixed budget {full}")
    return {"dir": fit_v, "vpath": vpath, "vman": vman, "vman_sha256": VM.sha256_file(vpath), "vf": vf,
            "frozen_dir": frozen_dir, "bundle": bundle, "kind": kind, "hp": hp, "report": report,
            "report_path": rpath, "report_sha256": VM.sha256_file(rpath), "budget": budget,
            "threads": int(report.get("threads") or VM.MAX_THREADS)}


def cv_fit_entry(report: Mapping[str, Any], kind: str, hp: Mapping[str, Any], k: int) -> Optional[Dict[str, Any]]:
    """fit-V's CV fit of fold k at the chosen value (report_e4.json cv[kind].fits)."""
    key = HP_KEY[kind]
    for f in ((report.get("cv") or {}).get(kind) or {}).get("fits") or []:
        if int(f.get("fold", -1)) == k and _same_value(f.get(key), hp[key]):
            return dict(f)
    return None


# ============================================================================ runtime projection
def project_runtime(fv: Mapping[str, Any]) -> Dict[str, Any]:
    """Projection of this script's run time from the fit-V report (its own CV / final fit seconds)."""
    rep, kind, hp = fv["report"], fv["kind"], fv["hp"]
    tim = rep.get("timings_s") or {}
    fits = [cv_fit_entry(rep, kind, hp, k) for k in range(N_FOLDS)]
    cv_sec = [float(f.get("seconds") or 0.0) for f in fits if f]
    load = float(tim.get("load") or 0.0)
    out: Dict[str, Any] = {"kind": kind, "hyperparameter": hp, "fit_v_load_s": load,
                           "fit_v_cv_fit_s_chosen_value": cv_sec}
    if kind in ("logistic", "lgbm"):
        fit_s = sum(cv_sec)
        extra = 0.0
        if kind == "lgbm":                                   # Dataset construction: lgbm block minus all its fits
            all_cv = sum(float(f.get("seconds") or 0.0) for f in (rep.get("cv") or {}).get("lgbm", {}).get("fits", []))
            final = float(((rep.get("final_fits") or {}).get("lgbm") or {}).get("seconds") or 0.0)
            extra = max(0.0, float(tim.get("lgbm") or 0.0) - all_cv - final)
        out.update(basis="the five fit-V CV fits of the chosen value are refitted (same rows, seeds, budget)",
                   fits_s=fit_s, dataset_s=extra, projected_s=round(load + fit_s + extra, 1))
    else:
        final = (rep.get("final_fits") or {}).get("mlp") or {}
        ep_final = [int(r.get("epochs_run", 0)) for r in (final.get("runs") or [])] or             [int(fv["budget"]["mlp_epochs"])] * len(VM.MLP_SEEDS_FINAL)
        ep_cv = [sum(int(r.get("epochs_run", 0)) for r in (f.get("runs") or [])) for f in fits if f]
        per_epoch = [cv_sec[i] / max(ep_cv[i], 1) for i in range(len(cv_sec))]   # CV fit rows == fold fit rows
        fit_s = sum(pe * sum(ep_final) for pe in per_epoch)
        upper = sum(pe * len(VM.MLP_SEEDS_FINAL) * int(fv["budget"]["mlp_epochs"]) for pe in per_epoch)
        out.update(basis="per fold: fit-V CV seconds per epoch (same fit rows, seed 0) x the epochs the final 3-seed "
                         "fit ran; upper = x 3 seeds x max epochs",
                   seconds_per_epoch=per_epoch, final_epochs=ep_final,
                   fits_s=fit_s, upper_s=round(load + upper, 1), projected_s=round(load + fit_s, 1))
    return out


# ============================================================================ fitting
def fit_fold(kind: str, k: int, X: np.ndarray, y: np.ndarray, folds: np.ndarray, inner: np.ndarray,
             hp: Mapping[str, Any], budget: Mapping[str, Any], threads: int, names: Sequence[str], perm: np.ndarray,
             ds: Any, log) -> Tuple[Any, Dict[str, np.ndarray]]:
    """Fold model k exactly like fit-V's CV fit of fold k (decisions O3 / O4; mlp with the final seeds)."""
    if kind == "logistic":
        tr = np.flatnonzero(folds != k)
        mu, sd = VM.pooled_standardizer(X, tr, VM.swap_pairs(perm))
        m = VM.LogisticModel(hp["C"]).fit(X, y, tr, mu, sd, maxiter=int(budget["logistic_maxiter"]))
        return m, {"fit": tr, "stop": np.zeros(0, dtype=np.int64)}
    tr = np.flatnonzero((folds != k) & ~inner)
    st = np.flatnonzero((folds != k) & inner)
    if kind == "lgbm":
        m = VM.LGBMModel(hp["num_leaves"], int(budget["lgbm_rounds"]), int(budget["lgbm_patience"]), threads)
        return m.fit(ds, tr, st), {"fit": tr, "stop": st}
    mu, sd = VM.pooled_standardizer(X, tr, VM.mlp_groups(names, perm))
    m = VM.MLPModel(hp["weight_decay"], names, seeds=VM.MLP_SEEDS_FINAL, max_epochs=int(budget["mlp_epochs"]),
                    patience=int(budget["mlp_patience"]), threads=threads)
    m.fit(X, y, tr, st, mu, sd, log=log)
    return m, {"fit": tr, "stop": st}


def _fit_info(m) -> Dict[str, Any]:
    fi = dict(m.fit_info)
    if "runs" in fi:
        fi["runs"] = [{k: v for k, v in r.items() if k != "history"} for r in fi["runs"]]
    return fi


def _logit(p: np.ndarray) -> np.ndarray:
    return VM.clipped_logit(np.asarray(p, dtype=np.float64))


def agreement(p: np.ndarray, p_ref: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Agreement of a fold model with the frozen V on the same rows (report only)."""
    if len(p) < 3:
        return {"n": int(len(p))}
    d = np.abs(p - p_ref)
    r = float(np.corrcoef(p, p_ref)[0, 1]) if np.std(p) > 0 and np.std(p_ref) > 0 else float("nan")
    lr = float(np.corrcoef(_logit(p), _logit(p_ref))[0, 1]) if np.std(p) > 0 and np.std(p_ref) > 0 else float("nan")
    return {"n": int(len(p)), "pearson_r": r, "pearson_r_logit": lr, "mean_abs_diff": float(d.mean()),
            "p95_abs_diff": float(np.quantile(d, 0.95)), "max_abs_diff": float(d.max()),
            "mean_diff": float(np.mean(p - p_ref)), "logloss": VM.log_loss(y, p), "logloss_frozen": VM.log_loss(y, p_ref)}


def coverage_check(ids: Mapping[int, Sequence[str]], mids: Sequence[str]) -> Dict[str, Any]:
    """Every match: in the training lists of its 4 other folds and not in its own."""
    sets = {k: set(v) for k, v in ids.items()}
    u = sorted(set(str(m) for m in mids))
    full, own, missing = 0, [], []
    for m in u:
        f = VM.cv_fold(m)
        if m in sets[f]:
            own.append(m)
        if all(m in sets[j] for j in sets if j != f):
            full += 1
        else:
            missing.append(m)
    return {"n": len(u), "full": int(full), "rate": (full / len(u)) if u else None, "in_own_fold": len(own),
            "missing_examples": missing[:5], "own_examples": own[:5]}


def eng_match_ids(d: Path, man: Mapping[str, Any]) -> List[str]:
    """15.14 engagement-row matches (the rows ev4_04 labels) from the verified chunk index tables."""
    import pandas as pd
    out: set = set()
    for c in FV.chunk_ids(man):
        p = Path(d) / f"chunk_{int(c):05d}_eng.parquet"
        if p.is_file():
            out.update(pd.read_parquet(p, columns=["match_id"])["match_id"].astype(str).tolist())
    return sorted(out)


def code_hashes() -> Dict[str, str]:
    out = {"scripts/exact_v4/ev4_03b_oof_v.py": VM.sha256_file(HERE)}
    for n in ("ev4_03_fit_v.py", "ev4_v_models.py", "ev4_04_labels.py", "ev4_common.py", "ev4_02_extract.py"):
        out[f"scripts/exact_v4/{n}"] = VM.sha256_file(HERE.parent / n)
    return out


# ============================================================================ main
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-v", type=Path, required=True, help="ev4_03_fit_v output directory (frozen_manifest.json)")
    ap.add_argument("--train", type=Path, default=None, help="ev4_02 extract directory of 15.14 (the fit's)")
    ap.add_argument("--select", type=Path, default=None, help="ev4_02 extract directory of 15.15 (the fit's)")
    ap.add_argument("--threads", type=int, default=None, help="default: report_e4.json threads of the fit")
    ap.add_argument("--agree-rows", type=int, default=AGREE_ROWS, help="V_SELECT rows for the agreement report")
    ap.add_argument("--allow-code-drift", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace an existing oof_v/ and oof_v_block.json")
    ap.add_argument("--smoke", action="store_true", help="smoke fit-V run / sample extracts; smoke OOF set")
    ap.add_argument("--plan", action="store_true", help="print the runtime projection from report_e4.json and exit")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    t_start = time.time()
    predec = EC.predecisions_info()
    fit_v = Path(args.fit_v)
    if args.plan:
        fv = read_fit_v(fit_v, smoke=True if args.smoke else False)
        proj = project_runtime(fv)
        print(json.dumps(proj, indent=2, default=VM._json_default))
        return {"plan": proj}
    if args.train is None or args.select is None:
        raise SystemExit("--train and --select are required (the 15.14 / 15.15 extracts of the fit)")
    if args.agree_rows < 100:
        raise SystemExit("--agree-rows must be >= 100")
    from gameplay.grid_guard import forbid_grid
    with forbid_grid():
        # ---- guards before any array is read (O9)
        mt, ms, checks = FV.check_inputs(args.train, args.select, args.smoke, args.allow_code_drift)
        _, _params, guard = EX.load_params()
        FV.check_params([mt, ms], guard)
        fv = read_fit_v(fit_v, args.smoke)
        for role, d in (("train", args.train), ("select", args.select)):
            want = ((fv["vman"].get("inputs") or {}).get(role) or {}).get("manifest_sha256")
            got = VM.sha256_file(Path(d) / "manifest.json")
            if want != got:
                raise RuntimeError(f"{role} extract manifest {got} is not the one fit-V used ({want})")
        threads = int(args.threads) if args.threads is not None else fv["threads"]
        if not 1 <= threads <= VM.MAX_THREADS:
            raise SystemExit(f"--threads must be in [1, {VM.MAX_THREADS}]")
        oof_dir = fit_v / OOF_DIRNAME
        side = fit_v / LB.OOF_BLOCK_FILE
        if (oof_dir.exists() or side.exists()) and not args.force:
            raise SystemExit(f"{oof_dir} or {side} exists; pass --force to replace them")
        if side.exists():
            side.unlink()                                  # the sidecar goes first: no half-built set is ever linked
        if oof_dir.exists():
            shutil.rmtree(oof_dir)
        oof_dir.mkdir(parents=True)
        log = Log(oof_dir / "run.log")
        log(f"fit-V {fit_v} (smoke={fv['vman'].get('smoke')}); kind {fv['kind']} {fv['hp']}; budget {fv['budget']}; "
            f"threads {threads} (fit-V {fv['threads']}); smoke={args.smoke}")
        if threads != fv["threads"]:
            log(f"WARNING --threads {threads} differs from the fit's {fv['threads']}: LightGBM / BLAS results may not "
                "reproduce the CV fits bit for bit")
        if checks["code"]["drift_found"]:
            log(f"WARNING --allow-code-drift: {checks['code']['drift']}")
        result = _run(args, fv, mt, ms, checks, predec, oof_dir, side, threads, log, t_start)
    return result


def _run(args, fv, mt, ms, checks, predec, oof_dir: Path, side: Path, threads: int, log, t_start: float):
    import torch
    torch.set_num_threads(threads)
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH
    names = list(STATE_V3_COLUMNS)
    perm = VM.swap_permutation(names)
    kind, hp, budget, rep = fv["kind"], fv["hp"], fv["budget"], fv["report"]
    timings: Dict[str, float] = {}

    # ---- rows (O2)
    t0 = time.time()
    inputs = {"train": {"dir": str(args.train), **FV.verify_files(args.train, mt, log)},
              "select": {"dir": str(args.select), **FV.verify_files(args.select, ms, log)}}
    Xtr, mtr = FV.load_v_rows(args.train, mt)
    Xse, mse = FV.load_v_rows(args.select, ms)
    FV.assert_no_remakes({"train_v": mtr, "select_v": mse})
    fill, n_nan_tr = VM.nan_fill_values(Xtr)
    V0 = VM.load_v(fv["frozen_dir"], expected_sha256=fv["vf"]["bundle_sha256"], expected_columns_hash=STATE_V3_NAME_HASH)
    if V0.fill.shape != fill.shape or not np.array_equal(V0.fill, fill):
        raise RuntimeError("15.14 NaN fill differs from V_frozen/prep.npz: these are not the rows V was fitted on")
    VM.fill_nan_inplace(Xtr, fill)
    n_nan_se = VM.fill_nan_inplace(Xse, fill)
    ytr = mtr["y_blue_win"].to_numpy().astype(np.float64)
    yse = mse["y_blue_win"].to_numpy().astype(np.float64)
    mids_tr = mtr["match_id"].astype(str).to_numpy()
    mids_se = mse["match_id"].astype(str).to_numpy()
    folds = VM.per_match(mids_tr, VM.cv_fold).astype(int)
    inner = VM.per_match(mids_tr, VM.inner_holdout).astype(bool)
    split = VM.per_match(mids_se, VM.v_split)
    sel_m, cal_m = split == "select", split == "cal"
    smask = VM.swap_mask(len(ytr))
    census = {"train_rows": int(len(ytr)), "train_matches": int(len(set(mids_tr))),
              "train_rows_inner": int(inner.sum()), "train_fold_rows": np.bincount(folds, minlength=5).tolist(),
              "select_rows": int(sel_m.sum()), "cal_rows": int(cal_m.sum()), "swapped_rows": int(smask.sum())}
    rc = rep.get("census") or {}
    diff = {k: (v, rc.get(k)) for k, v in census.items() if rc.get(k) != v}
    if diff:
        raise RuntimeError(f"census differs from report_e4.json (row set / order not the fit's): {diff}")
    census.update(nan_filled={"train": int(n_nan_tr), "select": int(n_nan_se)})
    # 15.15: keep V_CAL and an agreement sample of V_SELECT only
    X_cal, y_cal = Xse[cal_m], yse[cal_m]
    sel_idx = np.flatnonzero(sel_m)
    if len(sel_idx) > args.agree_rows:
        sel_idx = np.sort(np.random.default_rng(VM.SEED).choice(sel_idx, args.agree_rows, replace=False))
    X_ag, y_ag = Xse[sel_idx], yse[sel_idx]
    del Xse
    p_frozen_ag = V0.predict(X_ag)
    timings["load"] = round(time.time() - t0, 2)
    log(f"loaded 15.14 V {Xtr.shape}, V_CAL {X_cal.shape}, agreement sample {X_ag.shape}; census {census}; "
        f"{FV.memory_info()}")

    # ---- fold fits (O3 - O6)
    swapped = kind in ("logistic", "lgbm")
    if swapped:
        VM.apply_swap_inplace(Xtr, ytr, smask, perm)
    ds = None
    if kind == "lgbm":
        t0 = time.time()
        ds = VM.LGBMModel.dataset(Xtr, ytr)
        timings["lgbm_dataset"] = round(time.time() - t0, 2)
    frozen_sha = fv["vf"]["bundle_sha256"]
    train_rule = ("logistic: all rows of the 15.14 matches with cv_fold != k" if kind == "logistic" else
                  "15.14 matches with cv_fold != k minus the inner 10 % (fit rows), early-stopped on the inner ones")
    fold_bundles: Dict[int, Dict[str, Any]] = {}
    fold_ids: Dict[int, List[str]] = {}
    per_fold: Dict[str, Any] = {}
    p_oof_raw = np.full(len(ytr), np.nan)
    p_oof_cal = np.full(len(ytr), np.nan)
    y_nat = ytr.copy()
    if swapped:
        y_nat[smask] = 1.0 - y_nat[smask]
    p_ag_folds = []
    for k in range(N_FOLDS):
        t0 = time.time()
        log(f"fold {k}: fitting {kind} {hp}")
        m, rows = fit_fold(kind, k, Xtr, ytr, folds, inner, hp, budget, threads, names, perm, ds, log)
        te = np.flatnonzero(folds == k)
        xt = Xtr[te]
        ll_fit_orient = VM.log_loss(ytr[te], m.predict_proba(xt))  # as fit-V's CV scored it (swapped for log/lgbm)
        if swapped:                                         # natural orientation for the pooled OOF report
            sw = smask[te]
            xt[sw] = xt[sw][:, perm]
        p_te = m.predict_proba(xt)
        del xt
        cal = VM.PositiveSlopeSigmoid().fit(m.predict_proba(X_cal), y_cal, np.ones(len(y_cal)))
        p_oof_raw[te] = p_te
        p_oof_cal[te] = cal.predict(p_te)
        tr_all = np.concatenate([rows["fit"], rows["stop"]])
        ids = sorted(set(mids_tr[tr_all].tolist()))
        if set(ids) != set(mids_tr[folds != k].tolist()):
            raise RuntimeError(f"fold {k}: training matches are not exactly the matches with cv_fold != {k}")
        bad = [x for x in ids if VM.cv_fold(x) == k]
        if bad:
            raise RuntimeError(f"fold {k}: {len(bad)} training matches of its own held-out fold ({bad[:3]})")
        fold_ids[k] = ids
        extra = {"oof_heldout_fold": k, "patch_train": TRAIN_PATCH, "patch_calibration": SELECT_PATCH,
                 "calibration_rows": "V_CAL", "hyperparameter": dict(hp), "train_rule": train_rule,
                 "n_fit_rows": int(len(rows["fit"])), "n_stop_rows": int(len(rows["stop"])),
                 "n_train_matches": len(ids), "smoke": bool(args.smoke), "budget": budget,
                 "state_v3_name_hash": STATE_V3_NAME_HASH, "frozen_bundle_sha256": frozen_sha,
                 "fit_v_frozen_manifest_sha256": fv["vman_sha256"]}
        b = VM.save_bundle(oof_dir / f"fold_{k}", m, cal, names, fill, extra)
        fold_bundles[k] = b
        Vk = VM.load_v(oof_dir / f"fold_{k}", expected_sha256=b["bundle_sha256"],
                       expected_columns_hash=STATE_V3_NAME_HASH)
        spec_hp = _hp_from_bundle(kind, json.loads((oof_dir / f"fold_{k}" / "bundle.json").read_text(encoding="utf-8")))
        if Vk.kind != kind or spec_hp != dict(hp):
            raise RuntimeError(f"fold {k} bundle is {Vk.kind} {spec_hp}, chosen {kind} {hp}")
        if kind == "mlp" and list(Vk.model.seeds) != list(VM.MLP_SEEDS_FINAL):
            raise RuntimeError(f"fold {k} mlp seeds {Vk.model.seeds}")
        p_ag = Vk.predict(X_ag)
        reload_diff = float(np.max(np.abs(p_ag - cal.predict(m.predict_proba(X_ag))))) if len(p_ag) else 0.0
        if reload_diff > RELOAD_TOL:
            raise RuntimeError(f"fold {k} bundle reload differs from the in-memory model by {reload_diff}")
        p_ag_folds.append(p_ag)
        cvf = cv_fit_entry(rep, kind, hp, k)
        repro: Dict[str, Any] = {"fit_v_fold_logloss": (cvf or {}).get("fold_logloss"),
                                 "refit_fold_logloss": ll_fit_orient}
        if cvf is not None and cvf.get("fold_logloss") is not None:
            d_ll = abs(float(cvf["fold_logloss"]) - ll_fit_orient)
            repro["abs_diff" if kind != "mlp" else "abs_diff_1seed_vs_3seed"] = d_ll
        if kind == "lgbm":
            repro.update(fit_v_best_iteration=(cvf or {}).get("best_iteration"), refit_best_iteration=m.best_iteration)
        elif kind == "logistic":
            repro.update(fit_v_nit=(cvf or {}).get("nit"), refit_nit=m.fit_info.get("nit"))
        else:
            repro["note"] = "fit-V CV used seed 0 only; this fold model averages seeds 0, 1, 2 (not comparable)"
        per_fold[str(k)] = {"bundle_sha256": b["bundle_sha256"], "n_train_matches": len(ids),
                            "n_fit_rows": int(len(rows["fit"])), "n_stop_rows": int(len(rows["stop"])),
                            "n_heldout_rows": int(len(te)), "fit_info": _fit_info(m),
                            "calibrator": cal.describe(), "heldout_logloss_raw_natural": VM.log_loss(y_nat[te], p_te),
                            "heldout_logloss_cal_natural": VM.log_loss(y_nat[te], p_oof_cal[te]),
                            "cv_reproduction": repro, "reload_max_abs_diff": reload_diff,
                            "agreement_with_frozen_V_SELECT": agreement(p_ag, p_frozen_ag, y_ag),
                            "seconds": round(time.time() - t0, 2)}
        timings[f"fold_{k}"] = per_fold[str(k)]["seconds"]
        a = per_fold[str(k)]["agreement_with_frozen_V_SELECT"]
        log(f"fold {k}: {len(ids)} training matches, held-out log loss raw {per_fold[str(k)]['heldout_logloss_raw_natural']:.5f}; "
            f"CV repro {repro}; agreement r={a.get('pearson_r')} mean|dp|={a.get('mean_abs_diff')}; "
            f"{per_fold[str(k)]['seconds']} s; {FV.memory_info()}")
        del m
    del ds
    if swapped:
        VM.apply_swap_inplace(Xtr, ytr, smask, perm)       # restore (not used further)

    # ---- verification (O7)
    t0 = time.time()
    cov_v = coverage_check(fold_ids, mids_tr)
    if cov_v["in_own_fold"] or cov_v["full"] != cov_v["n"]:
        raise RuntimeError(f"15.14 V-row match coverage is not 100 %: {cov_v}")
    cov_e = coverage_check(fold_ids, eng_match_ids(args.train, mt))
    if cov_e["n"] and cov_e["full"] != cov_e["n"]:
        log(f"WARNING: {cov_e['n'] - cov_e['full']} of {cov_e['n']} 15.14 engagement-row matches have no V rows and "
            f"are in no fold's training list (ev4_04 needs >= 99 % of labelled matches): {cov_e['missing_examples']}")
    repro_diffs = [v["cv_reproduction"].get("abs_diff") for v in per_fold.values()
                   if v["cv_reproduction"].get("abs_diff") is not None]
    repro_max = max(repro_diffs) if repro_diffs else None
    if kind in ("logistic", "lgbm") and (repro_max is None or repro_max > REPRO_WARN):
        log(f"WARNING: CV reproduction max |fold log loss diff| = {repro_max} (> {REPRO_WARN}); see oof_report.json")
    p_mean = np.mean(p_ag_folds, axis=0)
    ag_all = {"per_fold_pearson_r": [per_fold[str(k)]["agreement_with_frozen_V_SELECT"].get("pearson_r")
                                     for k in range(N_FOLDS)],
              "per_fold_mean_abs_diff": [per_fold[str(k)]["agreement_with_frozen_V_SELECT"].get("mean_abs_diff")
                                         for k in range(N_FOLDS)],
              "mean_of_folds": agreement(p_mean, p_frozen_ag, y_ag),
              "sample": {"rows": int(len(sel_idx)), "of_select_rows": int(sel_m.sum()),
                         "rule": f"V_SELECT rows, all or {args.agree_rows} drawn by default_rng({VM.SEED})"},
              "gate": False}
    ok_oof = np.isfinite(p_oof_raw)
    oof_ll = {"raw_natural": VM.log_loss(y_nat[ok_oof], p_oof_raw[ok_oof]),
              "cal_natural": VM.log_loss(y_nat[ok_oof], p_oof_cal[ok_oof]), "n_rows": int(ok_oof.sum()),
              "fit_v_cv_logloss_chosen": (((rep.get("cv") or {}).get(kind) or {}).get("cv_logloss") or {}).get(
                  str(hp[HP_KEY[kind]])),
              "note": "pooled over the five held-out folds in natural orientation (fit-V's CV value was scored on "
                      "the swapped rows for logistic / lgbm and with one seed for mlp)"}
    timings["verify"] = round(time.time() - t0, 2)
    verification = {"train_ids_exclude_own_fold": True, "coverage_v_rows": cov_v, "coverage_engagement_rows": cov_e,
                    "cv_reproduction_max_abs_diff": repro_max,
                    "cv_reproduction_note": ("logistic / lgbm: held-out fold log loss of the refit vs fit-V's CV fit, "
                                             "same (swapped) rows" if kind != "mlp" else
                                             "mlp: fit-V CV was 1-seed; not comparable (reported per fold)"),
                    "agreement_with_frozen_V_SELECT": ag_all, "pooled_oof_logloss_15_14": oof_ll}

    # ---- outputs
    disclosures = [
        f"The hyperparameter {HP_KEY[kind]} = {hp[HP_KEY[kind]]} of the chosen kind '{kind}' was chosen by ev4_03's "
        "5-fold match-grouped CV over ALL 15.14 folds (lowest pooled out-of-fold log loss over the fixed grid "
        f"{VM.GRIDS[kind][HP_KEY[kind]]}), and the kind itself on V_SELECT (15.15).  Each fold model is therefore "
        "out-of-fold for its fitted weights, but fold k's outcomes took part in choosing this one grid value.",
        "The NaN fill values (15.14 column means) are computed on all 15.14 rows, as in ev4_03 and its CV (label-free)."
        + (" LightGBM bin edges come from a Dataset built on all 15.14 rows, as in ev4_03's CV (label-free)."
           if kind == "lgbm" else ""),
        "Each fold model is calibrated on V_CAL (15.15), like V_frozen; no 15.14 row enters calibration.",
        "fold_k/train_match_ids.txt lists fit AND early-stopping matches of fold k (both informed the model).",
    ] + (["The MLP fold models average seeds 0, 1, 2 like the final fit; ev4_03's CV used seed 0 only, so these are "
          "new fits, not the CV fits."] if kind == "mlp" else [])
    rep_out = {"script": "scripts/exact_v4/ev4_03b_oof_v.py", "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
               "smoke": bool(args.smoke), "kind": kind, "hyperparameter": hp, "budget": budget, "threads": threads,
               "fit_v_threads": fv["threads"], "decisions": DECISIONS, "census": census, "inputs": inputs,
               "input_checks": FV.checks_for_report(checks), "folds": per_fold, "verification": verification,
               "timings_s": timings, "runtime_projection_from_fit_v": project_runtime(fv),
               "memory": FV.memory_info(), "seconds_total": round(time.time() - t_start, 2)}
    EX.write_json_atomic(oof_dir / REPORT_NAME, rep_out)
    extra = {"script": rep_out["script"], "created_utc": rep_out["created_utc"],
             "hyperparameter_selection": {"by": "ev4_03 5-fold match-grouped CV over ALL 15.14 folds (pooled OOF log "
                                                "loss); kind by raw V_SELECT log loss", "reselected_here": False},
             "disclosures": disclosures, "train_rule": train_rule, "budget": budget, "threads": threads,
             "seeds": {"swap_mask": VM.SEED, "mlp": list(VM.MLP_SEEDS_FINAL) if kind == "mlp" else None,
                       "lgbm": VM.SEED if kind == "lgbm" else None},
             "fit_v": {"dir": str(fv["dir"]), "frozen_manifest_sha256": fv["vman_sha256"],
                       "report_e4_sha256": fv["report_sha256"], "chosen": kind},
             "inputs": {r: {"dir": v["dir"], "manifest_sha256": v["manifest_sha256"]} for r, v in inputs.items()},
             "predecisions_sha256": predec["sha256"], "decisions": DECISIONS, "code_sha256": code_hashes(),
             "verification": {"train_ids_exclude_own_fold": True, "coverage_v_rows_rate": cov_v["rate"],
                              "coverage_engagement_rows_rate": cov_e["rate"],
                              "cv_reproduction_max_abs_diff": repro_max,
                              "agreement_pearson_r_per_fold": ag_all["per_fold_pearson_r"],
                              "agreement_is_gate": False},
             "report": {"file": REPORT_NAME, "sha256": VM.sha256_file(oof_dir / REPORT_NAME)}}
    blk = LB.write_oof_manifest(oof_dir, kind, hp, fold_bundles, fold_ids, frozen_sha, smoke=bool(args.smoke),
                                pilot=False, extra=extra)
    om = json.loads((oof_dir / "oof_manifest.json").read_text(encoding="utf-8"))
    sidecar = {"format": LB.OOF_BLOCK_FORMAT, "dir": str(oof_dir), "dir_rel": OOF_DIRNAME,
               "manifest_sha256": blk["manifest_sha256"], "kind": kind, "hyperparameter": hp,
               "frozen_manifest_sha256": fv["vman_sha256"], "frozen_bundle_sha256": frozen_sha,
               "fold_bundle_sha256": {k: f["bundle_sha256"] for k, f in om["folds"].items()},
               "train_match_ids_sha256": {k: f["train_match_ids_sha256"] for k, f in om["folds"].items()},
               "report_sha256": extra["report"]["sha256"], "smoke": bool(args.smoke), "pilot": False,
               "script": rep_out["script"], "created_utc": rep_out["created_utc"],
               "code_sha256": {"scripts/exact_v4/ev4_03b_oof_v.py": VM.sha256_file(HERE)}}
    EX.write_json_atomic(side, sidecar)
    # self-check: the consumer accepts what was written
    spec = LB.check_oof(fv["vman"], fv["vf"], smoke=bool(args.smoke), vpath=fv["vpath"])
    LB.fold_train_ids(spec)
    total = round(time.time() - t_start, 2)
    log(f"done in {total} s: {side} (manifest {blk['manifest_sha256']}); coverage {cov_v['full']}/{cov_v['n']} V-row "
        f"matches, {cov_e['full']}/{cov_e['n']} engagement matches; agreement r {ag_all['per_fold_pearson_r']}")
    return {"oof_dir": str(oof_dir), "sidecar": str(side), "sidecar_sha256": VM.sha256_file(side),
            "manifest_sha256": blk["manifest_sha256"], "kind": kind, "hyperparameter": hp,
            "verification": verification, "folds": per_fold, "seconds_total": total, "timings_s": timings}


if __name__ == "__main__":
    main()
