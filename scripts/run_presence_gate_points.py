"""Presence-gate points (R, B, M) of the engagement definition, re-run end to end (CoG 2026 #118, R1).

The presence gate keeps a kill cluster only when at least M alive champions of each team stand within R of the
fight centre at the cutoff, B before the first kill (gameplay/fights.py::_check_teamfight_validity; CFG fields
TF2_VALIDITY_RADIUS, TF2_ENGAGE_PRE_KILL_MS, TF2_MIN_PER_TEAM).  R and B are anchored on game rules and M = 2 is
set by construction (docs/DEFINITION_EVIDENCE.md); none of them can be estimated from Match-V5, so their leverage
on the prediction result is measured by re-running the pipeline with only these constants changed.

Point spec   name:R_units:B_ms[:M]
    rule_R1600_B15:1600:15000   ref_R1800_B10:1800:10000   rule_R1600_B15_M3:1600:15000:3
M omitted leaves TF2_MIN_PER_TEAM at its CFG default, 2 (core/presets.py v3.3 does not set it).

Two engines.

--engine runner (default; the original behaviour of this script)
    Each point runs ``runner.py --mode train --split_mode patch_holdout`` in a subprocess with the point's
    constants in LOL_CFG_OVERRIDES.  core.config applies LOL_CFG_PRESET and then LOL_CFG_OVERRIDES at import, in
    runner.py and again in every spawned fight-index or data-loader worker, so both must be in the child's
    environment: a cfg attribute set in a parent process never reaches a spawned worker.  Before launching (and
    under --dry-run) the point's environment is probed in a fresh interpreter that imports core.config the way
    runner.py does; the effective constants go into the manifest, and a point whose overrides or preset did not
    land is not launched.  Without --preset the child inherits the caller's LOL_CFG_PRESET; with none set it runs
    the CFG defaults, i.e. the CoG 2026 constants (G 18 s, D 4,000 u, Eq.3 label, match-length time_norm and
    match-wide anchors), which is how runs_presence_gate/ was produced.  Inherited LOL_CFG_OVERRIDES are merged
    under the point's own instead of being dropped.  runner.py's LightGBM baseline is not the paper's learner
    (config BASELINE_LGB_PARAMS, constant and quasi-constant pruning, early stopping after 200 rounds), and
    app/experiment.py caps every split at GLOBAL_SUBSAMPLE_PER_SPLIT = 100,000 rows unless --no-row-cap (an
    uncapped (1,800 u, 10 s) point needs far more RAM than this machine has: build_tabular_Xy holds a row list and
    its copy).  Draws follow LABEL_TIE_STRATEGY (random by default); --tie-policy drop drops them.

--engine holdout (the v3.3 revision runs)
    The corpus-v3.3 protocol of scripts/run_gd_sweep_v33.py on every cached match (or a declared --n-matches
    subsample): preset v3.3 with only R / B / M overridden, label market_event with draws dropped, the clean
    feature path of the preset (TIME_NORM_ABSOLUTE, ANCHORS_CAUSAL, TAB_FRAME_AGE_FEATURE), rows built by
    data.index_split.build_fight_index -> train.baseline.build_tabular_Xy in spawned workers that receive the preset
    and overrides before importing core.config, and the published LightGBM configuration under the patch holdout:
    lgbm_paper of scripts/run_model_comparison_v33.py in the engagement-state-value worktree (Ke et al. 2017,
    "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS; 400 trees, learning rate 0.05, 31
    leaves, subsample 0.9, colsample_bytree 0.9, random_state 7, early stopping after 100 rounds on the
    validation patch), trained on 15.14, early-stopped on 15.15, scored once on 15.16.

    Memory.  The full (1,800 u, 10 s) point has about a million engagements, ~30 GB of float32 features.  Matches
    are processed patch by patch, each patch in its own worker pool that is closed before LightGBM touches the
    rows: the training patch's rows are binned into a LightGBM Dataset and their raw features freed, then the
    validation patch's rows (binned with the training bin mappers), then the booster is trained, and the test
    patch is detected only afterwards and scored chunk by chunk without being stored.  The first task of every
    phase reports its workers' effective cfg, and a phase whose workers did not run preset v3.3 with the point's
    R / B / M and draws dropped stops the point.
    Datasets are built from the list of per-chunk float32 blocks (LGBM_DatasetCreateFromMats), so nothing is
    concatenated.  Booster parameters are exactly those LGBMClassifier.fit derives (LightGBM 4.6
    sklearn._process_params plus its eval-metric merge: objective binary, metric [auc, binary]);
    --check-equivalence refits LGBMClassifier on the concatenated arrays of a small run and compares test
    predictions.  Within a patch, matches follow the corpus shard merge order (index in the sorted cache listing
    mod 32, then index), so the full (1,600 u, 15 s, M 2) point feeds the booster the rows of corpus_shards_v33 in
    the same order and is compared with lgbm_paper (features/model_comparison_v33_patch.json and .preds.npz).

    Per point: engagements, matches and labelled rows per patch, positive rates, the validation AUC at the best
    iteration, test AUC with a match-clustered percentile bootstrap interval (Efron & Tibshirani 1993, ch. 13;
    whole matches resampled, Field & Welsh 2007, JRSS-B 69(3)), AUC per participation class (pick / skirmish /
    teamfight on the smaller side, teamfight >= --teamfight-min) and per presence class (small / large at cuts 3
    and 4), and the pick - teamfight gap, all on the same replicates.  ``--summarize`` compares every finished
    point with ``--reference`` (default: the (1,600 u, 15 s, M 2) point) on all engagements and on the labelled
    test rows, matched one-to-one on (match id, first-kill time) within --tolerance-ms, with paired AUC differences
    whose replicates resample the union of both points' test matches (run_gd_sweep_v33.paired_vs_reference).

    Leak discipline.  No test-patch match is detected before the booster is final, early stopping reads only the
    validation patch, and nothing is calibrated.  Features are the preset's causal path.

    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/run_presence_gate_points.py --engine holdout \\
        --points rule_R1600_B15:1600:15000 --workers 8 --n-jobs 8
    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/run_presence_gate_points.py --engine holdout \\
        --summarize --points rule_R1600_B15:1600:15000 ref_R1800_B10:1800:10000 rule_R1600_B15_M3:1600:15000:3

    # original runner engine, under the v3.3 preset
    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_presence_gate_points.py --preset v3.3 --tie-policy drop \\
        --group runs_presence_gate_v33 --points ref_R1800_B10:1800:10000 rule_R1600_B15:1600:15000 --seed 7

Runner-engine outputs: <OUTPUT_ROOT>/<group>/<point>/run_*/models/lgbm/baseline/{report.json, pred_*.csv, ...}, a
log per point, and manifest.json (points, return codes, preset, overrides, effective constants); summarise with
scripts/summarize_presence_gate_points.py.  Holdout-engine outputs: <out-dir>/points/<point>/{result.json,
rows.npz, model.txt}, <out-dir>/presence_gate_summary.{json,csv}, <out-dir>/presence_gate.log.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):  # spawned workers inherit sys.path and import run_gd_sweep_v33
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_POINTS = ["ref_R1800_B10:1800:10000", "rule_R1600_B15:1600:15000"]
HOLDOUT_PRESET = "v3.3"
REFERENCE_CONSTANTS = (1600.0, 15000, 2)
NUM_CORPUS_SHARDS = 32
FEATURES = Path("D:/LOL_Project/fusion_2615/features")
DEFAULT_HOLDOUT_OUT = FEATURES / "tog_revision" / "A6-definition-sensitivity" / "presence_gate"
LGBM_PAPER_JSON = FEATURES / "model_comparison_v33_patch.json"
LGBM_PAPER_PREDS = FEATURES / "model_comparison_v33_patch.preds.npz"
LABEL_KEY = "market_event (preset LABEL_TYPE, engagement attribution, 300 g dead zone), tie policy drop"
GATE_KEYS = ("TF2_VALIDITY_RADIUS", "TF2_ENGAGE_PRE_KILL_MS", "TF2_MIN_PER_TEAM")
PROBE_KEYS = (
    "TF2_KILL_CLUSTER_GAP_MS", "CLUSTER_MAX_DIAMETER", "TF2_VALIDITY_RADIUS", "TF2_ENGAGE_PRE_KILL_MS",
    "TF2_MIN_PER_TEAM", "TF2_INTERACTION_RADIUS", "FIGHT_HORIZON_SEC", "LABEL_TYPE", "LABEL_EVENT_ATTRIBUTION",
    "LABEL_GOLD_DEADZONE", "LABEL_TIE_STRATEGY", "TIME_NORM_ABSOLUTE", "TIME_NORM_DENOM_MIN", "ANCHORS_CAUSAL",
    "TAB_FRAME_AGE_FEATURE", "GLOBAL_SUBSAMPLE_PER_SPLIT", "FIGHT_INDEX_CACHE_ENABLED", "FIGHT_INDEX_NUM_WORKERS",
    "BASELINE_LGB_PARAMS", "FEATURE_VERSION", "RUN_DIRNAME", "CACHE_DIRNAME",
)
_PROBE_SRC = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from core.config import cfg, CACHE_DIR, RUN_DIR
def js(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [js(x) for x in v]
    return str(v)
out = {k: js(getattr(cfg, k, None)) for k in json.loads(sys.argv[1])}
out["_env_LOL_CFG_PRESET"] = os.environ.get("LOL_CFG_PRESET")
out["_CACHE_DIR"] = str(CACHE_DIR)
out["_RUN_DIR"] = str(RUN_DIR)
print("CFG_PROBE " + json.dumps(out))
"""

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_LOG_PATH = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ----------------------------------------------------------------------------- shared helpers
def parse_point(s: str):
    parts = str(s).split(":")
    if len(parts) not in (3, 4) or not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]):
        raise SystemExit(f"point {s!r}: expected name:R_units:B_ms[:M] with a plain name")
    name, r_u, b_ms = parts[0], float(parts[1]), int(parts[2])
    m = int(parts[3]) if len(parts) == 4 else None
    return name, r_u, b_ms, m


def point_overrides(r_u: float, b_ms: int, m: int | None) -> dict:
    out = {"TF2_VALIDITY_RADIUS": float(r_u), "TF2_ENGAGE_PRE_KILL_MS": int(b_ms)}
    if m is not None:
        out["TF2_MIN_PER_TEAM"] = int(m)
    return out


def parse_set(pairs) -> dict:
    extra = {}
    for kv in pairs or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        extra[k.strip()] = v
    return extra


def load_boot_module():
    import scale_cut_sensitivity_v33 as boot_mod  # no core.config import at module level
    return boot_mod


def same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return a == b


def probe_cfg(env: dict, keys=PROBE_KEYS, timeout: int = 600) -> dict:
    """Effective CFG values in a fresh interpreter started with ``env`` (how runner.py and every spawned worker
    initialise core.config)."""
    proc = subprocess.run([sys.executable, "-c", _PROBE_SRC, json.dumps(list(keys))], cwd=str(PROJECT_ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("CFG_PROBE "):
            return json.loads(line[len("CFG_PROBE "):])
    raise RuntimeError(f"cfg probe failed (rc={proc.returncode}): {proc.stderr[-2000:]}")


def cfg_problems(effective: dict, overrides: dict, preset: str | None, boot_mod) -> list:
    problems = []
    for k, v in overrides.items():
        if k in effective and not same_value(effective[k], v):
            problems.append(f"override {k}={v!r} did not land (effective {effective[k]!r})")
    if preset:
        for k, v in boot_mod.preset_values(preset).items():
            if k not in overrides and k in effective and not same_value(effective[k], v):
                problems.append(f"preset {preset} {k}={v!r} did not land (effective {effective[k]!r})")
        if effective.get("_env_LOL_CFG_PRESET") != preset:
            problems.append(f"child LOL_CFG_PRESET={effective.get('_env_LOL_CFG_PRESET')!r}, expected {preset!r}")
    return problems


def _process_memory_counters(pid: int | None = None):
    """(peak working set, working set, peak commit) in bytes for this process or ``pid`` (Windows only)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    handle = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(0x1000 | 0x0010, False, int(pid))
    if not handle:
        return None
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not fn(handle, ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.PeakWorkingSetSize, pmc.WorkingSetSize, pmc.PeakPagefileUsage
    finally:
        if pid is not None:
            k32.CloseHandle(handle)


def peak_memory_gb() -> dict:
    """Peak working set and peak commit of this process (Windows), or ru_maxrss elsewhere."""
    try:
        if os.name == "nt":
            c = _process_memory_counters()
            return {"peak_working_set_gb": round(c[0] / 2 ** 30, 2), "peak_commit_gb": round(c[2] / 2 ** 30, 2)} if c else {}
        import resource
        return {"peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}
    except Exception:
        return {}


def pool_memory_gb(ex) -> dict:
    """Peak and current working sets of a ProcessPoolExecutor's live workers (Windows; empty elsewhere)."""
    try:
        if os.name != "nt":
            return {}
        counters = [c for c in (_process_memory_counters(pid) for pid in list((ex._processes or {}).keys())) if c]
        if not counters:
            return {}
        return {"workers": len(counters), "max_peak_working_set_gb": round(max(c[0] for c in counters) / 2 ** 30, 2),
                "sum_peak_working_set_gb": round(sum(c[0] for c in counters) / 2 ** 30, 2),
                "sum_working_set_gb": round(sum(c[1] for c in counters) / 2 ** 30, 2)}
    except Exception:
        return {}


# ----------------------------------------------------------------------------- runner engine
def run_runner_engine(args, points: list, extra: dict) -> int:
    boot_mod = load_boot_module()
    out_root = Path(os.environ.get("LOL_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))
    group = out_root / args.group
    group.mkdir(parents=True, exist_ok=True)
    manifest_path = group / "manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else {"points": {}}
    inherited = json.loads(os.environ.get("LOL_CFG_OVERRIDES", "").strip() or "{}")
    git = boot_mod.git_provenance()

    for name, R, B, M in points:
        overrides = {**inherited, "RUN_DIRNAME": f"{args.group}/{name}", **point_overrides(R, B, M),
                     "FIGHT_INDEX_NUM_WORKERS": int(args.workers)}
        if args.tie_policy:
            overrides["LABEL_TIE_STRATEGY"] = args.tie_policy
        if args.no_row_cap:
            overrides["GLOBAL_SUBSAMPLE_PER_SPLIT"] = 0
        overrides.update(extra)
        env = dict(os.environ)
        if args.preset:
            env["LOL_CFG_PRESET"] = args.preset
        preset = env.get("LOL_CFG_PRESET", "").strip() or None
        env["LOL_CFG_OVERRIDES"] = json.dumps(overrides)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        cmd = [sys.executable, "runner.py", "--mode", "train", "--max_matches", str(args.max_matches),
               "--seed", str(args.seed), "--split_mode", "patch_holdout",
               "--train_patches", args.train_patches, "--val_patches", args.val_patches,
               "--test_patches", args.test_patches, "--models", args.models]
        log_path = group / f"{name}.log"
        print(f"[{name}] R={R:.0f} u, B={B} ms, M={'default' if M is None else M}, preset={preset or 'none (CFG defaults)'} "
              f"-> {group / name}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        effective = probe_cfg(env)
        problems = cfg_problems(effective, overrides, preset, boot_mod)
        print("  effective: " + json.dumps({k: effective.get(k) for k in PROBE_KEYS if k not in ("BASELINE_LGB_PARAMS",)}),
              flush=True)
        if problems:
            print(f"[{name}] NOT LAUNCHED: " + "; ".join(problems), flush=True)
            if not args.dry_run:
                manifest["points"][name] = {"R_u": R, "B_ms": B, "M": M, "returncode": None, "not_launched": problems,
                                            "preset": preset, "overrides": overrides, "effective_cfg": effective}
                json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1)
            continue
        if args.dry_run:
            continue
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} preset={preset} overrides={json.dumps(overrides)}\n")
            lf.write("# " + " ".join(cmd) + "\n")
            lf.flush()
            rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - t0
        manifest["points"][name] = {"R_u": R, "B_ms": B, "seed": args.seed, "max_matches": args.max_matches,
                                    "models": args.models, "returncode": rc, "elapsed_s": round(elapsed, 1),
                                    "run_dir": str(group / name), "log": str(log_path),
                                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "M": M, "preset": preset, "overrides": overrides, "effective_cfg": effective,
                                    "python": sys.executable, **git}
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1)
        print(f"[{name}] rc={rc} elapsed={elapsed / 60:.1f} min", flush=True)
    return 0


# ----------------------------------------------------------------------------- holdout engine: inputs
def paper_booster_params(seed: int, n_jobs: int, sweep) -> dict:
    """The parameters LGBMClassifier(**lgbm_paper).fit(..., eval_metric='auc') hands to lightgbm.train."""
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(**sweep.PAPER_LGBM, random_state=int(seed), n_jobs=int(n_jobs), verbose=-1)
    clf._n_classes = 2
    params = clf._process_params(stage="fit")
    metric = params.get("metric")
    metric = [metric] if isinstance(metric, (str, type(None))) else list(metric)
    params["metric"] = [m for m in ["auc"] if m not in metric] + metric
    params["metric"] = [m for m in params["metric"] if m is not None]
    return params


def read_match_patches(mids: list, cache_dir: Path, cache_file: Path, threads: int) -> dict:
    """match id -> normalised patch, as data.index_split reads it (meta 'patch', else 'patch_full')."""
    from gameplay.fights import normalize_patch
    digest = hashlib.blake2b("\n".join(mids).encode("utf-8"), digest_size=12).hexdigest()
    if cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            if blob.get("cache_dir") == str(cache_dir) and blob.get("digest") == digest:
                return blob["patch"]
        except Exception:
            pass
    from concurrent.futures import ThreadPoolExecutor

    def one(mid):
        meta = json.loads((cache_dir / f"{mid}.meta.json").read_bytes())
        return str(normalize_patch(str(meta.get("patch", meta.get("patch_full", "0.0")))))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        patches = list(ex.map(one, mids, chunksize=256))
    out = dict(zip(mids, patches))
    tmp = cache_file.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"cache_dir": str(cache_dir), "digest": digest, "n": len(mids), "patch": out}), encoding="utf-8")
    os.replace(tmp, cache_file)
    log(f"read {len(mids):,} match patches in {time.time() - t0:.0f}s -> {cache_file}")
    return out


def phase_tasks(mids: list, pos: dict, chunk_size: int, first_idx: int) -> list:
    ordered = sorted(mids, key=lambda m: (pos[m] % NUM_CORPUS_SHARDS, pos[m]))  # corpus shard merge order
    return [(first_idx + i, ordered[s:s + chunk_size]) for i, s in enumerate(range(0, len(ordered), chunk_size))]


def run_phase(ex, tasks: list, phase: str, allowed: set, sweep, booster=None, keep_x: bool = False,
              predict_threads: int = 0) -> dict:
    """Consume one patch phase from the worker pool.  Without a booster the per-chunk X blocks are kept (to be
    binned); with one, each block is scored and dropped."""
    xs, ys, preds, ref_parts, used_parts = [], [], [], [], []
    names, names_sha1, effective = None, None, None
    offset, done, n_total = 0, 0, sum(len(t[1]) for t in tasks)
    t0, next_report = time.time(), 0.1
    for res in ex.map(sweep._work_chunk, tasks):
        if res["cfg"] is not None:
            effective = res["cfg"]
        patches = res["refs"]["patch"]
        if len(patches) and not set(np.unique(patches).tolist()) <= allowed:
            raise RuntimeError(f"{phase}: engagements from patches {sorted(set(patches.tolist()) - allowed)}")
        if res["names"]:
            sha = hashlib.sha1("\n".join(res["names"]).encode("utf-8")).hexdigest()
            if names_sha1 is None:
                names, names_sha1 = res["names"], sha
            elif sha != names_sha1:
                raise RuntimeError("feature names differ between chunks")
        ref_parts.append(res["refs"])
        used_parts.append(res["used_idx"] + offset)
        offset += len(patches)
        ys.append(res["y"])
        if res["X"] is not None and len(res["y"]):
            if booster is not None:
                preds.append(booster.predict(res["X"], **({"num_threads": int(predict_threads)} if predict_threads else {})))
                if keep_x:
                    xs.append(res["X"])
            else:
                xs.append(np.ascontiguousarray(res["X"], dtype=np.float32))
        done += res["n_matches"]
        if done >= next_report * n_total or done == n_total:
            log(f"    {phase}: {done:,}/{n_total:,} matches, {sum(len(v) for v in ys):,} rows ({time.time() - t0:.0f}s)")
            next_report += 0.1
    refs = {k: (np.concatenate([p[k] for p in ref_parts]) if ref_parts else np.zeros(0)) for k in sweep.REF_FIELDS}
    return {"xs": xs, "y": (np.concatenate(ys).astype(np.int8) if ys else np.zeros(0, np.int8)),
            "pred": (np.concatenate(preds) if preds else None), "refs": refs,
            "used_idx": (np.concatenate(used_parts).astype(np.int64) if used_parts else np.zeros(0, np.int64)),
            "names": names, "names_sha1": names_sha1, "effective": effective, "seconds": round(time.time() - t0, 1),
            "n_matches": n_total}


def phase_counts(ph: dict) -> dict:
    refs, y = ph["refs"], ph["y"]
    n_refs = int(len(refs["match_id"]))
    return {"n_matches": ph["n_matches"], "n_matches_with_engagement": int(len(np.unique(refs["match_id"]))),
            "n_engagements": n_refs, "n_rows_labelled": int(len(y)), "engagements_without_label": n_refs - int(len(y)),
            "positive_rate": (float(y.mean()) if len(y) else None),
            "matches_with_labelled_row": int(len(np.unique(refs["match_id"][ph["used_idx"]]))) if len(y) else 0,
            "build_seconds": ph["seconds"]}


def compare_with_lgbm_paper(y_test, pred_test, groups_test, val_auc, best_iteration, rows: dict) -> dict:
    out = {"reference_json": str(LGBM_PAPER_JSON), "reference_preds": str(LGBM_PAPER_PREDS)}
    if not LGBM_PAPER_JSON.exists():
        return {**out, "available": False}
    from sklearn.metrics import roc_auc_score
    ref = json.loads(LGBM_PAPER_JSON.read_text(encoding="utf-8"))
    m = (ref.get("models") or {}).get("lgbm_paper") or {}
    ours = float(roc_auc_score(y_test, pred_test))
    out.update({"available": True, "test_auc_reference": m.get("overall_auc"), "test_auc_this_run": ours,
                "test_auc_diff": (ours - m["overall_auc"]) if m.get("overall_auc") is not None else None,
                "val_auc_reference": m.get("val_auc"), "val_auc_this_run": val_auc,
                "best_iteration_reference": m.get("best_iteration"), "best_iteration_this_run": best_iteration,
                "rows_reference": (ref.get("split") or {}).get("rows"), "rows_this_run": rows,
                "n_features_reference": m.get("n_features")})
    if LGBM_PAPER_PREDS.exists():
        with np.load(LGBM_PAPER_PREDS, allow_pickle=False) as z:
            ry, rg, rp = z["y"].astype(np.int64), z["groups"].astype(str), z["lgbm_paper"].astype(np.float64)
        aligned = len(ry) == len(y_test) and np.array_equal(rg, np.asarray(groups_test).astype(str)) and \
            np.array_equal(ry, np.asarray(y_test).astype(np.int64))
        out["test_rows_aligned_with_reference"] = bool(aligned)
        if aligned:
            d = np.abs(rp - pred_test)
            out.update({"max_abs_pred_diff": float(d.max()), "mean_abs_pred_diff": float(d.mean()),
                        "pearson_r": float(np.corrcoef(rp, pred_test)[0, 1])})
    return out


# ----------------------------------------------------------------------------- holdout engine: one point
def run_holdout_point(name, R, B, M, args, ctx: dict) -> None:
    import lightgbm as lgb
    sweep, boot_mod = ctx["sweep"], ctx["boot_mod"]
    pdir = args.out_dir / "points" / name
    js_path, npz_path = pdir / "result.json", pdir / "rows.npz"
    overrides = {**point_overrides(R, B, M), **ctx["extra"], **sweep.operational_overrides()}
    params = paper_booster_params(args.seed, args.n_jobs, sweep)
    signature = {"engine": "holdout", "preset": HOLDOUT_PRESET, "overrides": overrides, "n_matches": len(ctx["mids"]),
                 "seed": int(args.seed), "sample_digest": ctx["digest"], "cache_dir": ctx["cache_dir"],
                 "patches": {"train": ctx["train"], "val": ctx["val"], "test": ctx["test"]},
                 "lgbm": sweep.PAPER_LGBM, "early_stopping_rounds": sweep.EARLY_STOPPING_ROUNDS,
                 "teamfight_min": int(args.teamfight_min), "negative_count_as": args.negative_count_as,
                 "n_boot": int(args.n_boot), "order": "corpus shard merge order within patch"}
    if js_path.exists() and npz_path.exists() and not args.force:
        prev = json.loads(js_path.read_text(encoding="utf-8"))
        if prev.get("complete") and prev.get("signature") == signature:
            log(f"[{name}] finished earlier; skipped")
            return
        if prev.get("complete"):
            raise SystemExit(f"[{name}] exists with a different signature; pass --force or another --out-dir")
    pdir.mkdir(parents=True, exist_ok=True)
    log(f"[{name}] R={R:g} u B={B} ms M={'default' if M is None else M}: {len(ctx['mids']):,} matches "
        f"(train {len(ctx['phase_mids']['train']):,} / val {len(ctx['phase_mids']['val']):,} / test {len(ctx['phase_mids']['test']):,})")
    t_start = time.time()
    check = bool(args.check_equivalence)
    # each phase numbers its tasks from 0, so the first task of every phase reports its workers' effective cfg
    tasks_tr = phase_tasks(ctx["phase_mids"]["train"], ctx["pos"], args.chunk_size, 0)
    tasks_va = phase_tasks(ctx["phase_mids"]["val"], ctx["pos"], args.chunk_size, 0)
    tasks_te = phase_tasks(ctx["phase_mids"]["test"], ctx["pos"], args.chunk_size, 0)
    expected = {**boot_mod.preset_values(HOLDOUT_PRESET), **overrides}
    required = set(boot_mod.preset_values(HOLDOUT_PRESET)) | set(point_overrides(R, B, M)) | {"TF2_MIN_PER_TEAM"}
    if M is None:
        expected.setdefault("TF2_MIN_PER_TEAM", 2)  # CFG default; v3.3 does not set it
    keep, worker_memory = {}, {}

    def checked_phase(tasks, phase, allowed, booster=None, keep_x=False):
        """One patch phase in its own worker pool: the pool is closed before LightGBM bins or trains, so idle
        row-building workers never hold memory next to the largest raw-feature block."""
        with sweep.make_pool(overrides, args.workers, HOLDOUT_PRESET) as ex:
            ph = run_phase(ex, tasks, phase, allowed, sweep, booster=booster, keep_x=keep_x, predict_threads=args.n_jobs)
            worker_memory[phase] = pool_memory_gb(ex)
        e = ph["effective"] or {}
        bad = {k: (e.get(k), v) for k, v in expected.items() if (k in e or k in required) and not same_value(e.get(k), v)}
        if e.get("LOL_CFG_PRESET") != HOLDOUT_PRESET or e.get("LABEL_TIE_POLICY") != "drop" or bad:
            raise SystemExit(f"[{name}] {phase} workers did not run the requested definition: "
                             f"preset={e.get('LOL_CFG_PRESET')} tie={e.get('LABEL_TIE_POLICY')} {bad}")
        log(f"[{name}] {phase} workers {worker_memory[phase]}")
        return ph

    tr = checked_phase(tasks_tr, "train", set(ctx["train"]))
    eff = tr["effective"]
    if check:
        keep["Xtr"], keep["ytr"] = np.concatenate(tr["xs"]), tr["y"].copy()
    t0 = time.time()
    dtrain = lgb.Dataset(tr["xs"], label=tr["y"], params=params, free_raw_data=True)
    dtrain.construct()
    tr["xs"] = None
    log(f"[{name}] train rows {len(tr['y']):,} binned ({time.time() - t0:.0f}s) {peak_memory_gb()}")

    va = checked_phase(tasks_va, "val", set(ctx["val"]))
    if va["names_sha1"] != tr["names_sha1"]:
        raise SystemExit(f"[{name}] feature names differ between the train and validation patches")
    if check:
        keep["Xva"], keep["yva"] = np.concatenate(va["xs"]), va["y"].copy()
    t0 = time.time()
    dval = lgb.Dataset(va["xs"], label=va["y"], reference=dtrain, params=params, free_raw_data=True)
    dval.construct()
    va["xs"] = None
    log(f"[{name}] val rows {len(va['y']):,} binned ({time.time() - t0:.0f}s) {peak_memory_gb()}")

    t0 = time.time()
    evals = {}
    booster = lgb.train(params, dtrain, num_boost_round=int(sweep.PAPER_LGBM["n_estimators"]), valid_sets=[dval],
                        callbacks=[lgb.early_stopping(sweep.EARLY_STOPPING_ROUNDS, verbose=False),
                                   lgb.log_evaluation(0), lgb.record_evaluation(evals)])
    fit_s = time.time() - t0
    best_iteration = int(booster.best_iteration) if booster.best_iteration else None
    best_score = {k: float(v) for k, v in booster.best_score.get("valid_0", {}).items()}
    importance = booster.feature_importance("split")
    model_str = booster.model_to_string()
    (pdir / "model.txt").write_text(model_str, encoding="utf-8")
    del booster, dtrain, dval
    booster = lgb.Booster(model_str=model_str)  # frees the training Datasets; predictions are unchanged
    log(f"[{name}] booster: best_iteration={best_iteration} val AUC={best_score.get('auc')} ({fit_s:.0f}s) {peak_memory_gb()}")

    # the test patch is detected only now, after the booster is final
    te = checked_phase(tasks_te, "test", set(ctx["test"]), booster=booster, keep_x=check)
    if te["names_sha1"] != tr["names_sha1"]:
        raise SystemExit(f"[{name}] feature names differ between the train and test patches")

    equivalence = None
    if check:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        clf = LGBMClassifier(**sweep.PAPER_LGBM, random_state=int(args.seed), n_jobs=int(args.n_jobs), verbose=-1)
        clf.fit(keep["Xtr"], keep["ytr"], eval_set=[(keep["Xva"], keep["yva"])], eval_metric="auc",
                callbacks=[early_stopping(sweep.EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])
        p_wrapper = clf.predict_proba(np.concatenate(te["xs"]))[:, 1] if te["xs"] else np.zeros(0)
        diff = float(np.max(np.abs(p_wrapper - te["pred"]))) if len(p_wrapper) else 0.0
        equivalence = {"what": "LGBMClassifier(**lgbm_paper) on the concatenated arrays vs lightgbm.train on chunk blocks",
                       "best_iteration_wrapper": int(clf.best_iteration_ or 0), "best_iteration_native": best_iteration,
                       "max_abs_test_pred_diff": diff, "passed": bool(diff <= 1e-12 and (clf.best_iteration_ or 0) == (best_iteration or 0))}
        keep.clear()
        te["xs"] = None
        log(f"[{name}] equivalence check: {equivalence}")

    # ---- test statistics
    refs_te, y_te, p_te = te["refs"], te["y"].astype(np.int64), te["pred"]
    idx = te["used_idx"]
    groups_te = refs_te["match_id"][idx]
    nmin = sweep.smaller_side(refs_te["cluster_blue"][idx], refs_te["cluster_red"][idx], args.negative_count_as)
    pres = sweep.smaller_side(refs_te["present_blue"][idx], refs_te["present_red"][idx], "unknown")
    extra_masks = {f"presence_small_c{c}": (pres >= 0) & (pres < c) for c in (3, 4)}
    extra_masks.update({f"presence_large_c{c}": pres >= c for c in (3, 4)})
    test = sweep.test_auc_block(y_te, p_te, groups_te, nmin, args.teamfight_min, args.n_boot, args.seed, boot_mod,
                                extra_masks=extra_masks)

    rows = {"train": int(len(tr["y"])), "val": int(len(va["y"])), "test": int(len(y_te))}
    counts = {"train": phase_counts(tr), "val": phase_counts(va), "test": phase_counts(te)}
    all_refs = {k: np.concatenate([tr["refs"][k], va["refs"][k], te["refs"][k]]) for k in sweep.REF_FIELDS}
    n_eng = int(len(all_refs["match_id"]))
    nmin_all = sweep.smaller_side(all_refs["cluster_blue"], all_refs["cluster_red"], args.negative_count_as)
    counts["all"] = {"n_matches_sampled": len(ctx["mids"]), "n_engagements": n_eng,
                     "engagements_per_sampled_match": n_eng / max(1, len(ctx["mids"])),
                     "n_rows_labelled": int(sum(rows.values())),
                     "positive_rate": float(np.concatenate([tr["y"], va["y"], te["y"]]).mean()) if sum(rows.values()) else None,
                     "engagement_scale_share": {k: float(m.mean()) for k, m in sweep.scale_masks(nmin_all, args.teamfight_min).items()}}

    reproduction = None
    is_reference = (float(R), int(B), int(eff.get("TF2_MIN_PER_TEAM", -1))) == REFERENCE_CONSTANTS
    if is_reference and args.n_matches == 0 and (ctx["train"], ctx["val"], ctx["test"]) == (["15.14"], ["15.15"], ["15.16"]):
        reproduction = compare_with_lgbm_paper(y_te, p_te, groups_te, best_score.get("auc"), best_iteration, rows)
        log(f"[{name}] vs lgbm_paper: {json.dumps(reproduction)}")

    offsets = np.cumsum([0, len(tr["refs"]["match_id"]), len(va["refs"]["match_id"])])
    row_ref_idx = np.concatenate([tr["used_idx"] + offsets[0], va["used_idx"] + offsets[1], te["used_idx"] + offsets[2]])
    row_split = np.concatenate([np.zeros(rows["train"], np.int8), np.ones(rows["val"], np.int8), np.full(rows["test"], 2, np.int8)])
    pred_all = np.concatenate([np.full(rows["train"] + rows["val"], np.nan), p_te])
    np.savez_compressed(npz_path, **{f"ref_{k}": v for k, v in all_refs.items()}, row_ref_idx=row_ref_idx, row_split=row_split,
                        y=np.concatenate([tr["y"], va["y"], te["y"]]).astype(np.int8), pred=pred_all)

    wall = round(time.time() - t_start, 1)
    result = {
        "item": boot_mod.ITEM, "complete": True, "signature": signature,
        "point": {"name": name, "R_u": float(R), "B_ms": int(B), "M": (None if M is None else int(M)),
                  "M_effective": eff.get("TF2_MIN_PER_TEAM"), "is_reference_constants": bool(is_reference)},
        "provenance": {**ctx["git"], "script": "scripts/run_presence_gate_points.py --engine holdout", "python": sys.executable,
                       "preset": HOLDOUT_PRESET, "label_key": LABEL_KEY,
                       "split": {"kind": "patch holdout", "train": ctx["train"], "val": ctx["val"], "test": ctx["test"]},
                       "seed": int(args.seed), "n_matches": len(ctx["mids"]), "n_matches_in_cache": ctx["n_cache"],
                       "full_corpus": args.n_matches == 0, "sample_digest": ctx["digest"],
                       "lol_output_root": os.environ.get("LOL_OUTPUT_ROOT"), "worker_overrides": overrides,
                       "effective_cfg": eff, "feature_names_sha1": tr["names_sha1"], "workers": int(args.workers),
                       "wall_clock_s": wall, "parent_memory": peak_memory_gb(), "worker_memory": worker_memory,
                       "memory_note": "parent peak (the whole point) plus, per phase, the row-building workers' peaks "
                                      "read before that phase's pool closed; pools never overlap LightGBM binning or "
                                      "training"},
        "counts": counts,
        "model": {"learner": "lightgbm.train with the LGBMClassifier(**lgbm_paper) fit parameters",
                  "lgbm_paper": sweep.PAPER_LGBM, "booster_params": params,
                  "early_stopping_rounds": sweep.EARLY_STOPPING_ROUNDS, "best_iteration": best_iteration,
                  "val_best_score": best_score, "n_features": len(tr["names"] or []),
                  "n_features_split_on": int((importance > 0).sum()), "fit_seconds": round(fit_s, 1),
                  "model_file": str(pdir / "model.txt")},
        "test": test, "reproduces_lgbm_paper": reproduction, "equivalence_check": equivalence,
        "deviations": [
            ("Declared subsample of %d cache matches (seed %d)." % (len(ctx["mids"]), args.seed)) if args.n_matches else
            "Full corpus: every cached match.",
            "LightGBM is trained with lightgbm.train on per-chunk float32 blocks instead of LGBMClassifier on one matrix, "
            "so the raw features never have to be held at once; the parameters are the ones the wrapper derives "
            "(checked bit for bit with --check-equivalence on a subsample).",
            "All 7,106 columns are passed; columns constant on the training patch are never split on, which is what the "
            "corpus-wide constant filter (6,164 columns) of lgbm_paper does to the same rows.",
            "The validation AUC is the early-stopping best_score at the best iteration (no second validation pass).",
            "Draws are dropped through the label tie policy while rows are built, not by masking y_market_event.",
            f"Negative participation counts are read as {args.negative_count_as} (int(0 or -1) in "
            "data/index_split._fight_to_ref_row); presence counts are >= M and never negative.",
            "subsample=0.9 is inert (subsample_freq 0, bagging off), as in lgbm_paper and the OOF headline.",
            f"num_threads={args.n_jobs} instead of all cores; the thread count does not change the fitted trees.",
            "Presence classes are small / large because the gate forces n_min >= M.",
        ],
    }
    js_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[{name}] engagements {n_eng:,}; rows train/val/test {rows['train']:,}/{rows['val']:,}/{rows['test']:,}; "
        f"test AUC {boot_mod.fmt_ci(test['auc'])}; pick AUC {boot_mod.fmt_ci(test['classes']['pick']['auc'])}; "
        f"teamfight AUC {boot_mod.fmt_ci(test['classes']['teamfight']['auc'])} ({wall:.0f}s) {peak_memory_gb()}")


# ----------------------------------------------------------------------------- holdout engine: summary
def summarize_holdout(args, points: list, sweep, boot_mod, git: dict) -> int:
    t0 = time.time()
    loaded = {}
    for name, R, B, M in points:
        jp, zp = args.out_dir / "points" / name / "result.json", args.out_dir / "points" / name / "rows.npz"
        if jp.exists() and zp.exists():
            js = json.loads(jp.read_text(encoding="utf-8"))
            if js.get("complete"):
                with np.load(zp, allow_pickle=False) as z:
                    loaded[name] = (js, {k: z[k] for k in z.files})
    missing = [p[0] for p in points if p[0] not in loaded]
    if not loaded:
        log("no finished points")
        return 1
    ref_name = args.reference or next((n for n, (js, _) in loaded.items() if js["point"]["is_reference_constants"]), next(iter(loaded)))
    if ref_name not in loaded:
        log(f"reference point {ref_name} has not finished")
        return 1
    comparable = lambda js: (js["signature"]["n_matches"], js["signature"]["sample_digest"], json.dumps(js["signature"]["patches"]),
                             js["signature"]["seed"])
    if len({comparable(js) for js, _ in loaded.values()}) != 1:
        raise SystemExit("finished points were run on different samples, patches or seeds")
    tol = int(args.tolerance_ms)

    def unpack(js, z):
        refs = {k[4:]: z[k] for k in z if k.startswith("ref_")}
        t = sweep.anchor_time(refs, int(js["point"]["B_ms"]))
        te = z["row_split"] == 2
        ridx = z["row_ref_idx"][te]
        return {"match_id": refs["match_id"], "t": t, "test_mid": refs["match_id"][ridx], "test_t": t[ridx],
                "y": z["y"][te].astype(np.int64), "p": z["pred"][te]}

    ref = unpack(*loaded[ref_name])
    rows = []
    for name, R, B, M in points:
        if name not in loaded:
            continue
        js, z = loaded[name]
        cmp = sweep.paired_vs_reference(ref, unpack(js, z), tol, args.n_boot, args.seed, boot_mod)
        rows.append({"name": name, "R_u": js["point"]["R_u"], "B_ms": js["point"]["B_ms"], "M_effective": js["point"]["M_effective"],
                     "is_reference": name == ref_name, "counts": js["counts"],
                     "model": {k: js["model"][k] for k in ("best_iteration", "val_best_score", "n_features_split_on")},
                     "test": js["test"], "overlap_engagements": cmp["overlap"], "test_rows_vs_reference": cmp["shared"],
                     "auc_vs_reference": cmp["auc"], "paired_vs_reference": cmp["paired"],
                     "reproduces_lgbm_paper": js.get("reproduces_lgbm_paper"), "wall_clock_s": js["provenance"]["wall_clock_s"]})
    sig = loaded[ref_name][0]["signature"]
    summary = {
        "item": boot_mod.ITEM, "what": "presence-gate points (R, B, M) of the v3.3 definition, patch holdout, paper LightGBM",
        "provenance": {**git, "script": "scripts/run_presence_gate_points.py --engine holdout --summarize", "preset": HOLDOUT_PRESET,
                       "label_key": LABEL_KEY, "split": sig["patches"], "seed": int(args.seed), "n_matches": sig["n_matches"],
                       "sample_digest": sig["sample_digest"], "n_boot": int(args.n_boot), "tolerance_ms": tol,
                       "reference": ref_name, "point_commits": sorted({js["provenance"].get("git_commit") for js, _ in loaded.values()}),
                       "points_wall_clock_s": round(sum(js["provenance"]["wall_clock_s"] for js, _ in loaded.values()), 1),
                       "summary_wall_clock_s": None},
        "points_requested": [p[0] for p in points], "points_missing": missing, "points": rows,
        "deviations": [
            "Engagements and test rows are matched one-to-one on (match id, first-kill time) within the tolerance; G and D "
            "are fixed across points, so shared engagements normally match at dt = 0 (reported).",
            "Labels of a shared engagement can differ between points because the label window starts at the cutoff, "
            "which moves with B; the shared-row difference is split into label and model effects.",
            "Per-point intervals and the paired differences come from separate replicate streams with the same seed.",
        ],
    }
    summary["provenance"]["summary_wall_clock_s"] = round(time.time() - t0, 1)
    out = args.out_dir / "presence_gate_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(args.out_dir / "presence_gate_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "R_u", "B_ms", "M", "engagements", "rows_test", "positive_rate", "share_ref_matched", "jaccard",
                    "auc_test", "auc_lo", "auc_hi", "d_full", "d_full_lo", "d_full_hi", "n_shared", "label_agreement_shared",
                    "d_shared", "d_shared_lo", "d_shared_hi", "auc_pick", "auc_skirmish", "auc_teamfight"])
        for r in rows:
            t, o, p, s = r["test"], r["overlap_engagements"], r["paired_vs_reference"], r["test_rows_vs_reference"]
            w.writerow([r["name"], r["R_u"], r["B_ms"], r["M_effective"], r["counts"]["all"]["n_engagements"],
                        r["counts"]["test"]["n_rows_labelled"], r["counts"]["all"]["positive_rate"],
                        round(o["share_of_operating_point_matched"], 4), round(o["jaccard"], 4), t["auc"].get("point"),
                        t["auc"].get("ci_2.5"), t["auc"].get("ci_97.5"), p["full_setting_minus_op"].get("point"),
                        p["full_setting_minus_op"].get("ci_2.5"), p["full_setting_minus_op"].get("ci_97.5"), s["n_shared"],
                        s["label_agreement_shared"], p["shared_setting_minus_op"].get("point"),
                        p["shared_setting_minus_op"].get("ci_2.5"), p["shared_setting_minus_op"].get("ci_97.5"),
                        t["classes"]["pick"]["auc"].get("point"), t["classes"]["skirmish"]["auc"].get("point"),
                        t["classes"]["teamfight"]["auc"].get("point")])
    f_ = boot_mod.fmt_ci
    log("\n| point | R (u) | B (s) | M | engagements | test rows | ref matched | test AUC | vs ref (full) | shared | vs ref (shared) | label effect | pick | teamfight |")
    log("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t, o, p = r["test"], r["overlap_engagements"], r["paired_vs_reference"]
        log(f"| {r['name']} | {r['R_u']:g} | {r['B_ms'] / 1000:g} | {r['M_effective']} | {r['counts']['all']['n_engagements']:,} | "
            f"{r['counts']['test']['n_rows_labelled']:,} | {o['share_of_operating_point_matched']:.3f} | {f_(t['auc'])} | "
            f"{f_(p['full_setting_minus_op'], True)} | {r['test_rows_vs_reference']['n_shared']:,} | {f_(p['shared_setting_minus_op'], True)} | "
            f"{f_(p['shared_label_effect'], True)} | {f_(t['classes']['pick']['auc'])} | {f_(t['classes']['teamfight']['auc'])} |")
    log(f"wrote {out} (missing: {missing})")
    return 0


def run_holdout_engine(args, points: list, extra: dict) -> int:
    global _LOG_PATH
    import run_gd_sweep_v33 as sweep
    boot_mod = load_boot_module()
    preset = args.preset or HOLDOUT_PRESET
    if preset != HOLDOUT_PRESET:
        raise SystemExit(f"--engine holdout is defined on preset {HOLDOUT_PRESET}")
    inherited = os.environ.get("LOL_CFG_PRESET", "").strip()
    if inherited and inherited != preset:
        raise SystemExit(f"LOL_CFG_PRESET={inherited!r}; the holdout engine runs preset {preset}")
    if os.environ.get("LOL_CFG_OVERRIDES", "").strip():
        raise SystemExit("unset LOL_CFG_OVERRIDES: the holdout engine passes each point's constants to its workers itself")
    clash = sorted(set(extra) & (set(GATE_KEYS) | set(sweep.operational_overrides())))
    if clash:
        raise SystemExit(f"--set may not change {clash}; use the point spec")
    os.environ["LOL_CFG_PRESET"] = preset
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # one log per point job, so points queued as parallel jobs never interleave their lines
    single = len(points) == 1 and not args.summarize and not args.dry_run
    _LOG_PATH = args.out_dir / (f"presence_gate_{points[0][0]}.log" if single else "presence_gate.log")
    git = boot_mod.git_provenance()
    names = [p[0] for p in points]
    if len(set(names)) != len(names):
        raise SystemExit("point names must be unique")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} git={git['git_commit'][:10]} engine=holdout preset={preset} "
        f"points={[':'.join(str(x) for x in p if x is not None) for p in points]}")
    if args.summarize:
        return summarize_holdout(args, points, sweep, boot_mod, git)

    if args.dry_run:
        for name, R, B, M in points:
            overrides = {**point_overrides(R, B, M), **extra, **sweep.operational_overrides()}
            env = {**os.environ, "LOL_CFG_PRESET": preset, "LOL_CFG_OVERRIDES": json.dumps(overrides)}
            effective = probe_cfg(env)
            problems = cfg_problems(effective, overrides, preset, boot_mod)
            log(f"[{name}] worker overrides {json.dumps(overrides)}")
            log(f"[{name}] effective: " + json.dumps({k: effective.get(k) for k in PROBE_KEYS if k != "BASELINE_LGB_PARAMS"}))
            log(f"[{name}] " + ("cfg OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
        log(f"booster parameters: {json.dumps(paper_booster_params(args.seed, args.n_jobs, sweep), default=str)}")
        return 0

    from core.config import CACHE_DIR
    all_mids = sorted(p.name[: -len(".meta.json")] for p in CACHE_DIR.glob("*.meta.json"))
    mids = list(all_mids)
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    if args.check_equivalence and (not args.n_matches or args.n_matches > 2000):
        raise SystemExit("--check-equivalence keeps every array; use it with --n-matches <= 2000")
    digest = hashlib.blake2b("\n".join(mids).encode("utf-8"), digest_size=12).hexdigest()
    patch_cache = args.out_dir / ("match_patch_map.json" if not args.n_matches else f"match_patch_map_{digest}.json")
    patch_of = read_match_patches(mids, CACHE_DIR, patch_cache, args.patch_threads)
    split_patches = {k: [p.strip() for p in str(v).split(",") if p.strip()]
                     for k, v in (("train", args.train_patches), ("val", args.val_patches), ("test", args.test_patches))}
    if set(split_patches["train"]) & set(split_patches["val"]) or set(split_patches["test"]) & (set(split_patches["train"]) | set(split_patches["val"])):
        raise SystemExit(f"patches overlap between splits: {split_patches}")
    phase_mids = {k: [m for m in mids if patch_of[m] in set(v)] for k, v in split_patches.items()}
    other = len(mids) - sum(len(v) for v in phase_mids.values())
    for k, v in phase_mids.items():
        if not v:
            raise SystemExit(f"no sampled match on the {k} patches {split_patches[k]}")
    log(f"matches: {len(mids):,}/{len(all_mids):,} sampled; train {len(phase_mids['train']):,} val {len(phase_mids['val']):,} "
        f"test {len(phase_mids['test']):,} other patches {other:,}")
    ctx = {"sweep": sweep, "boot_mod": boot_mod, "git": git, "extra": extra, "mids": mids, "digest": digest,
           "cache_dir": str(CACHE_DIR), "n_cache": len(all_mids), "pos": {m: i for i, m in enumerate(all_mids)},
           "phase_mids": phase_mids, **split_patches}
    for name, R, B, M in points:
        run_holdout_point(name, R, B, M, args, ctx)
    return 0


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", default=DEFAULT_POINTS, help="name:R_units:B_ms[:M]")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-matches", type=int, default=210000, help="runner engine: runner.py --max_matches")
    ap.add_argument("--workers", type=int, default=8, help="fight-index (runner) or row-building (holdout) worker processes")
    ap.add_argument("--models", default="lgbm", help="runner engine: runner.py --models")
    ap.add_argument("--train-patches", default="15.14")
    ap.add_argument("--val-patches", default="15.15")
    ap.add_argument("--test-patches", default="15.16")
    ap.add_argument("--group", default="runs_presence_gate", help="runner engine: output folder under LOL_OUTPUT_ROOT")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra CFG override applied to every point, e.g. --set TF2_KILL_CLUSTER_GAP_MS=13700 "
                         "--set CLUSTER_MAX_DIAMETER=4264 (corpus v3: G 13.7 s, D 4,264 u)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and probe each point's effective cfg in a fresh interpreter; run nothing")
    ap.add_argument("--engine", choices=("runner", "holdout"), default="runner")
    ap.add_argument("--preset", default=None,
                    help="core/presets.py preset for the child processes (runner default: inherit LOL_CFG_PRESET; holdout: v3.3)")
    ap.add_argument("--tie-policy", choices=("drop", "random"), default=None,
                    help="runner engine: LABEL_TIE_STRATEGY for the child (default: unchanged, i.e. random)")
    ap.add_argument("--no-row-cap", action="store_true",
                    help="runner engine: GLOBAL_SUBSAMPLE_PER_SPLIT=0 (no 100,000-row cap per split)")
    ap.add_argument("--n-matches", type=int, default=0, help="holdout engine: declared match subsample (0 = full corpus)")
    ap.add_argument("--chunk-size", type=int, default=40, help="holdout engine: matches per worker task")
    ap.add_argument("--n-jobs", type=int, default=8, help="holdout engine: LightGBM threads")
    ap.add_argument("--n-boot", type=int, default=1000, help="holdout engine: match bootstrap replicates")
    ap.add_argument("--teamfight-min", type=int, default=4)
    ap.add_argument("--negative-count-as", choices=("zero", "unknown"), default="zero")
    ap.add_argument("--tolerance-ms", type=int, default=5000, help="holdout summary: one-to-one matching tolerance")
    ap.add_argument("--patch-threads", type=int, default=16, help="holdout engine: threads reading match patches")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_HOLDOUT_OUT, help="holdout engine output folder")
    ap.add_argument("--summarize", action="store_true", help="holdout engine: paired comparison of finished points")
    ap.add_argument("--reference", default=None, help="holdout summary: reference point name")
    ap.add_argument("--check-equivalence", action="store_true",
                    help="holdout engine: also fit LGBMClassifier on concatenated arrays and compare (small runs only)")
    ap.add_argument("--force", action="store_true", help="holdout engine: recompute finished points")
    args = ap.parse_args(argv)

    points = [parse_point(s) for s in args.points]
    extra = parse_set(args.set)
    if args.engine == "holdout":
        return run_holdout_engine(args, points, extra)
    return run_runner_engine(args, points, extra)


if __name__ == "__main__":
    sys.exit(main())
