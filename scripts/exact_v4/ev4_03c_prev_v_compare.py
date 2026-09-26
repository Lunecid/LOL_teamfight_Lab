"""v4-exact stage 2, E4 add-on: score the previous frozen evaluator A_MLP_expanded on the 15.15 V_SELECT rows.

Pre-specified by the V-revision record (author, chat, before any refit):
  outputs/reest_exact_v4_20260925/records/v_revision_prespec_20260925T233953Z.json, key 'also_in_this_refit':
  "E4 comparison with the previous frozen evaluator A_MLP_expanded, read from C:/Users/todtj/PycharmProjects/
  LOL_teamfight/outputs/v_redesign_wave4_corrected_20260919/evaluators/ only (author permission for this one folder),
  scored on the same 15.15 V_SELECT rows with its own StateV2 features".  Nothing else from that record is done here.

Implementation decisions (2026-09-26, before any full-run result; recorded here, in DECISIONS and in the summary):
  P1  Rows = the ev4_03 V_SELECT rows: every row of <extract>/15.15/chunk_*_v.parquet (match_id, t, y_blue_win)
      whose match has ev4_v_models.v_split(match_id) == 'select' (sha256('<mid>:ev4_03:vsplit') even), in extract
      order (chunk, row).  The manifest must say patch 15.15 (split_guard.assert_selection_patches first, then
      equality to 15.15); every _v.parquet is checked against its manifest sha256; no row may have
      game_end < 300,000 (remake rule).  --limit-matches N (smoke only) keeps the first N V_SELECT matches by
      (sha256(match_id), match_id).
  P2  Bundle = <evaluators>/A_MLP_expanded_evaluator.joblib, read in place (never copied; weights live only in
      memory).  Its sha256 must equal <evaluators>/A_MLP_expanded_evaluator.sha256; the name / bytes / sha256 of
      every file in that folder are recorded.  Loaded and scored with the old module
      C:/Users/todtj/PycharmProjects/LOL_teamfight/scripts/v_redesign_evaluator_bundle.py (imported read-only from
      that folder, bytecode writing off; its feature adapters v_redesign_feature_adapters.py come with it):
      p_prev_raw = predict_raw_mlp(ev, X), p_prev = PosSlopeSigmoid.from_dict(ev['calibration']).transform(raw),
      i.e. exactly predict_calibrated (V = g(f(T(X)))).  p_prev is the frozen evaluator's output.
  P3  StateV2 at t exactly as the old full-corpus extraction (scripts/fc20260915_extract.py): gameplay.state_value_v2
      StateBuilder(pack, core.config.NODE_FEATURE_NAMES).at(t), matrix by state_value_v2.state_matrix in the bundle's
      raw_names order (which must equal the StateV2 names built on a synthetic fixture), float64.  The worktree
      state_value_v2.py must hash to the bundle's meta state_value_v2_sha256_16.  Pack = ev4_02_extract.load_pack
      (the same cache, meta patch must be 15.15; no xy_raw_minute) with node_minute cast to float32 and the
      meta feature_version required to equal cfg.FEATURE_VERSION, as data.cache_io.load_match_cache does.
      Parity gate: for the first --parity-matches matches (default 20) the matrix from the old loader
      data.cache_io.load_match_cache must be identical (NaN == NaN); any difference stops the run.
  P4  Linkage gate: StateV2 snapshot_age_s * 1000 must equal the extract's frame_age_ms (|diff| <= 1 ms) on every
      scored row (same cache, same frame rule); any difference stops the run.
  P5  Rows the old builder cannot serve (e.g. t after the last frame, roster errors, feature-version mismatch) or
      with a non-finite prediction get p_prev = NaN and a status; they are counted, never imputed.  Unknown champions
      (embedding index 0 = UNK in the old vocab) are counted per slot.
  P6  Summary: log loss, Brier, AUC, calibration slope / intercept / calibration-in-the-large (ev4_v_models.
      binary_metrics) for p_prev and p_prev_raw, overall and by the E4 minute bins (<15, 15-25, >=25 min) and by
      5-minute bins (2-5, 5-10, ..., 35-40, >=40 min); frame-age distribution of the scored rows.
  P7  compare_with_fit_v(prev_parquet, fit_v_dir): identical rows = inner join on (match_id, t) with the fit-V
      directory's predictions_v_15.15.parquet split == 'select'; the prev set must equal that set (a subset only
      with allow_subset, smoke); y_blue_win must agree; rows with a non-finite p on either side are dropped from BOTH
      (counted).  Chosen V = frozen_manifest.json 'chosen', column p_cal_<chosen> (the frozen, calibrated V), or
      --v-column (e.g. a recalibrated V after the refit); p_raw_<chosen> reported secondary.  Differences
      delta = metric(prev) - metric(V) (positive = previous evaluator worse on loss / Brier): log loss and Brier as
      row means with match-cluster CR1 SEs (ev4_v_models.cluster_mean_se), overall and by E4 minute bin; AUC
      difference with a match-cluster bootstrap SE (--auc-boot replicates, default 200, seed 20260926).
  P8  Guards: ev4_common.predecisions_info(); the v-revision record must exist (its sha256 is recorded); the run is
      inside grid_guard.forbid_grid(); nothing is ever written under C:/Users/todtj/PycharmProjects (refused) and
      nothing there is read except the evaluators folder and the two old modules; the extract and fit-V directories
      are read only; an existing output parquet needs --force.
  Disclosure (summary 'disclosures'): the old evaluator was fitted on minute-grid queries (t = 120 s + k * 60 s,
  node features from the frame at that minute); these rows are uniform within 2-minute buckets, so node features
  come from a frame up to ~60 s old while event counts are current at t (snapshot_age_s is dropped by the old
  schema).  Both evaluators see the same frame age at each row.

Outputs (<out>, default outputs/reest_exact_v4_20260925/stage2/prev_v_compare, or .../prev_v_compare_smoke with
--limit-matches):
  prev_v_15.15.parquet         match_id, t, y_blue_win, p_prev, p_prev_raw, status
  summary_prev_v_15.15.json    bundle files / sha256, inputs, checks, metrics, runtime and projection
  compare_vs_<fit-V dir name>.json   (with --fit-v, or the 'compare' subcommand)
  run.log

CLI
  python scripts/exact_v4/ev4_03c_prev_v_compare.py score [--extract DIR] [--out DIR] [--workers 1]
         [--threads 4] [--parity-matches 20] [--fit-v DIR [--v-column COL]] [--force]
  smoke: ... score --limit-matches 200 [--fit-v DIR]          (writes to prev_v_compare_smoke; subset compare)
  python scripts/exact_v4/ev4_03c_prev_v_compare.py compare --prev PARQUET --fit-v DIR [--v-column COL]
         [--out JSON] [--allow-subset] [--auc-boot 200]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys  # noqa: E402

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
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

PATCH = "15.15"
OUT_BASE = EC.OUT_BASE
DEFAULT_EXTRACT = OUT_BASE / "stage2" / "extract" / PATCH
DEFAULT_OUT = OUT_BASE / "stage2" / "prev_v_compare"
DEFAULT_OUT_SMOKE = OUT_BASE / "stage2" / "prev_v_compare_smoke"
PREV_PARQUET = f"prev_v_{PATCH}.parquet"
SUMMARY_JSON = f"summary_prev_v_{PATCH}.json"
V_REVISION_RECORD = OUT_BASE / "records" / "v_revision_prespec_20260925T233953Z.json"

OLD_REPO = Path(r"C:/Users/todtj/PycharmProjects/LOL_teamfight")
OLD_SCRIPTS = OLD_REPO / "scripts"
EVAL_DIR = OLD_REPO / "outputs" / "v_redesign_wave4_corrected_20260919" / "evaluators"
BUNDLE_NAME = "A_MLP_expanded_evaluator.joblib"
BUNDLE_SHA_NAME = "A_MLP_expanded_evaluator.sha256"
OLD_MODULES = ("v_redesign_evaluator_bundle", "v_redesign_feature_adapters")
EVALUATOR_ID = "A_MLP_expanded"

FRAME_AGE_TOL_MS = 1.0
AUC_BOOT = 200
AUC_SEED = 20260926
BATCH_MATCHES = 500
FIVE_MIN_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = tuple(
    [("2to5", 120_000, 300_000)]
    + [(f"{a}to{a + 5}", a * 60_000, (a + 5) * 60_000) for a in range(5, 40, 5)]
    + [("ge40", 2_400_000, None)])

DECISIONS = {
    "P1_rows": "ev4_03 V_SELECT rows: extract 15.15 chunk_*_v.parquet rows with ev4_v_models.v_split == 'select', "
               "extract order; manifest patch 15.15; _v.parquet sha256 verified; no remake rows",
    "P2_bundle": "A_MLP_expanded_evaluator.joblib read in place, sha256 == .sha256 file; old "
                 "v_redesign_evaluator_bundle predict_raw_mlp + PosSlopeSigmoid (== predict_calibrated)",
    "P3_statev2": "state_value_v2.StateBuilder(pack, NODE_FEATURE_NAMES).at(t) + state_matrix in bundle raw_names "
                  "order; ev4_02 load_pack + float32 node + feature_version check; parity with "
                  "data.cache_io.load_match_cache on the first --parity-matches matches (exact)",
    "P4_linkage": "snapshot_age_s * 1000 == extract frame_age_ms (<= 1 ms) on every scored row",
    "P5_failures": "unservable rows / non-finite predictions -> NaN + status, counted, never imputed; UNK champions "
                   "counted",
    "P6_summary": "binary_metrics for p_prev and p_prev_raw, overall, E4 minute bins, 5-minute bins",
    "P7_compare": "inner join with fit-V predictions (split select) on (match_id, t); equal row sets (subset only "
                  "smoke); y agrees; non-finite dropped from both; V = p_cal_<chosen> (or --v-column); "
                  "delta = prev - V; log loss / Brier CR1 match-cluster SE; AUC match-cluster bootstrap SE",
    "P8_guards": "predecisions_info; v-revision record present; forbid_grid; no writes under PycharmProjects; "
                 "extract / fit-V read only; --force to overwrite",
}
DISCLOSURES = [
    "The previous evaluator was fitted on minute-grid queries (t = 120 s + k * 60 s); the V_SELECT rows are uniform "
    "within 2-minute buckets, so its node (frame) features come from a frame up to ~60 s old while event counts are "
    "current at t (snapshot_age_s is dropped by its schema).  The new V sees the same frame at each row.",
    "The previous evaluator was fitted and calibrated on earlier-patch data (its own TRAIN / V_CAL); it is scored "
    "here without any refit or recalibration on 15.15.",
]


# ============================================================================ logging / small helpers
class Log:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[ev4_03c {time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _is_under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def assert_writable_out(p: Path) -> None:
    """Refuse any output location inside the old repository (read-only by author rule)."""
    if _is_under(p, OLD_REPO):
        raise PermissionError(f"refusing to write under {OLD_REPO}: {p}")


def _write_json(p: Path, obj: Any) -> None:
    assert_writable_out(p)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=VM._json_default) + "\n", encoding="utf-8")


def code_hashes() -> Dict[str, str]:
    out = {"scripts/exact_v4/ev4_03c_prev_v_compare.py": VM.sha256_file(HERE),
           "scripts/exact_v4/ev4_v_models.py": VM.sha256_file(HERE.parent / "ev4_v_models.py"),
           "scripts/exact_v4/ev4_common.py": VM.sha256_file(HERE.parent / "ev4_common.py"),
           "scripts/exact_v4/ev4_02_extract.py": VM.sha256_file(HERE.parent / "ev4_02_extract.py")}
    for m in ("gameplay/state_value_v2.py", "gameplay/state_value.py", "data/cache_io.py", "core/config.py",
              "gameplay/split_guard.py", "gameplay/grid_guard.py"):
        p = WT / m
        if p.exists():
            out[m] = VM.sha256_file(p)
    for name in OLD_MODULES:
        out[f"{OLD_SCRIPTS.as_posix()}/{name}.py"] = VM.sha256_file(OLD_SCRIPTS / f"{name}.py")
    return out


# ============================================================================ previous evaluator bundle
def bundle_files_info(eval_dir: Path = EVAL_DIR) -> Dict[str, Any]:
    """name / bytes / sha256 of every file in the evaluators folder; the joblib must match its .sha256 file."""
    eval_dir = Path(eval_dir)
    files = {}
    for f in sorted(eval_dir.iterdir(), key=lambda x: x.name):
        if f.is_file():
            files[f.name] = {"bytes": f.stat().st_size, "sha256": VM.sha256_file(f)}
    if BUNDLE_NAME not in files or BUNDLE_SHA_NAME not in files:
        raise FileNotFoundError(f"{eval_dir} lacks {BUNDLE_NAME} or {BUNDLE_SHA_NAME}")
    expected = (eval_dir / BUNDLE_SHA_NAME).read_text(encoding="utf-8").strip().split()[0].lower()
    got = files[BUNDLE_NAME]["sha256"]
    if got != expected:
        raise RuntimeError(f"{BUNDLE_NAME} sha256 {got} != {BUNDLE_SHA_NAME} {expected}")
    return {"dir": str(eval_dir), "files": files, "bundle": BUNDLE_NAME, "bundle_sha256": got,
            "bundle_sha256_matches_sidecar": True}


def import_old_modules():
    """Import v_redesign_evaluator_bundle (and its adapters) from the old scripts folder, read only."""
    sys.dont_write_bytecode = True
    if str(OLD_SCRIPTS) not in sys.path:
        sys.path.append(str(OLD_SCRIPTS))
    import v_redesign_evaluator_bundle as EB  # noqa: E402
    import v_redesign_feature_adapters as FA  # noqa: E402
    for mod in (EB, FA):
        if not _is_under(Path(mod.__file__), OLD_SCRIPTS):
            raise RuntimeError(f"{mod.__name__} imported from {mod.__file__}, not from {OLD_SCRIPTS}")
    return EB


def load_prev_evaluator(eval_dir: Path = EVAL_DIR) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    info = bundle_files_info(eval_dir)
    EB = import_old_modules()
    ev = EB.load_evaluator(Path(eval_dir) / BUNDLE_NAME)
    if ev.get("evaluator_id") != EVALUATOR_ID or ev.get("kind") != "mlp_embedding":
        raise RuntimeError(f"unexpected evaluator {ev.get('evaluator_id')!r} / {ev.get('kind')!r}")
    sv2 = WT / "gameplay" / "state_value_v2.py"
    want = (ev.get("meta") or {}).get("state_value_v2_sha256_16")
    have = VM.sha256_file(sv2)[:16]
    if want != have:
        raise RuntimeError(f"state_value_v2.py hash16 {have} != bundle meta {want}")
    info.update(evaluator_id=ev["evaluator_id"], kind=ev["kind"], input_impl=ev.get("input_impl"),
                fit_scope=ev.get("fit_scope"), calibration=ev.get("calibration"),
                evaluation_map=ev.get("evaluation_map"), state_value_v2_sha256_16=have,
                n_raw_names=len(ev["preproc"]["schema"]["raw_names"]),
                model_spec={k: v for k, v in ev["model"].items() if k != "state_dict"})
    return ev, EB, info


def predict_prev(ev: Mapping[str, Any], EB, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(raw f(T(X)), calibrated g(f(T(X)))) with the old module; rows with non-finite X get NaN."""
    n = len(X)
    raw = np.full(n, np.nan)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        raw[ok] = EB.predict_raw_mlp(ev, X[ok])
    cal = np.full(n, np.nan)
    fin = np.isfinite(raw)
    if fin.any():
        cal[fin] = EB.PosSlopeSigmoid.from_dict(ev["calibration"]).transform(raw[fin])
    return raw, cal


def unk_champion_counts(ev: Mapping[str, Any], EB, X: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(X).all(axis=1)
    if not ok.any():
        return {"rows": 0}
    bun = EB.bundle_from_dict(ev["preproc"])
    ids = bun.embedding_ids(X[ok])
    unk = ids == 0
    return {"rows": int(ok.sum()), "rows_with_any_unk": int(unk.any(axis=1).sum()), "unk_slots": int(unk.sum()),
            "unk_per_slot": unk.sum(axis=0).astype(int).tolist()}


# ============================================================================ StateV2 at t
_CTX: Dict[str, Any] = {}


def _context() -> Dict[str, Any]:
    if not _CTX:
        from core.config import cfg, NODE_FEATURE_NAMES
        from gameplay.state_value_v2 import StateBuilder, STATE_VERSION, state_matrix
        _CTX.update(cfg=cfg, node_names=list(NODE_FEATURE_NAMES), V2=StateBuilder, state_matrix=state_matrix,
                    state_version=STATE_VERSION, feature_version=str(cfg.FEATURE_VERSION))
    return _CTX


def expected_state_names() -> List[str]:
    """StateV2 names from code only (synthetic fixture, as fc20260915_extract.expected_state_names)."""
    ctx = _context()
    nn = ctx["node_names"]
    pack = dict(minute_ts=np.array([0, 60000, 120000]), node_minute=np.zeros((3, 10, len(nn))), events=[],
                meta=dict(team_map={p: 100 if p <= 5 else 200 for p in range(1, 11)}))
    return list(ctx["V2"](pack, nn).at(60000).values)


def load_pack_v2(mid: str, cache: Path = EX.CACHE) -> Dict[str, Any]:
    """ev4_02 load_pack (patch 15.15 enforced) + the two things data.cache_io.load_match_cache adds that the state
    builder can see: feature_version check and float32 node_minute."""
    pack = EX.load_pack(mid, PATCH, cache)
    fv = str(pack["meta"].get("feature_version", ""))
    if fv != _context()["feature_version"]:
        raise ValueError(f"feature_version_mismatch:{fv!r}")
    pack["node_minute"] = np.asarray(pack["node_minute"]).astype(np.float32)
    return pack


def states_for_match(pack: Mapping[str, Any], ts: Sequence[int], names: Sequence[str]
                     ) -> Tuple[np.ndarray, List[str]]:
    """(len(ts), len(names)) float64 StateV2 matrix (NaN rows where the builder refuses) and a status per row."""
    ctx = _context()
    X = np.full((len(ts), len(names)), np.nan)
    status = ["ok"] * len(ts)
    try:
        b = ctx["V2"](pack, ctx["node_names"])
    except Exception as exc:  # roster / schema
        return X, [f"builder:{type(exc).__name__}:{exc}"[:120]] * len(ts)
    for i, t in enumerate(ts):
        try:
            st = b.at(int(t))
            X[i] = ctx["state_matrix"]([st], names)[0]
        except Exception as exc:
            status[i] = f"state:{type(exc).__name__}:{exc}"[:120]
    return X, status


def build_batch(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Worker unit: StateV2 rows of a batch of matches.  task = {items: [(mid, [row idx], [t])], names, cache}."""
    t0 = time.time()
    names = task["names"]
    n = sum(len(it[1]) for it in task["items"])
    X = np.full((n, len(names)), np.nan)
    idx = np.empty(n, dtype=np.int64)
    status: List[str] = []
    a = 0
    for mid, rows, ts in task["items"]:
        k = len(rows)
        idx[a:a + k] = rows
        try:
            pack = load_pack_v2(mid, Path(task["cache"]))
        except Exception as exc:
            status += [f"load:{type(exc).__name__}:{exc}"[:120]] * k
            a += k
            continue
        Xm, st = states_for_match(pack, ts, names)
        X[a:a + k] = Xm
        status += st
        a += k
    return {"idx": idx, "X": X, "status": status, "seconds": time.time() - t0, "n_matches": len(task["items"])}


def old_loader_parity(items: Sequence[Tuple[str, Sequence[int], Sequence[int]]], names: Sequence[str],
                      cache: Path = EX.CACHE) -> Dict[str, Any]:
    """StateV2 from data.cache_io.load_match_cache (the old pipeline's loader) vs load_pack_v2: must be identical."""
    import data.cache_io as cio
    from data import ram_cache
    ctx = _context()
    ctx["cfg"].CACHE_IN_RAM = False
    if hasattr(ctx["cfg"], "CACHE_MATCH_PACKS_IN_RAM"):
        ctx["cfg"].CACHE_MATCH_PACKS_IN_RAM = False
    if ram_cache._ram_cache_enabled():
        raise RuntimeError("RAM pack cache unexpectedly enabled")
    saved = cio.CACHE_DIR
    cio.CACHE_DIR = Path(cache)
    try:
        n_rows = n_diff = 0
        bad: List[str] = []
        for mid, _rows, ts in items:
            old = cio.load_match_cache(mid)
            if old is None:
                bad.append(f"{mid}:old_loader_none")
                continue
            if str(old["meta"].get("patch")) != PATCH:
                raise RuntimeError(f"{mid}: old loader patch {old['meta'].get('patch')!r}")
            Xo, so = states_for_match(old, ts, names)
            Xn, sn = states_for_match(load_pack_v2(mid, cache), ts, names)
            same = (so == sn) and bool(np.array_equal(Xo, Xn, equal_nan=True))
            n_rows += len(ts)
            if not same:
                n_diff += 1
                bad.append(mid)
    finally:
        cio.CACHE_DIR = saved
    return {"n_matches": len(items), "n_rows": n_rows, "n_matches_differing": n_diff, "bad": bad[:20],
            "pass": n_diff == 0 and not bad}


# ============================================================================ rows
def read_extract_manifest(d: Path) -> Dict[str, Any]:
    from gameplay.split_guard import assert_selection_patches
    p = Path(d) / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"extract manifest missing: {p}")
    man = json.loads(p.read_text(encoding="utf-8"))
    assert_selection_patches([str(man.get("patch"))])
    if str(man.get("patch")) != PATCH:
        raise RuntimeError(f"{d}: extract patch {man.get('patch')!r}; this script scores {PATCH} only")
    return man


def load_select_rows(d: Path, man: Mapping[str, Any], limit_matches: Optional[int] = None):
    """V_SELECT rows (match_id, t, y_blue_win, frame_age_ms, game_end, chunk, row) in extract order."""
    import pandas as pd
    parts = []
    for cid in sorted((man.get("chunks") or {}).keys()):
        fn = f"chunk_{int(cid):05d}_v.parquet"
        ent = man["chunks"][cid]["files"][fn]
        if VM.sha256_file(Path(d) / fn) != ent["sha256"]:
            raise RuntimeError(f"{d}/{fn} does not match its manifest sha256")
        v = pd.read_parquet(Path(d) / fn, columns=["row", "match_id", "t", "y_blue_win", "frame_age_ms", "game_end"])
        v["chunk"] = int(cid)
        parts.append(v)
    v = pd.concat(parts, ignore_index=True)
    v["match_id"] = v["match_id"].astype(str)
    split = VM.per_match(v["match_id"].tolist(), VM.v_split)
    v = v.loc[split == "select"].reset_index(drop=True)
    if (v["game_end"] < EC.REMAKE_MAX_GAME_END_MS).any():
        raise RuntimeError("V_SELECT rows with game_end below the remake threshold")
    if v.duplicated(["match_id", "t"]).any():
        raise RuntimeError("duplicate (match_id, t) among V_SELECT rows")
    if limit_matches:
        mids = sorted(set(v["match_id"]), key=lambda m: (EX.sha256_text(m), m))[:int(limit_matches)]
        v = v.loc[v["match_id"].isin(set(mids))].reset_index(drop=True)
    return v


def make_tasks(v, names: Sequence[str], cache: Path, batch_matches: int = BATCH_MATCHES) -> List[Dict[str, Any]]:
    items = []
    for mid, g in v.groupby("match_id", sort=False):
        items.append((mid, g.index.to_numpy().tolist(), g["t"].astype(int).tolist()))
    return [{"items": items[i:i + batch_matches], "names": list(names), "cache": str(cache)}
            for i in range(0, len(items), batch_matches)]


# ============================================================================ metrics
def _finite_y(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    return y[ok], p[ok], ok


def metrics_block(y, p, t_ms) -> Dict[str, Any]:
    y, p, ok = _finite_y(y, p)
    t_ms = np.asarray(t_ms)[ok]
    out = VM.metrics_by_bins(y, p, t_ms)
    out["by_5min"] = {}
    for name, lo, hi in FIVE_MIN_BINS:
        m = (t_ms >= lo) & ((t_ms < hi) if hi is not None else True)
        out["by_5min"][name] = VM.binary_metrics(y[m], p[m])
    return out


def summarize_predictions(df) -> Dict[str, Any]:
    y, t = df["y_blue_win"].to_numpy(), df["t"].to_numpy()
    fa = df["frame_age_ms"].to_numpy() if "frame_age_ms" in df else None
    out = {"n_rows": int(len(df)), "n_matches": int(df["match_id"].nunique()),
           "n_scored": int(np.isfinite(df["p_prev"]).sum()),
           "status_counts": dict(Counter(df["status"].tolist())),
           "calibrated": metrics_block(y, df["p_prev"].to_numpy(), t),
           "raw": metrics_block(y, df["p_prev_raw"].to_numpy(), t)}
    if fa is not None:
        q = np.percentile(fa, [0, 10, 25, 50, 75, 90, 100]).tolist()
        out["frame_age_ms_quantiles"] = dict(zip(["min", "p10", "p25", "p50", "p75", "p90", "max"], q))
    return out


def _weighted_auc(y, p, w) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p, sample_weight=w))


def auc_delta_bootstrap(y, p_a, p_b, clusters, reps: int = AUC_BOOT, seed: int = AUC_SEED) -> Dict[str, Any]:
    """AUC(a) - AUC(b) with a match-cluster bootstrap SE (matches resampled with replacement = row weights)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.float64)
    if not (0 < y.sum() < len(y)):
        return {"auc_a": None, "auc_b": None, "delta": None, "se_cluster_boot": None, "reps": 0}
    a, b = float(roc_auc_score(y, p_a)), float(roc_auc_score(y, p_b))
    codes, inv = np.unique(np.asarray(clusters), return_inverse=True)
    G = len(codes)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(int(reps)):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(np.float64)
        keep = w > 0
        yk = y[keep]
        if not (0 < yk.sum() < len(yk)):
            continue
        ds.append(_weighted_auc(yk, p_a[keep], w[keep]) - _weighted_auc(yk, p_b[keep], w[keep]))
    se = float(np.std(ds, ddof=1)) if len(ds) > 1 else float("nan")
    d = a - b
    return {"auc_a": a, "auc_b": b, "delta": d, "se_cluster_boot": se, "reps": len(ds), "seed": seed,
            "ci95": [d - 1.96 * se, d + 1.96 * se]}


def paired_deltas(y, p_prev, p_v, t_ms, clusters, auc_boot: int = AUC_BOOT) -> Dict[str, Any]:
    """delta = metric(prev) - metric(V) on identical rows, overall and by E4 minute bin; CR1 match-cluster SEs."""
    y = np.asarray(y, dtype=np.float64)
    p_prev = np.asarray(p_prev, dtype=np.float64)
    p_v = np.asarray(p_v, dtype=np.float64)
    t_ms = np.asarray(t_ms)
    clusters = np.asarray(clusters)
    bins = [("all", 0, None)] + list(VM.MINUTE_BINS)
    out: Dict[str, Any] = {}
    for name, lo, hi in bins:
        m = (t_ms >= lo) & ((t_ms < hi) if hi is not None else True)
        if not m.any():
            out[name] = {"n": 0}
            continue
        ll = VM.paired_logloss_delta(y[m], p_prev[m], p_v[m], clusters[m])
        bd = (p_prev[m] - y[m]) ** 2 - (p_v[m] - y[m]) ** 2
        bm, bse, G = VM.cluster_mean_se(bd, clusters[m])
        ent = {"n": int(m.sum()), "n_matches": int(G),
               "log_loss_prev": VM.log_loss(y[m], p_prev[m]), "log_loss_v": VM.log_loss(y[m], p_v[m]),
               "log_loss_delta": ll,
               "brier_prev": float(np.mean((p_prev[m] - y[m]) ** 2)), "brier_v": float(np.mean((p_v[m] - y[m]) ** 2)),
               "brier_delta": {"delta": bm, "se_cluster": bse, "n_clusters": G,
                               "ci95": [bm - 1.96 * bse, bm + 1.96 * bse]}}
        if auc_boot:
            ent["auc_delta"] = auc_delta_bootstrap(y[m], p_prev[m], p_v[m], clusters[m], reps=auc_boot)
        out[name] = ent
    return out


# ============================================================================ comparison with a fit-V directory
def compare_with_fit_v(prev_parquet: Path, fit_v_dir: Path, v_column: Optional[str] = None,
                       allow_subset: bool = False, auc_boot: int = AUC_BOOT) -> Dict[str, Any]:
    """Previous evaluator vs a fit-V directory's chosen V on identical 15.15 V_SELECT rows (decision P7).
    Works on any ev4_03 output directory (the first fit or the refit)."""
    import pandas as pd
    fit_v_dir = Path(fit_v_dir)
    fm_path = fit_v_dir / "frozen_manifest.json"
    pr_path = fit_v_dir / f"predictions_v_{PATCH}.parquet"
    fm = json.loads(fm_path.read_text(encoding="utf-8"))
    chosen = fm.get("chosen")
    if not chosen:
        raise RuntimeError(f"{fm_path} has no chosen V")
    import pyarrow.parquet as pq
    avail = pq.read_schema(pr_path).names
    col = v_column or f"p_cal_{chosen}"
    raw_col = f"p_raw_{chosen}"
    if col not in avail:
        raise KeyError(f"{pr_path} has no column {col!r} (available: {avail})")
    want = ["match_id", "t", "split", "y_blue_win", col] + ([raw_col] if raw_col in avail and raw_col != col else [])
    pv = pd.read_parquet(pr_path, columns=want)
    pv = pv.loc[pv["split"] == "select"].drop(columns=["split"]).reset_index(drop=True)
    pv["match_id"] = pv["match_id"].astype(str)
    prev = pd.read_parquet(prev_parquet, columns=["match_id", "t", "y_blue_win", "p_prev", "p_prev_raw", "status"])
    prev["match_id"] = prev["match_id"].astype(str)
    for name, df in (("fit-V select", pv), ("prev", prev)):
        if df.duplicated(["match_id", "t"]).any():
            raise RuntimeError(f"{name}: duplicate (match_id, t)")
    m = pv.merge(prev, on=["match_id", "t"], how="outer", suffixes=("_v", "_prev"), indicator=True)
    only_v = int((m["_merge"] == "left_only").sum())
    only_prev = int((m["_merge"] == "right_only").sum())
    if only_prev:
        raise RuntimeError(f"{only_prev} prev rows are not fit-V V_SELECT rows (different row definition)")
    if only_v and not allow_subset:
        raise RuntimeError(f"{only_v} fit-V V_SELECT rows have no prev score (use allow_subset only for a smoke)")
    b = m.loc[m["_merge"] == "both"].reset_index(drop=True)
    if not np.array_equal(b["y_blue_win_v"].to_numpy().astype(int), b["y_blue_win_prev"].to_numpy().astype(int)):
        raise RuntimeError("y_blue_win differs between the fit-V predictions and the prev rows")
    ok = np.isfinite(b["p_prev"].to_numpy()) & np.isfinite(b[col].to_numpy())
    r = b.loc[ok].reset_index(drop=True)
    y, t, cl = r["y_blue_win_v"].to_numpy(), r["t"].to_numpy(), r["match_id"].to_numpy()
    res: Dict[str, Any] = {
        "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "fit_v_dir": str(fit_v_dir), "frozen_manifest_sha256": VM.sha256_file(fm_path),
        "predictions_sha256": VM.sha256_file(pr_path), "frozen": fm.get("frozen"), "smoke": fm.get("smoke"),
        "pilot": fm.get("pilot"), "chosen": chosen, "v_column": col,
        "raw_column": raw_col if raw_col in r else None,
        "prev_parquet": str(prev_parquet), "prev_sha256": VM.sha256_file(Path(prev_parquet)),
        "rows": {"fit_v_select": int(len(pv)), "prev": int(len(prev)), "both": int(len(b)),
                 "only_fit_v": only_v, "only_prev": only_prev, "dropped_nonfinite": int((~ok).sum()),
                 "compared": int(len(r)), "matches_compared": int(r["match_id"].nunique()),
                 "subset": bool(only_v > 0)},
        "sign": "delta = metric(prev A_MLP_expanded) - metric(V); positive log loss / Brier delta = prev worse",
        "metrics": {"prev_cal": metrics_block(y, r["p_prev"], t), "prev_raw": metrics_block(y, r["p_prev_raw"], t),
                    "v": metrics_block(y, r[col], t)},
        "delta_prev_cal_vs_v": paired_deltas(y, r["p_prev"], r[col], t, cl, auc_boot=auc_boot),
    }
    if raw_col in r and raw_col != col:
        res["metrics"]["v_raw"] = metrics_block(y, r[raw_col], t)
        res["delta_prev_cal_vs_v_raw"] = paired_deltas(y, r["p_prev"], r[raw_col], t, cl, auc_boot=0)
    return res


def write_compare(res: Mapping[str, Any], out_dir: Path, fit_v_dir: Path, out_json: Optional[Path] = None) -> Path:
    p = Path(out_json) if out_json else Path(out_dir) / f"compare_vs_{Path(fit_v_dir).resolve().name}.json"
    assert_writable_out(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_json(p, res)
    return p


# ============================================================================ score
def run_score(args) -> Dict[str, Any]:
    import pandas as pd
    t_start = time.time()
    smoke = bool(args.limit_matches)
    out = Path(args.out) if args.out else (DEFAULT_OUT_SMOKE if smoke else DEFAULT_OUT)
    assert_writable_out(out)
    predec = EC.predecisions_info()
    if not V_REVISION_RECORD.is_file():
        raise FileNotFoundError(f"v-revision record missing: {V_REVISION_RECORD}")
    out.mkdir(parents=True, exist_ok=True)
    pq_path = out / PREV_PARQUET
    if pq_path.exists() and not args.force:
        raise FileExistsError(f"{pq_path} exists (use --force)")
    log = Log(out / "run.log")
    log(f"out={out} smoke={smoke} workers={args.workers} threads={args.threads}")
    man = read_extract_manifest(args.extract)
    ev, EB, binfo = load_prev_evaluator(args.eval_dir)
    log(f"bundle {binfo['bundle']} sha256 {binfo['bundle_sha256']} (matches sidecar)")
    names = list(ev["preproc"]["schema"]["raw_names"])
    exp = expected_state_names()
    if names != exp:
        raise RuntimeError("bundle raw_names != StateV2 names built by the current state_value_v2")
    import torch
    torch.set_num_threads(int(args.threads))
    from gameplay.grid_guard import forbid_grid
    with forbid_grid():
        v = load_select_rows(args.extract, man, args.limit_matches)
        n_rows, n_m = len(v), v["match_id"].nunique()
        log(f"V_SELECT rows {n_rows} matches {n_m}")
        tasks = make_tasks(v, names, args.cache, args.batch_matches)
        # parity with the old loader (first K matches)
        t0 = time.time()
        par_items = [it for tk in tasks for it in tk["items"]][:int(args.parity_matches)]
        parity = old_loader_parity(par_items, names, args.cache) if par_items else {"n_matches": 0, "pass": True}
        parity["seconds"] = round(time.time() - t0, 2)
        log(f"old-loader parity: {parity}")
        if not parity["pass"]:
            raise RuntimeError(f"StateV2 differs between data.cache_io.load_match_cache and load_pack_v2: {parity}")
        raw = np.full(n_rows, np.nan)
        cal = np.full(n_rows, np.nan)
        snap_age = np.full(n_rows, np.nan)
        status = np.empty(n_rows, dtype=object)
        unk = {"rows": 0, "rows_with_any_unk": 0, "unk_slots": 0, "unk_per_slot": [0] * 10}
        i_age = names.index("snapshot_age_s")
        build_s = pred_s = 0.0
        t_loop = time.time()

        def consume(res):
            nonlocal build_s, pred_s
            build_s += res["seconds"]
            t1 = time.time()
            r, c = predict_prev(ev, EB, res["X"])
            pred_s += time.time() - t1
            idx = res["idx"]
            raw[idx], cal[idx] = r, c
            snap_age[idx] = res["X"][:, i_age]
            st = list(res["status"])
            for j in range(len(st)):
                if st[j] == "ok" and not np.isfinite(c[j]):
                    st[j] = "nonfinite_prediction"
            status[idx] = st
            u = unk_champion_counts(ev, EB, res["X"])
            for k in ("rows", "rows_with_any_unk", "unk_slots"):
                unk[k] += u.get(k, 0)
            if "unk_per_slot" in u:
                unk["unk_per_slot"] = [a + b for a, b in zip(unk["unk_per_slot"], u["unk_per_slot"])]

        done = 0
        if int(args.workers) <= 1:
            for tk in tasks:
                res = build_batch(tk)
                consume(res)
                done += res["n_matches"]
                log(f"  {done}/{n_m} matches")
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(int(args.workers)) as pool:
                for res in pool.imap(build_batch, tasks):
                    consume(res)
                    done += res["n_matches"]
                    log(f"  {done}/{n_m} matches")
        wall = time.time() - t_loop
    # P4 linkage: StateV2 snapshot age == extract frame age
    ok_state = np.isfinite(snap_age)
    age_diff = np.abs(snap_age[ok_state] * 1000.0 - v["frame_age_ms"].to_numpy()[ok_state])
    n_age_bad = int((age_diff > FRAME_AGE_TOL_MS).sum())
    linkage = {"rows_checked": int(ok_state.sum()), "rows_mismatch": n_age_bad,
               "max_abs_diff_ms": float(age_diff.max()) if len(age_diff) else None, "pass": n_age_bad == 0}
    log(f"frame-age linkage: {linkage}")
    if n_age_bad:
        raise RuntimeError(f"StateV2 snapshot age differs from the extract frame_age_ms on {n_age_bad} rows")
    df = pd.DataFrame({"match_id": v["match_id"].to_numpy(), "t": v["t"].to_numpy().astype(np.int64),
                       "y_blue_win": v["y_blue_win"].to_numpy().astype(np.int8), "p_prev": cal,
                       "p_prev_raw": raw, "status": status.astype(str)})
    df.to_parquet(pq_path, index=False)
    log(f"wrote {pq_path} ({len(df)} rows, {int(np.isfinite(cal).sum())} scored)")
    summ = summarize_predictions(df.assign(frame_age_ms=v["frame_age_ms"].to_numpy()))
    # runtime projection for all V_SELECT rows
    per_match_build = build_s / max(n_m, 1)
    per_row_pred = pred_s / max(n_rows, 1)
    full_rows = full_matches = None
    if smoke:
        vall = load_select_rows(args.extract, man, None)
        full_rows, full_matches = int(len(vall)), int(vall["match_id"].nunique())
        del vall
    else:
        full_rows, full_matches = int(n_rows), int(n_m)
    proj = {"measured": {"matches": int(n_m), "rows": int(n_rows), "wall_s": round(wall, 2),
                         "state_build_worker_s": round(build_s, 2), "predict_s": round(pred_s, 2),
                         "state_build_s_per_match": per_match_build, "predict_s_per_row": per_row_pred,
                         "workers": int(args.workers)},
            "all_v_select": {"matches": full_matches, "rows": full_rows,
                             "single_process_s": round(full_matches * per_match_build + full_rows * per_row_pred, 1),
                             "four_workers_s_ideal": round(full_matches * per_match_build / 4
                                                           + full_rows * per_row_pred, 1),
                             "note": "state building parallelises over workers; scoring runs in the main process"}}
    summary = {
        "script": "scripts/exact_v4/ev4_03c_prev_v_compare.py", "created_utc": time.strftime("%Y%m%dT%H%M%SZ",
                                                                                             time.gmtime()),
        "smoke": smoke, "limit_matches": args.limit_matches, "patch": PATCH,
        "prespec": {"path": str(V_REVISION_RECORD), "sha256": VM.sha256_file(V_REVISION_RECORD),
                    "key": "also_in_this_refit"},
        "predecisions": {"path": predec["path"], "sha256": predec["sha256"]},
        "decisions": DECISIONS, "disclosures": DISCLOSURES,
        "evaluator": binfo,
        "inputs": {"extract": str(args.extract), "extract_manifest_sha256": VM.sha256_file(Path(args.extract)
                                                                                            / "manifest.json"),
                   "extract_git_head": man.get("git_head"), "cache": str(args.cache),
                   "v_parquet_sha256": {f"chunk_{int(c):05d}_v.parquet":
                                        man["chunks"][c]["files"][f"chunk_{int(c):05d}_v.parquet"]["sha256"]
                                        for c in sorted(man["chunks"])}},
        "checks": {"names_equal_state_v2": True, "old_loader_parity": parity, "frame_age_linkage": linkage,
                   "bundle_sha256_matches_sidecar": True,
                   "state_value_v2_hash16_matches_bundle": binfo["state_value_v2_sha256_16"]},
        "unk_champions": unk,
        "results": summ,
        "runtime": proj, "seconds_total": round(time.time() - t_start, 1),
        "code_sha256": code_hashes(), "git_head": EX.git_head(),
        "output": {"parquet": str(pq_path), "parquet_sha256": VM.sha256_file(pq_path)},
    }
    _write_json(out / SUMMARY_JSON, summary)
    log(f"wrote {out / SUMMARY_JSON}")
    if args.fit_v:
        res = compare_with_fit_v(pq_path, args.fit_v, v_column=args.v_column, allow_subset=smoke,
                                 auc_boot=args.auc_boot)
        p = write_compare(res, out, args.fit_v)
        summary["compare"] = str(p)
        log(f"wrote {p}")
    return summary


# ============================================================================ CLI
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score A_MLP_expanded on the 15.15 V_SELECT rows")
    s.add_argument("--extract", type=Path, default=DEFAULT_EXTRACT)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--cache", type=Path, default=EX.CACHE)
    s.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    s.add_argument("--limit-matches", type=int, default=None, help="smoke: first N V_SELECT matches by sha256")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--batch-matches", type=int, default=BATCH_MATCHES)
    s.add_argument("--parity-matches", type=int, default=20)
    s.add_argument("--fit-v", type=Path, default=None, help="also compare with this fit-V directory's chosen V")
    s.add_argument("--v-column", default=None)
    s.add_argument("--auc-boot", type=int, default=AUC_BOOT)
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("compare", help="compare a prev_v parquet with a fit-V directory's chosen V")
    c.add_argument("--prev", type=Path, default=DEFAULT_OUT / PREV_PARQUET)
    c.add_argument("--fit-v", type=Path, required=True)
    c.add_argument("--v-column", default=None)
    c.add_argument("--out", type=Path, default=None, help="output JSON (default <prev dir>/compare_vs_<fit-V>.json)")
    c.add_argument("--allow-subset", action="store_true", help="smoke only: prev rows a subset of V_SELECT")
    c.add_argument("--auc-boot", type=int, default=AUC_BOOT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "score":
        if args.workers > 1 and args.limit_matches:
            print("note: smoke runs are meant single-process", flush=True)
        return run_score(args)
    res = compare_with_fit_v(args.prev, args.fit_v, v_column=args.v_column, allow_subset=args.allow_subset,
                             auc_boot=args.auc_boot)
    p = write_compare(res, Path(args.prev).parent, args.fit_v, args.out)
    print(f"wrote {p}", flush=True)
    d = res["delta_prev_cal_vs_v"]["all"]
    print(json.dumps({"compared": res["rows"]["compared"], "log_loss_delta": d["log_loss_delta"],
                      "brier_delta": d["brier_delta"]}, indent=2, default=VM._json_default), flush=True)
    return res


if __name__ == "__main__":
    main()
