"""TreeSHAP leak forensics on corpus v3.3: the same engagements built clean and leaky, attributed side by side.

Why SHAP, and only here (CoG review R3).  A SHAP value phi_j(x) splits one prediction -- the model's
log-odds for "blue wins this engagement" minus its average log-odds -- into additive per-feature
contributions, using the Shapley value of the coalition game whose payoff for a feature subset S is the
expected model output when only x_S is known:

    phi_j(x) = sum_{S subset F \\ {j}}  |S|! (|F| - |S| - 1)! / |F|!  * ( f_x(S u {j}) - f_x(S) ),
    f_x(S) = E[ f(x) | x_S ],        sum_j phi_j(x) + E[f(x)] = f(x)          (local accuracy)

and mean_x |phi_j(x)| over many engagements is the global importance of feature j.  The revision uses it
for one job: forensic evidence for the paper's central retraction.  The v2/v3 teamfight AUCs of .78-.81
were carried by two input leaks (docs/DEFINITION_EVIDENCE.md sections 20 and 22):

    TIME_NORM_ABSOLUTE = False   time_norm = t / (match length - 1); the match length is unknown at the cutoff
    ANCHORS_CAUSAL     = False   map anchors = every tower / objective destroyed anywhere in the match

If those numbers came from the leaks, a model fitted on the leaky matrix must put its attribution on the
leak columns, and the same learner fitted on the clean matrix of the SAME engagements must not.  The paired
test-AUC difference says how much the leak is worth; TreeSHAP says where the model found it.

Protocol
  1. one declared match sample (--n-matches, --seed) drawn from the v3.3 shard groups exactly as
     run_deep_tabular_baselines.load_subsample draws it, so the same n and seed name the same matches
  2. fight index under the v3.3 preset; y_market_event (or --label market_lex, the retracted v3 artefact's label)
     computed per engagement by build_corpus_shard.label_ref under --extra-tie-policy drop; draws (-1) removed
  3. the tabular matrix built twice on those engagements -- clean (v3.3) and leaky (both switches False,
     everything else v3.3) -- with the corpus's row population (row label market_event, tie policy random);
     row keys (match, cutoff), ref objects, the row label and the feature names asserted identical
  4. the clean build cross-checked against the published shard rows (keys, label, every X value)
  5. leak columns DERIVED from the switches, never listed by hand.  A column is leak-affected when its value
     differs between the two builds on any sample row (a label-free property of the feature code; counts are
     kept per split).  Probe rows -- an even spread plus, for every affected column, --probe-per-column rows
     on which it changed -- are rebuilt under each single-switch configuration (and once more clean, as a
     determinism check), which attributes every column to TIME_NORM_ABSOLUTE, ANCHORS_CAUSAL, both, or only
     their interaction.  Feature names close the set to bases ("<base>__<statistic>": a base is a leak base
     when any of its statistics is affected); shares are reported for the base closure and for the affected
     columns alone
  6. LightGBM on the patch holdout (train 15.14 / early stopping on val 15.15 / test 15.16) exactly as the
     learner comparison runs it (run_deep_tabular_baselines: prepare_matrix standardisation with train-row
     statistics, LGBM_PUBLISHED hyper-parameters, early stopping 50 on validation AUC), one model per
     configuration; test AUC with match-clustered percentile bootstrap CIs and the paired difference
     (run_deep_tabular_baselines.cluster_bootstrap_auc, test matches resampled jointly), overall and by
     participation scale (teamfight = smaller side >= --teamfight-min)
  7. TreeSHAP on one declared seeded sample of test rows, the same rows for both configurations: mean |SHAP|
     per column and per base, top-k bases, rank and share of the leak columns (with a match-clustered CI and
     the paired difference), rank shift between configurations, role rollup (analysis/shap_role_rollup.py),
     and the ranks of the retracted v3-era artefact's top bases (features/shap_mlex.json)
  8. the SHAP values verified against the definition by code that shares nothing with LightGBM (below)

Reference: S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb,
N. Bansal, S.-I. Lee, "From local explanations to global understanding with explainable AI for trees",
Nature Machine Intelligence 2, 56-67 (2020).  Followed: its SHAP values for trees with the path-dependent
conditional expectation (Methods, "Tree SHAP": E[f(x)|x_S] follows x at splits on features in S and averages
both children by training cover otherwise), exact computation by the polynomial-time Tree SHAP algorithm,
and mean |SHAP| per feature as global importance (its summary analyses).  The primary values come from
LightGBM's Booster.predict(pred_contrib=True), which implements that algorithm.  shap 0.50 is installed, but
shap.TreeExplainer(feature_perturbation="tree_path_dependent") on a LightGBM booster calls that same
pred_contrib routine (shap/explainers/_tree.py), so it is only an equality check.  The independent check
(verify_tree_shap) evaluates the definition directly from the dumped trees:
  * by enumeration: f_x(S) for every subset S of a tree's split features, computed with the exponential-time
    EXPVALUE procedure (Algorithm 1 of S. M. Lundberg, G. G. Erion, S.-I. Lee, "Consistent individualized
    feature attribution for tree ensembles", arXiv:1802.03888, 2018), then the Shapley sum above; used for
    trees with at most --verify-enum-max-features split features (all trees of a small auxiliary model fitted
    on train rows, plus any such tree in the explained models);
  * by leaf decomposition, exact and polynomial: f_x(S) = sum_leaves v_l prod_{d on the leaf's path}
    (d in S ? A_ld : B_ld), with A_ld the product of "x follows this path" indicators and B_ld the product of
    child/parent cover ratios at the splits on feature d; the Shapley value of each product game is
    v_l (A_lj - B_lj) sum_k k!(m-k-1)!/m! [z^k] prod_{d != j} (B_ld + A_ld z); used on sampled trees of the
    explained models and on every tree for --verify-full-rows rows.
Local accuracy is checked on every explained row.  Every deviation from the reference and from the
published protocols is written to the output JSON under "deviations".

    LOL_OUTPUT_ROOT=D:/LOL_Project LOL_CFG_PRESET=v3.3 OMP_NUM_THREADS=8 \\
        .venv/Scripts/python.exe scripts/run_shap_leak_forensics_v33.py --n-matches 20000 --n-jobs 8 \\
        --output-dir D:/LOL_Project/fusion_2615/features/tog_revision/A7-shap-forensics

Outputs (stem = --tag, default shap_leak_forensics_v33_m<n>, or shap_leak_forensics_v33_market_lex_m<n> with
--label market_lex): <stem>.json (everything above, provenance and deviations), <stem>.preds.npz (test labels,
groups, scale and both models' test probabilities, for re-bootstrapping), <stem>_{clean,leaky}_mean_abs_shap.csv
(every column; the analysis/shap_role_rollup.py input) and <stem>_{clean,leaky}_base.csv (base rollup with rank,
share and derived leak switch).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 7
CLASSES = ("pick", "skirmish", "teamfight")
ROW_LABEL_TYPE = "market_event"      # the v3.3 shard row population: row label market_event, tie policy random
LABELS = ("market_event", "market_lex")   # --label: the stored y_<type> column (tie policy drop) the models fit
# (TIME_NORM_ABSOLUTE, ANCHORS_CAUSAL); every other field stays at the preset
CONFIGS = {"clean": (True, True), "leaky": (False, False)}
SWITCH_PROBES = {"TIME_NORM_ABSOLUTE": (False, True), "ANCHORS_CAUSAL": (True, False)}
LEAK_GROUPS = ("TIME_NORM_ABSOLUTE", "ANCHORS_CAUSAL", "interaction")
DEFAULT_SHARDS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
DEFAULT_OUTPUT_DIR = Path("D:/LOL_Project/fusion_2615/features/tog_revision/A7-shap-forensics")
DEFAULT_LEGACY_SHAP = Path("D:/LOL_Project/fusion_2615/features/shap_mlex.json")
LGBM_ZERO_THRESHOLD = 1e-35      # LightGBM kZeroThreshold (include/LightGBM/meta.h)
VERIFY_TOLERANCE = 1e-8


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60).stdout.strip()
    try:
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "git_dirty": bool(dirty), "git_dirty_files": dirty.splitlines()[:40],
                "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()}
    except Exception as e:  # git missing: still record the script itself
        return {"git_commit": "", "git_error": repr(e), "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in (sorted(o) if isinstance(o, set) else o)]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sample_matches(shard_dir: Path, n_matches: int | None, seed: int) -> tuple[list[str], int]:
    """The match draw of run_deep_tabular_baselines.load_subsample, read from the shard groups only."""
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir}")
    every = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as blob:
            every.update(np.unique(blob["groups"]).tolist())
    every = sorted(every)
    if not n_matches:
        return every, len(every)
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(np.array(every), size=min(n_matches, len(every)), replace=False).tolist())
    return sorted(keep), len(every)


def configure(preset: str, index_workers: int):
    """Apply the preset through the environment so spawned fight-index workers see the same constants."""
    env_preset = os.environ.get("LOL_CFG_PRESET", "").strip()
    if env_preset and env_preset != preset:
        raise SystemExit(f"LOL_CFG_PRESET={env_preset!r} conflicts with --preset {preset!r}")
    if "core.config" in sys.modules and not env_preset:
        raise SystemExit("core.config was imported before the preset could be set")
    os.environ["LOL_CFG_PRESET"] = preset
    from core.config import cfg
    from core.presets import PRESETS
    drift = {k: [getattr(cfg, k), v] for k, v in PRESETS[preset].items() if getattr(cfg, k) != v}
    if drift:
        raise SystemExit(f"cfg differs from preset {preset!r} (LOL_CFG_OVERRIDES?): {drift}")
    # as build_corpus_shard.py: a per-sample index cache would only churn, dumps are irrelevant
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.DUMP_FIGHTS = False
    cfg.FIGHT_INDEX_NUM_WORKERS = int(index_workers)
    return cfg, dict(PRESETS[preset])


def drop_policy_labels(refs, cfg, label_type: str) -> np.ndarray:
    """y_<label_type> exactly as build_corpus_shard.py --extra-labels <label_type> --extra-tie-policy drop stores it.

    The per-engagement call is build_corpus_shard.label_ref itself (-1 = draw), so the two cannot drift."""
    from data.cache_io import load_match_cache
    bcs = load_module("build_corpus_shard")
    cfg.LABEL_TYPE, cfg.LABEL_TIE_POLICY = label_type, "drop"
    y = np.full(len(refs), -1, dtype=np.int8)
    by_match: dict[str, list[int]] = {}
    for i, r in enumerate(refs):
        by_match.setdefault(r.match_id, []).append(i)
    for mid, idxs in by_match.items():
        pack = load_match_cache(mid)
        if not pack:
            continue
        tm = bcs.team_map_int(pack)
        for i in idxs:
            y[i] = bcs.label_ref(pack, tm, refs[i])
    return y


def build_matrix(refs, switches, cfg, chunk: int, label: str):
    """build_tabular_Xy in ref chunks (rows are independent, so chunking changes nothing but the progress log)."""
    from train.baseline import build_tabular_Xy
    cfg.TIME_NORM_ABSOLUTE, cfg.ANCHORS_CAUSAL = bool(switches[0]), bool(switches[1])
    cfg.LABEL_TYPE, cfg.LABEL_TIE_POLICY = ROW_LABEL_TYPE, "random"   # the corpus's common row population
    parts, row_labels, used_all, names = [], [], [], None
    t0 = time.time()
    for s in range(0, len(refs), chunk):
        X, y_row, nm, used = build_tabular_Xy(refs[s:s + chunk], feature_set="full")
        if len(used):
            if names is None:
                names = list(nm)
            elif list(nm) != names:
                raise SystemExit(f"[{label}] feature names changed between chunks")
            parts.append(X); row_labels.append(np.asarray(y_row)); used_all.extend(used)
        if len(refs) > chunk:
            print(f"  [{label}] {min(s + chunk, len(refs))}/{len(refs)} refs -> {len(used_all)} rows"
                  f" ({time.time() - t0:.0f}s)", flush=True)
    if not parts:
        raise SystemExit(f"[{label}] no rows built")
    out = np.empty((sum(len(p) for p in parts), parts[0].shape[1]), dtype=np.float32)
    pos = 0
    while parts:
        p = parts.pop(0)
        out[pos:pos + len(p)] = p
        pos += len(p)
    return out, names, used_all, np.concatenate(row_labels).astype(np.int8), time.time() - t0


def column_diff(A: np.ndarray, B: np.ndarray, rows=None, chunk: int = 4096, want_examples: int = 0):
    """Compare A and B at the same row positions (``rows`` of both, or every row), chunk by chunk.

    Returns per column the number of rows whose value differs (NaN == NaN) and the largest absolute change,
    up to ``want_examples`` positions (into ``rows``) per column where it differs, and the number of rows
    with any difference.  A selection is never materialised whole."""
    n = len(rows) if rows is not None else len(A)
    if rows is None and len(B) != n:
        raise ValueError("column_diff: row counts differ")
    counts = np.zeros(A.shape[1], dtype=np.int64)
    max_abs = np.zeros(A.shape[1], dtype=np.float64)
    examples = [[] for _ in range(A.shape[1])] if want_examples else None
    n_rows_any = 0
    for s in range(0, n, chunk):
        if rows is not None:
            sel = rows[s:s + chunk]
            a, b = A[sel], B[sel]
        else:
            a, b = A[s:s + chunk], B[s:s + chunk]
        d = (a != b) & ~(np.isnan(a) & np.isnan(b))
        col_hits = d.sum(axis=0)
        counts += col_hits
        n_rows_any += int(d.any(axis=1).sum())
        if col_hits.any():
            with np.errstate(invalid="ignore"):
                ad = np.where(d, np.abs(a.astype(np.float64) - b.astype(np.float64)), 0.0)
            max_abs = np.maximum(max_abs, np.nan_to_num(ad, nan=np.inf).max(axis=0))
            if want_examples:
                for j in np.flatnonzero(col_hits):
                    room = want_examples - len(examples[j])
                    if room > 0:
                        examples[j].extend((s + np.flatnonzero(d[:, j])[:room]).tolist())
    return counts, max_abs, examples, n_rows_any


def shard_crosscheck(shard_dir: Path, keys: list, X: np.ndarray, y: np.ndarray, names: list, label_key: str) -> dict:
    """Does the in-process clean build reproduce the published v3.3 shard rows?"""
    where: dict[tuple, int] = {}
    dup = 0
    for i, k in enumerate(keys):
        if k in where:
            dup += 1
        where[k] = i
    mids = sorted({k[0] for k in keys})
    shard_names = json.loads((shard_dir / "feature_names.json").read_text(encoding="utf-8"))["names"]
    found = np.zeros(len(keys), dtype=bool)
    shard_only, label_mismatch, rows_differ = 0, 0, 0
    col_counts = np.zeros(X.shape[1], dtype=np.int64)
    max_abs = 0.0
    for path in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(path, allow_pickle=True) as blob:
            g = blob["groups"]
            sel = np.flatnonzero(np.isin(g, mids))
            if not len(sel):
                continue
            ts, ys = blob["engage_ts"][sel], blob[label_key][sel]
            lab = ys >= 0
            sel, ts, ys = sel[lab], ts[lab], ys[lab]
            if not len(sel):
                continue
            Xs = blob["X"][sel]
        mine = np.array([where.get((str(g[j]), int(t)), -1) for j, t in zip(sel, ts)], dtype=np.int64)
        hit = mine >= 0
        shard_only += int((~hit).sum())
        if not hit.any():
            continue
        mi = mine[hit]
        found[mi] = True
        label_mismatch += int((ys[hit].astype(np.int64) != y[mi].astype(np.int64)).sum())
        if Xs.shape[1] != X.shape[1]:
            return {"verdict": "column count differs", "shard_columns": int(Xs.shape[1]), "build_columns": int(X.shape[1]),
                    "identical": False}
        counts, mx, _, n_any = column_diff(Xs[hit], X[mi])
        col_counts += counts
        rows_differ += n_any
        max_abs = max(max_abs, float(mx.max()) if mx.size else 0.0)
        del Xs
    differing = np.flatnonzero(col_counts > 0)
    rec = {
        "shards": str(shard_dir), "label_key": label_key,
        "feature_names_equal": shard_names == names,
        "n_build_rows": int(len(keys)), "n_duplicate_keys": int(dup),
        "n_matched": int(found.sum()), "n_build_only": int((~found).sum()), "n_shard_only_labelled": int(shard_only),
        "n_label_mismatch": int(label_mismatch), "n_rows_X_differ": int(rows_differ),
        "n_columns_X_differ": int(len(differing)), "max_abs_X_diff": float(max_abs),
        "columns_X_differ": [names[j] for j in differing[:40]],
    }
    rec["identical"] = bool(rec["feature_names_equal"] and not dup and rec["n_build_only"] == 0 and shard_only == 0
                            and label_mismatch == 0 and rows_differ == 0)
    return rec


def base_layout(names: list[str]):
    """Bases in first-appearance order and each column's base index ("<base>__<statistic>"; no suffix = own base)."""
    bases: list[str] = []
    base_id: dict[str, int] = {}
    for nm in names:
        b = nm.rsplit("__", 1)[0]
        if b not in base_id:
            base_id[b] = len(bases); bases.append(b)
    return bases, base_id, np.array([base_id[nm.rsplit("__", 1)[0]] for nm in names], dtype=np.int64)


def derive_leak_columns(X, split_masks, rows_used, keys, names, bases, base_id, col_base, cfg, args,
                        global_names, spatial_names) -> dict:
    """Protocol step 5: which columns do the two switches change, and which switch changes each of them."""
    n_f = len(names)
    k = int(args.probe_per_column)
    total = np.zeros(n_f, dtype=np.int64)
    max_abs = np.zeros(n_f)
    per_split = {}
    examples = [[] for _ in range(n_f)]
    for split_name, mask in split_masks.items():
        idx = np.flatnonzero(mask)
        cnt, mx, ex, n_any = column_diff(X["clean"], X["leaky"], idx, want_examples=k)
        per_split[split_name] = {"n_rows": int(len(idx)), "n_rows_any_column_changed": int(n_any), "counts": cnt}
        total += cnt
        max_abs = np.maximum(max_abs, mx)
        if k:
            for j in np.flatnonzero(cnt):
                room = k - len(examples[j])
                if room > 0:
                    examples[j].extend(idx[np.asarray(ex[j][:room], dtype=np.int64)].tolist())

    n = len(rows_used)
    spread = np.unique(np.linspace(0, n - 1, num=min(int(args.probe_refs), n)).round().astype(np.int64))
    chosen = np.asarray([r for ex in examples for r in ex], dtype=np.int64)
    probe_rows = np.unique(np.concatenate([spread, chosen]))
    probe_refs = [rows_used[i] for i in probe_rows]
    probe_keys = [keys[i] for i in probe_rows]
    ref_clean = X["clean"][probe_rows]
    probe = {"n_rows": int(len(probe_rows)), "n_spread_rows": int(len(spread)), "rows_per_affected_column": k,
             "n_example_rows": int(len(np.unique(chosen)))}

    def rebuild(switches, label):
        Xp, _, used_p, _, secs = build_matrix(probe_refs, switches, cfg, len(probe_refs) + 1, label)
        if [(r.match_id, int(r.t_start_ts)) for r in used_p] != probe_keys:
            raise SystemExit(f"probe rows differ from the full build under {label}")
        cnt, _, _, _ = column_diff(ref_clean, Xp)
        return cnt, secs

    det, secs = rebuild(CONFIGS["clean"], "probe_clean")
    probe["determinism"] = {"n_columns_differ": int((det > 0).sum()), "seconds": round(secs, 1)}
    if (det > 0).any():
        raise SystemExit(f"clean build is not deterministic: {[names[j] for j in np.flatnonzero(det)[:20]]}")
    single = {}
    for sw_name, sw in SWITCH_PROBES.items():
        cnt, secs = rebuild(sw, f"probe_{sw_name}")
        single[sw_name] = cnt > 0
        probe[sw_name] = {"n_columns_changed": int(single[sw_name].sum()), "seconds": round(secs, 1)}
    cfg.TIME_NORM_ABSOLUTE, cfg.ANCHORS_CAUSAL = CONFIGS["clean"]
    both_probe, _, _, _ = column_diff(ref_clean, X["leaky"][probe_rows])

    affected = (total > 0) | single["TIME_NORM_ABSOLUTE"] | single["ANCHORS_CAUSAL"]
    col_switch: dict[int, list[str]] = {}
    for j in np.flatnonzero(affected):
        tags = [s for s in SWITCH_PROBES if single[s][j]]
        col_switch[int(j)] = tags if tags else ["interaction"]
    # an "interaction" column must still change on its own example rows when both switches flip
    unresolved = [names[j] for j, t in col_switch.items() if t == ["interaction"] and both_probe[j] == 0]

    exact = {g: np.array(sorted(j for j, t in col_switch.items() if g in t), dtype=np.int64) for g in LEAK_GROUPS}
    exact["union"] = np.flatnonzero(affected).astype(np.int64)
    group_bases = {g: sorted({bases[col_base[j]] for j in cols}) for g, cols in exact.items()}
    closed = {g: np.flatnonzero(np.isin(col_base, [base_id[b] for b in bs])).astype(np.int64)
              for g, bs in group_bases.items()}
    base_tags: dict[str, set] = {}
    for j, t in col_switch.items():
        base_tags.setdefault(bases[col_base[j]], set()).update(t)
    order = {g: i for i, g in enumerate(LEAK_GROUPS)}
    base_switch = {b: "+".join(sorted(t, key=order.get)) for b, t in base_tags.items()}

    test_only = []
    if "test" in per_split:
        other_counts = sum(v["counts"] for s, v in per_split.items() if s != "test")
        test_only = [names[j] for j in np.flatnonzero((per_split["test"]["counts"] > 0) & (other_counts == 0))]
    record = {
        "derivation": "a column is leak-affected when its value differs between the clean and leaky builds on any "
                      "sample row, or when a single-switch rebuild of the probe rows changes it; each column is "
                      "attributed by the single-switch rebuilds (interaction = changes only when both switches flip); "
                      "bases closed by feature name <base>__<statistic>",
        "per_split_rows": {s: {"n_rows": v["n_rows"], "n_rows_any_column_changed": v["n_rows_any_column_changed"]}
                           for s, v in per_split.items()},
        "probe": probe,
        "groups": {g: {"bases": group_bases[g], "n_bases": len(group_bases[g]),
                       "n_columns_base_closure": int(len(closed[g])), "n_columns_affected": int(len(exact[g]))}
                   for g in exact},
        "affected_columns": [{"feature": names[j], "leak_switch": "+".join(col_switch[j]),
                              "base_leak_switch": base_switch[bases[col_base[j]]],
                              "changed_in_leaky_vs_clean": bool(total[j] > 0),
                              "frac_rows_changed": {s: float(v["counts"][j] / max(v["n_rows"], 1)) for s, v in per_split.items()},
                              "max_abs_change": float(max_abs[j])} for j in sorted(col_switch)],
        "columns_changed_on_test_rows_only": test_only,
        "unresolved_columns": unresolved,
        "family_check": {
            "TIME_NORM_ABSOLUTE_bases_within_GLOBAL_FEATURE_NAMES": set(group_bases["TIME_NORM_ABSOLUTE"]) <= set(global_names),
            "ANCHORS_CAUSAL_bases_within_SPATIAL_FEATURE_NAMES": set(group_bases["ANCHORS_CAUSAL"]) <= set(spatial_names),
            "union_within_GLOBAL_or_SPATIAL": set(group_bases["union"]) <= (set(global_names) | set(spatial_names)),
            "spatial_bases_not_affected": sorted(set(spatial_names) & set(bases) - set(group_bases["union"])),
        },
    }
    return {"record": record, "exact": exact, "closed": closed, "group_bases": group_bases,
            "base_switch": base_switch, "col_switch": col_switch}


def ratio_bootstrap(groups, num: dict, den: dict, n_boot: int, seed: int) -> dict:
    """sum(num)/sum(den) per configuration, CI and paired difference, resampling matches."""
    uniq, inv = np.unique(groups, return_inverse=True)
    agg_n = {k: np.bincount(inv, weights=v, minlength=len(uniq)) for k, v in num.items()}
    agg_d = {k: np.bincount(inv, weights=v, minlength=len(uniq)) for k, v in den.items()}
    point = {k: float(num[k].sum() / max(den[k].sum(), 1e-300)) for k in num}
    out = {"share_clean": point["clean"], "share_leaky": point["leaky"],
           "diff_leaky_minus_clean": point["leaky"] - point["clean"],
           "n_boot": int(n_boot), "n_matches": int(len(uniq)),
           "method": "match-clustered percentile bootstrap over the explained test rows' matches, paired"}
    if n_boot <= 0:
        return {**out, "ci_clean": None, "ci_leaky": None, "ci_diff": None, "p_diff_leq_0": None}
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in num}
    for b in range(n_boot):
        c = np.bincount(rng.integers(0, len(uniq), size=len(uniq)), minlength=len(uniq))
        for k in num:
            draws[k][b] = float((c * agg_n[k]).sum() / max((c * agg_d[k]).sum(), 1e-300))
    d = draws["leaky"] - draws["clean"]
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {**out, "ci_clean": pct(draws["clean"]), "ci_leaky": pct(draws["leaky"]), "ci_diff": pct(d),
            "p_diff_leq_0": float((d <= 0).mean())}


def tree_shap(booster, X, rows, row_class, group_cols: dict, chunk: int, n_jobs: int) -> dict:
    """Path-dependent Tree SHAP via LightGBM pred_contrib, accumulated chunk by chunk."""
    n_f = X.shape[1]
    sum_abs, sum_signed = np.zeros(n_f), np.zeros(n_f)
    class_abs = {c: np.zeros(n_f) for c in CLASSES}
    class_n = {c: 0 for c in CLASSES}
    row_total = np.zeros(len(rows))
    row_group = {g: np.zeros(len(rows)) for g in group_cols}
    top1 = np.zeros(len(rows), dtype=np.int64)
    add_err, expected, t0 = 0.0, None, time.time()
    for s in range(0, len(rows), chunk):
        idx = rows[s:s + chunk]
        xb = X[idx]
        contrib = booster.predict(xb, pred_contrib=True, num_threads=n_jobs)
        margin = booster.predict(xb, raw_score=True, num_threads=n_jobs)
        add_err = max(add_err, float(np.abs(contrib.sum(axis=1) - margin).max()))
        expected = float(contrib[0, -1])
        phi = contrib[:, :-1]
        a = np.abs(phi)
        sum_abs += a.sum(axis=0); sum_signed += phi.sum(axis=0)
        e = s + len(idx)
        row_total[s:e] = a.sum(axis=1)
        for g, cols in group_cols.items():
            row_group[g][s:e] = a[:, cols].sum(axis=1) if len(cols) else 0.0
        top1[s:e] = a.argmax(axis=1)
        for c in CLASSES:
            m = row_class[s:e] == c
            if m.any():
                class_abs[c] += a[m].sum(axis=0); class_n[c] += int(m.sum())
        del contrib, phi, a
    n = len(rows)
    return {"mean_abs": sum_abs / n, "mean_signed": sum_signed / n, "row_total": row_total, "row_group": row_group,
            "top1": top1, "class_mean_abs": {c: class_abs[c] / class_n[c] for c in CLASSES if class_n[c]},
            "class_n": class_n, "additivity_max_abs_err": add_err, "expected_value": expected,
            "seconds": round(time.time() - t0, 1)}


# ---- SHAP from the definition, independent of LightGBM's implementation (protocol step 8) ----

def _count(node) -> float:
    return float(node["internal_count"] if "split_feature" in node else node["leaf_count"])


def _goes_left(node, fval: float) -> bool:
    """LightGBM's numerical decision (include/LightGBM/tree.h, NumericalDecision)."""
    if node.get("decision_type", "<=") != "<=":
        raise ValueError("categorical split: the corpus has no categorical columns")
    missing = node.get("missing_type", "None")
    if math.isnan(fval) and missing != "NaN":
        fval = 0.0
    if (missing == "Zero" and -LGBM_ZERO_THRESHOLD <= fval <= LGBM_ZERO_THRESHOLD) or (missing == "NaN" and math.isnan(fval)):
        return bool(node["default_left"])
    return fval <= float(node["threshold"])


def _tree_features(tree) -> set:
    out, stack = set(), [tree]
    while stack:
        node = stack.pop()
        if "split_feature" in node:
            out.add(int(node["split_feature"])); stack.extend((node["left_child"], node["right_child"]))
    return out


def shap_by_leaf_decomposition(tree, x) -> tuple[dict, float]:
    """Exact path-dependent SHAP values of one tree at x, from the definition (see module docstring)."""
    if "split_feature" not in tree:
        return {}, float(tree["leaf_value"])
    phi: dict[int, float] = {}
    expected = 0.0
    stack = [(tree, {}, {})]
    while stack:
        node, A, B = stack.pop()
        if "split_feature" in node:
            d = int(node["split_feature"])
            left = _goes_left(node, float(x[d]))
            c = _count(node)
            for side, is_left in (("left_child", True), ("right_child", False)):
                child = node[side]
                A2, B2 = dict(A), dict(B)
                A2[d] = A2.get(d, 1.0) * (1.0 if left == is_left else 0.0)
                B2[d] = B2.get(d, 1.0) * (_count(child) / c)
                stack.append((child, A2, B2))
            continue
        v = float(node["leaf_value"])
        feats = list(A)
        m = len(feats)
        expected += v * float(np.prod([B[d] for d in feats])) if m else v
        if not m:
            continue
        lin = [np.array([B[d], A[d]]) for d in feats]           # B + A z: coefficient of z^1 = "d in S"
        prefix = [np.array([1.0])]
        for p in lin[:-1]:
            prefix.append(np.convolve(prefix[-1], p))
        suffix = [np.array([1.0])]
        for p in lin[:0:-1]:
            suffix.append(np.convolve(suffix[-1], p))
        w = np.array([1.0 / (m * math.comb(m - 1, kk)) for kk in range(m)])   # k!(m-k-1)!/m!
        for i, d in enumerate(feats):
            e = np.convolve(prefix[i], suffix[m - 1 - i])        # prod over the other m-1 features
            phi[d] = phi.get(d, 0.0) + v * (A[d] - B[d]) * float(e @ w)
    return phi, expected


def shap_by_enumeration(tree, x) -> tuple[dict, float]:
    """EXPVALUE (Lundberg, Erion & Lee 2018, Algorithm 1) for every subset of the tree's features + the Shapley sum."""
    feats = sorted(_tree_features(tree))
    m = len(feats)

    def G(node, S):
        if "split_feature" not in node:
            return float(node["leaf_value"])
        d = int(node["split_feature"])
        if d in S:
            return G(node["left_child"] if _goes_left(node, float(x[d])) else node["right_child"], S)
        return (G(node["left_child"], S) * _count(node["left_child"])
                + G(node["right_child"], S) * _count(node["right_child"])) / _count(node)

    f = np.array([G(tree, frozenset(feats[i] for i in range(m) if mask >> i & 1)) for mask in range(1 << m)])
    phi = {}
    for i, d in enumerate(feats):
        total = 0.0
        for mask in range(1 << m):
            if not mask >> i & 1:
                total += (f[mask | (1 << i)] - f[mask]) / (m * math.comb(m - 1, bin(mask).count("1")))
        phi[d] = total
    return phi, float(f[0])


def _dense(phi: dict, n_f: int) -> np.ndarray:
    out = np.zeros(n_f)
    for d, v in phi.items():
        out[d] = v
    return out


def verify_tree_shap(booster, X, rows, n_trees: int, enum_max: int, n_full_rows: int, n_jobs: int) -> dict:
    """Per-tree pred_contrib against the definition on sampled trees; the whole ensemble on n_full_rows rows."""
    t0 = time.time()
    dump = booster.dump_model()["tree_info"]
    T, n_f = len(dump), X.shape[1]
    feats = [_tree_features(d["tree_structure"]) for d in dump]
    ids = set(np.unique(np.linspace(0, T - 1, num=min(n_trees, T)).round().astype(int)).tolist()) if n_trees > 0 else set()
    ids |= {int(t) for t in np.argsort([len(f) for f in feats], kind="stable")[:4] if len(feats[t]) <= enum_max}
    Xr = np.ascontiguousarray(X[np.asarray(rows)], dtype=np.float32)
    err = {"leaf_decomposition_phi": 0.0, "leaf_decomposition_expected": 0.0, "enumeration": 0.0,
           "enumeration_vs_leaf_decomposition": 0.0}
    n_dec = n_enum = 0
    for t in sorted(ids):
        tree = dump[t]["tree_structure"]
        ref = booster.predict(Xr, pred_contrib=True, start_iteration=int(t), num_iteration=1, num_threads=n_jobs)
        for i in range(len(Xr)):
            phi, ev = shap_by_leaf_decomposition(tree, Xr[i])
            dense = _dense(phi, n_f)
            err["leaf_decomposition_phi"] = max(err["leaf_decomposition_phi"], float(np.abs(dense - ref[i, :-1]).max()))
            err["leaf_decomposition_expected"] = max(err["leaf_decomposition_expected"], abs(ev - float(ref[i, -1])))
            n_dec += 1
            if len(feats[t]) <= enum_max:
                phi_e, ev_e = shap_by_enumeration(tree, Xr[i])
                dense_e = _dense(phi_e, n_f)
                err["enumeration"] = max(err["enumeration"], float(np.abs(dense_e - ref[i, :-1]).max()),
                                         abs(ev_e - float(ref[i, -1])))
                err["enumeration_vs_leaf_decomposition"] = max(err["enumeration_vs_leaf_decomposition"],
                                                               float(np.abs(dense_e - dense).max()))
                n_enum += 1
    full = {"rows": 0}
    if n_full_rows > 0:
        Xf = Xr[:n_full_rows]
        ref_full = booster.predict(Xf, pred_contrib=True, num_threads=n_jobs)
        worst = 0.0
        for i in range(len(Xf)):
            acc = np.zeros(n_f + 1)
            for t in range(T):
                phi, ev = shap_by_leaf_decomposition(dump[t]["tree_structure"], Xf[i])
                for d, v in phi.items():
                    acc[d] += v
                acc[-1] += ev
            worst = max(worst, float(np.abs(acc - ref_full[i]).max()))
        full = {"rows": int(len(Xf)), "trees": T, "max_abs_diff_phi_and_expected": worst,
                "max_abs_phi": float(np.abs(ref_full[:, :-1]).max())}
    all_errs = [err["leaf_decomposition_phi"], err["leaf_decomposition_expected"]] + \
        ([err["enumeration"], err["enumeration_vs_leaf_decomposition"]] if n_enum else []) + \
        ([full["max_abs_diff_phi_and_expected"]] if full["rows"] else [])
    return {"n_trees_in_model": T, "trees_checked": sorted(ids), "rows_per_tree": int(len(Xr)),
            "tree_split_features": {"min": int(min(map(len, feats))) if T else 0, "max": int(max(map(len, feats))) if T else 0},
            "n_tree_row_checks_leaf_decomposition": n_dec, "n_tree_row_checks_enumeration": n_enum,
            "enumeration_max_tree_features": int(enum_max), "max_abs_diff": err, "whole_ensemble": full,
            "tolerance": VERIFY_TOLERANCE, "agrees": bool(max(all_errs) <= VERIFY_TOLERANCE),
            "seconds": round(time.time() - t0, 1)}


def shap_package_check(booster, X_rows: np.ndarray, n_jobs: int) -> dict:
    """Equality check only: shap dispatches path-dependent LightGBM explanations to pred_contrib."""
    try:
        import shap
        phi = booster.predict(X_rows, pred_contrib=True, num_threads=n_jobs)[:, :-1]
        explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(X_rows)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv, dtype=np.float64)
        if sv.ndim == 3:
            sv = sv[..., -1]
        return {"shap_version": shap.__version__, "rows": int(len(X_rows)),
                "max_abs_diff_vs_pred_contrib": float(np.abs(sv - phi).max()),
                "independent": False,
                "note": "shap.TreeExplainer(tree_path_dependent, no background data) calls LightGBM pred_contrib itself"}
    except Exception as e:  # a cross-check only; the primary values come from LightGBM
        return {"rows": int(len(X_rows)), "error": repr(e)}


def attribution_report(names, bases, col_base, mean_abs, closed: dict, exact: dict, base_switch: dict, top_k: int):
    from scipy.stats import rankdata
    total = float(mean_abs.sum())
    base_vals = np.bincount(col_base, weights=mean_abs, minlength=len(bases))
    base_rank = rankdata(-base_vals, method="min").astype(int)
    col_rank = rankdata(-mean_abs, method="min").astype(int)

    def share(v):
        return float(v) / total if total > 0 else None

    rep = {
        "total_mean_abs_shap": total,
        "rank_convention": "rank 1 = largest mean |SHAP|; ties share the smallest rank, so every zero-attribution "
                           "base has rank n_bases_nonzero + 1",
        "n_columns_nonzero": int((mean_abs > 0).sum()), "n_bases_nonzero": int((base_vals > 0).sum()),
        "top_bases": [{"rank": int(base_rank[b]), "base": bases[b], "mean_abs_shap": float(base_vals[b]),
                       "share": share(base_vals[b]), "leak_switch": base_switch.get(bases[b])}
                      for b in np.argsort(-base_vals, kind="stable")[:top_k]],
        "top_columns": [{"rank": int(col_rank[j]), "feature": names[j], "mean_abs_shap": float(mean_abs[j]),
                         "share": share(mean_abs[j]), "leak_switch": base_switch.get(bases[col_base[j]])}
                        for j in np.argsort(-mean_abs, kind="stable")[:top_k]],
        "leak": {},
    }
    for g, cols in closed.items():
        ids = np.unique(col_base[cols]) if len(cols) else np.array([], dtype=np.int64)
        v = float(mean_abs[cols].sum()) if len(cols) else 0.0
        ex = exact[g]
        v_ex = float(mean_abs[ex].sum()) if len(ex) else 0.0
        rep["leak"][g] = {
            "n_bases": int(len(ids)), "n_columns": int(len(cols)), "mean_abs_shap": v, "share": share(v),
            "best_base_rank": int(base_rank[ids].min()) if len(ids) else None,
            "n_bases_in_top_k": int((base_rank[ids] <= top_k).sum()) if len(ids) else 0,
            "best_column_rank": int(col_rank[cols].min()) if len(cols) else None,
            "bases": sorted(({"base": bases[i], "rank": int(base_rank[i]), "mean_abs_shap": float(base_vals[i]),
                              "share": share(base_vals[i])} for i in ids), key=lambda d: d["rank"]),
            "affected_columns_only": {"n_columns": int(len(ex)), "mean_abs_shap": v_ex, "share": share(v_ex),
                                      "best_column_rank": int(col_rank[ex].min()) if len(ex) else None},
        }
    return rep, base_vals, base_rank


def write_csvs(stem: Path, names, mean_abs, bases, col_base, base_vals, base_rank, base_switch) -> dict:
    col_path = stem.with_name(stem.name + "_mean_abs_shap.csv")
    base_path = stem.with_name(stem.name + "_base.csv")
    with open(col_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean_abs_shap"])       # analysis/shap_role_rollup.py input format
        for nm, v in zip(names, mean_abs):
            w.writerow([nm, repr(float(v))])
    counts = np.bincount(col_base, minlength=len(bases))
    total = float(base_vals.sum())
    with open(base_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "base", "mean_abs_shap", "share", "n_columns", "leak_switch"])
        for b in np.argsort(-base_vals, kind="stable"):
            w.writerow([int(base_rank[b]), bases[b], repr(float(base_vals[b])),
                        repr(float(base_vals[b] / total)) if total > 0 else "", int(counts[b]), base_switch.get(bases[b], "")])
    return {"columns_csv": str(col_path), "bases_csv": str(base_path)}


def role_rollup(csv_path: str) -> dict:
    from analysis.shap_role_rollup import ROLES, SIGNALS, rollup_one
    role_shap, role_sig, role_total, team_level, total = rollup_one(csv_path)
    roles = {}
    for r in ROLES:
        rt = role_shap[r]
        roles[r] = {"pct": 100.0 * rt / role_total if role_total else 0.0,
                    **{s: (100.0 * role_sig[r][s] / rt if rt else 0.0) for s in SIGNALS}}
    return {"roles": roles, "team_level_pct": 100.0 * team_level / total if total else 0.0,
            "note": "role % sums to 100 over slotted columns; team-level (non-slotted) columns, which include every "
                    "leak base, are excluded there and reported as team_level_pct"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS,
                    help="v3.3 shards: source of the declared match sample and of the clean-row cross-check")
    ap.add_argument("--n-matches", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=SEED, help="match draw, SHAP row sample, bootstrap and LightGBM")
    ap.add_argument("--preset", default="v3.3")
    ap.add_argument("--label", choices=LABELS, default="market_event",
                    help="label the models fit (draws dropped): market_event = the v3.3 primary label; market_lex = the "
                         "label of the retracted v3-era SHAP artefact (a robustness column in v3.3)")
    ap.add_argument("--train-patch", default="15.14")
    ap.add_argument("--val-patch", default="15.15")
    ap.add_argument("--test-patch", default="15.16")
    ap.add_argument("--teamfight-min", type=int, default=4, help="participation cut for the scale breakdown (v3: 4)")
    ap.add_argument("--standardize", choices=("train", "none"), default="train",
                    help="train: run_deep_tabular_baselines.prepare_matrix before LightGBM, as the learner comparison")
    ap.add_argument("--shap-rows", type=int, default=10000, help="declared seeded sample of test rows to explain")
    ap.add_argument("--shap-chunk", type=int, default=1000)
    ap.add_argument("--verify-trees", type=int, default=16, help="trees per explained model checked against the definition")
    ap.add_argument("--verify-rows", type=int, default=3, help="explained rows used in the per-tree definition check")
    ap.add_argument("--verify-full-rows", type=int, default=1, help="rows checked over every tree of the explained models")
    ap.add_argument("--verify-enum-max-features", type=int, default=10,
                    help="largest tree (distinct split features) checked by full subset enumeration")
    ap.add_argument("--verify-aux-rows", type=int, default=4000,
                    help="train rows for the small auxiliary model whose every tree is checked by enumeration (0 = skip)")
    ap.add_argument("--verify-shap-rows", type=int, default=200,
                    help="rows compared with shap.TreeExplainer (an equality check of the same routine; 0 = skip)")
    ap.add_argument("--probe-refs", type=int, default=300, help="evenly spread sample rows rebuilt per single switch")
    ap.add_argument("--probe-per-column", type=int, default=3,
                    help="rows per affected column (where it changed) added to the single-switch probe")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=8, help="LightGBM threads (fit and pred_contrib)")
    ap.add_argument("--index-workers", type=int, default=1, help="fight-index processes (0 = auto)")
    ap.add_argument("--build-chunk", type=int, default=4000)
    ap.add_argument("--skip-shard-check", action="store_true")
    ap.add_argument("--allow-shard-mismatch", action="store_true",
                    help="continue when the clean build does not reproduce the shard rows (default: stop)")
    ap.add_argument("--legacy-shap", type=Path, default=DEFAULT_LEGACY_SHAP,
                    help="retracted v3-era attribution whose top bases are located in this run")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--tag", default=None, help="output file stem (default shap_leak_forensics_v33_m<n>)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    started = time.time()
    stages: dict[str, float] = {}
    label_key = f"y_{args.label}"
    tag = args.tag or (f"shap_leak_forensics_v33_m{args.n_matches}" if args.label == "market_event"
                       else f"shap_leak_forensics_v33_{args.label}_m{args.n_matches}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"{tag}.json"
    if out_json.exists() and not args.overwrite:
        raise SystemExit(f"{out_json} exists; pass --overwrite or another --tag")

    cfg, preset_values = configure(args.preset, args.index_workers)
    from core.config import GLOBAL_FEATURE_NAMES
    from data.index_split import build_fight_index
    from gameplay.feature_spatial import SPATIAL_FEATURE_NAMES
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    dtb = load_module("run_deep_tabular_baselines")
    sd = load_module("run_scale_decomposition")

    # 1. declared match sample
    t = time.time()
    mids, n_available = sample_matches(args.shards, args.n_matches, args.seed)
    sample_sha1 = hashlib.sha1("\n".join(mids).encode("utf-8")).hexdigest()
    stages["sample"] = time.time() - t
    print(f"matches available={n_available} sampled={len(mids)} sha1={sample_sha1[:12]}", flush=True)

    # 2. fight index and the label (draws dropped)
    t = time.time()
    refs = build_fight_index(cache_match_ids=mids)
    stages["fight_index"] = time.time() - t
    print(f"refs={len(refs)} ({stages['fight_index']:.0f}s)", flush=True)
    t = time.time()
    y_all = drop_policy_labels(refs, cfg, args.label)
    stages["label"] = time.time() - t
    keep = y_all >= 0
    refs_kept = [r for r, k in zip(refs, keep) if k]
    y_by_ref = {id(r): int(v) for r, v in zip(refs, y_all)}
    print(f"{label_key}: {int(keep.sum())} labelled, {int((~keep).sum())} draws dropped ({stages['label']:.0f}s)", flush=True)

    # 3. the two builds on the same engagements
    X, names, used, row_label, build_s = {}, {}, {}, {}, {}
    for name, sw in CONFIGS.items():
        print(f"[{name}] TIME_NORM_ABSOLUTE={sw[0]} ANCHORS_CAUSAL={sw[1]}", flush=True)
        X[name], names[name], used[name], row_label[name], build_s[name] = build_matrix(
            refs_kept, sw, cfg, args.build_chunk, name)
        stages[f"build_{name}"] = build_s[name]
        print(f"[{name}] X={X[name].shape} ({build_s[name]:.0f}s)", flush=True)
    cfg.TIME_NORM_ABSOLUTE, cfg.ANCHORS_CAUSAL = CONFIGS["clean"]
    keys = {n: [(r.match_id, int(r.t_start_ts)) for r in used[n]] for n in CONFIGS}
    identity = {
        "n_rows_clean": len(keys["clean"]), "n_rows_leaky": len(keys["leaky"]),
        "keys_equal": keys["clean"] == keys["leaky"],
        "same_ref_objects": len(used["clean"]) == len(used["leaky"]) and all(a is b for a, b in zip(used["clean"], used["leaky"])),
        "feature_names_equal": names["clean"] == names["leaky"],
        "row_label_equal": bool(np.array_equal(row_label["clean"], row_label["leaky"])),
        "keys_sha1": hashlib.sha1(json.dumps(keys["clean"]).encode("utf-8")).hexdigest(),
        "n_refs_labelled": int(len(refs_kept)), "n_refs_not_built": int(len(refs_kept) - len(keys["clean"])),
    }
    if not (identity["keys_equal"] and identity["same_ref_objects"] and identity["feature_names_equal"] and identity["row_label_equal"]):
        raise SystemExit(f"row identity failed between clean and leaky builds: {identity}")
    rows_used = used["clean"]
    names = names["clean"]
    y = np.array([y_by_ref[id(r)] for r in rows_used], dtype=np.int8)
    ok = (row_label["clean"] >= 0)
    identity[f"{args.label}_agrees_with_row_label"] = float((row_label["clean"][ok] == y[ok]).mean()) if ok.any() else None
    groups = np.array([r.match_id for r in rows_used])
    patch = np.array([str(r.patch) for r in rows_used])
    engage_ts = np.array([int(r.t_start_ts) for r in rows_used], dtype=np.int64)
    scale = np.asarray(sd.scale_class(np.array([r.det_cluster_blue for r in rows_used]),
                                      np.array([r.det_cluster_red for r in rows_used]),
                                      teamfight_min=args.teamfight_min)).astype(str)
    print(f"row identity ok: {len(y)} rows, {len(np.unique(groups))} matches, positives={y.mean():.3f}", flush=True)

    # 4. clean build vs the published shard rows
    if args.skip_shard_check:
        crosscheck = {"skipped": True}
    else:
        t = time.time()
        crosscheck = shard_crosscheck(args.shards, keys["clean"], X["clean"], y, names, label_key)
        stages["shard_crosscheck"] = time.time() - t
        print(f"shard cross-check: identical={crosscheck.get('identical')} matched={crosscheck.get('n_matched')}"
              f" build_only={crosscheck.get('n_build_only')} shard_only={crosscheck.get('n_shard_only_labelled')}"
              f" X_rows_differ={crosscheck.get('n_rows_X_differ')} ({stages['shard_crosscheck']:.0f}s)", flush=True)
        if not crosscheck.get("identical") and not args.allow_shard_mismatch:
            raise SystemExit(f"the clean build does not reproduce the v3.3 shard rows (shared code changed?): {crosscheck}")

    # patch holdout (the learner comparison's split helper)
    tr, va, te = dtb.split_by_patch(patch, args.train_patch, args.val_patch, args.test_patch)
    split = {"kind": "patch", "helper": "run_deep_tabular_baselines.split_by_patch",
             "train_patch": args.train_patch, "val_patch": args.val_patch, "test_patch": args.test_patch}
    for nm_, m in (("train", tr), ("val", va), ("test", te)):
        split[nm_] = {"rows": int(m.sum()), "matches": int(len(np.unique(groups[m]))), "positive_rate": float(y[m].mean())}
    other = ~(tr | va | te)
    split["rows_in_other_patches"] = int(other.sum())
    split["matches_shared_between_splits"] = int(len(set(groups[tr]) & set(groups[te])) + len(set(groups[va]) & set(groups[te]))
                                                 + len(set(groups[tr]) & set(groups[va])))
    print(f"split rows: train={tr.sum()} val={va.sum()} test={te.sum()} other={other.sum()}", flush=True)

    # 5. leak columns, derived from the switches
    t = time.time()
    bases, base_id, col_base = base_layout(names)
    split_masks = {"train": tr, "val": va, "test": te}
    if other.any():
        split_masks["other"] = other
    leak = derive_leak_columns(X, split_masks, rows_used, keys["clean"], names, bases, base_id, col_base, cfg, args,
                               GLOBAL_FEATURE_NAMES, SPATIAL_FEATURE_NAMES)
    group_bases, base_switch = leak["group_bases"], leak["base_switch"]
    closed = {g: c for g, c in leak["closed"].items() if g == "union" or len(c)}
    exact = {g: leak["exact"][g] for g in closed}
    stages["leak_derivation"] = time.time() - t
    for g in closed:
        print(f"leak bases {g}: {group_bases[g]}", flush=True)
    if leak["record"]["unresolved_columns"]:
        print(f"WARNING unresolved leak columns: {leak['record']['unresolved_columns']}", flush=True)

    # 6. LightGBM on the patch holdout, one model per configuration
    if args.standardize == "train":
        t = time.time()
        for name in CONFIGS:
            dtb.prepare_matrix({"X": X[name]}, tr)          # in place, train-row mean / std
        stages["standardize"] = time.time() - t
    lgbm_config = dict(dtb.LGBM_PUBLISHED)
    models, boosters, preds = {}, {}, {}
    for name in CONFIGS:
        t = time.time()
        model = lgb.LGBMClassifier(**lgbm_config, random_state=args.seed, n_jobs=args.n_jobs, verbose=-1)
        Xtr, Xva = X[name][tr], X[name][va]
        model.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        nonconstant = int((Xtr.max(axis=0) > Xtr.min(axis=0)).sum())
        del Xtr
        best = int(model.best_iteration_ or model.n_estimators)
        # explain exactly the trees behind the reported AUC
        booster = lgb.Booster(model_str=model.booster_.model_to_string(num_iteration=best))
        p_va = booster.predict(Xva, num_threads=args.n_jobs)
        with warnings.catch_warnings():   # sklearn's feature-name notice for an ndarray input; values are unaffected
            warnings.simplefilter("ignore", UserWarning)
            trunc = float(np.abs(p_va[:2000] - model.predict_proba(Xva[:2000], num_iteration=best)[:, 1]).max())
        del Xva
        preds[name] = booster.predict(X[name][te], num_threads=args.n_jobs)
        models[name] = {"best_iteration": best, "trees_in_explained_model": int(booster.num_trees()),
                        "truncated_booster_max_abs_diff_val": trunc,
                        "val_auc": float(roc_auc_score(y[va], p_va)),
                        "test_auc": float(roc_auc_score(y[te], preds[name])),
                        "n_nonconstant_train_columns": nonconstant,
                        "fit_seconds": round(time.time() - t, 1)}
        boosters[name] = booster
        del model
        stages[f"fit_{name}"] = time.time() - t
        print(f"[{name}] best_iter={best} val={models[name]['val_auc']:.4f} test={models[name]['test_auc']:.4f}"
              f" ({models[name]['fit_seconds']}s)", flush=True)

    t = time.time()
    y_te, g_te, s_te = y[te], groups[te], scale[te]
    masks = {"overall": np.ones(len(y_te), dtype=bool)}
    for c in CLASSES:
        m = s_te == c
        if m.sum() >= 50 and len(np.unique(y_te[m])) == 2:
            masks[c] = m
    test_auc = {}
    for c, m in masks.items():
        rec = {"n_rows": int(m.sum()), "n_matches": int(len(np.unique(g_te[m]))), "positive_rate": float(y_te[m].mean()),
               "auc_clean": float(roc_auc_score(y_te[m], preds["clean"][m])),
               "auc_leaky": float(roc_auc_score(y_te[m], preds["leaky"][m]))}
        rec["diff_leaky_minus_clean"] = rec["auc_leaky"] - rec["auc_clean"]
        if args.n_boot > 0:
            b = dtb.cluster_bootstrap_auc(y_te[m], {"clean": preds["clean"][m], "leaky": preds["leaky"][m]}, g_te[m],
                                          n_boot=args.n_boot, seed=args.seed, pairs=[("leaky", "clean")])
            rec.update({"ci_clean": b["auc"]["clean"]["ci95"], "ci_leaky": b["auc"]["leaky"]["ci95"],
                        "ci_diff": b["paired"]["leaky-clean"]["ci95"], "n_boot": b["n_boot"], "method": b["method"]})
        test_auc[c] = rec
    stages["auc_bootstrap"] = time.time() - t
    o = test_auc["overall"]
    print(f"test AUC clean {o['auc_clean']:.4f} {o.get('ci_clean')} leaky {o['auc_leaky']:.4f} {o.get('ci_leaky')}"
          f" diff {o['diff_leaky_minus_clean']:+.4f} {o.get('ci_diff')}", flush=True)

    # 7. TreeSHAP on one declared test-row sample, identical for both configurations
    test_idx = np.flatnonzero(te)
    rng = np.random.default_rng(args.seed)
    shap_rows = np.sort(rng.choice(test_idx, size=min(args.shap_rows, len(test_idx)), replace=False))
    shap_class = scale[shap_rows]
    shap_groups = groups[shap_rows]
    track = {**{g: closed[g] for g in closed}, **{f"{g}[affected columns]": exact[g] for g in closed}}
    shap_res, reports, base_vals, base_rank, csvs, rollups, pkg = {}, {}, {}, {}, {}, {}, {}
    leak_union = np.zeros(len(names), dtype=bool); leak_union[closed["union"]] = True
    for name in CONFIGS:
        t = time.time()
        res = tree_shap(boosters[name], X[name], shap_rows, shap_class, track, args.shap_chunk, args.n_jobs)
        shap_res[name] = res
        reports[name], base_vals[name], base_rank[name] = attribution_report(
            names, bases, col_base, res["mean_abs"], closed, exact, base_switch, args.top_k)
        reports[name]["by_scale_leak_share"] = {
            c: {"n_rows": int(res["class_n"][c]),
                **{g: (float(res["class_mean_abs"][c][cols].sum() / res["class_mean_abs"][c].sum())
                       if res["class_mean_abs"][c].sum() > 0 else None) for g, cols in track.items()}}
            for c in res["class_mean_abs"]}
        reports[name]["top1_column_is_leak_frac"] = float(leak_union[res["top1"]].mean())
        csvs[name] = write_csvs(args.output_dir / f"{tag}_{name}", names, res["mean_abs"], bases, col_base,
                                base_vals[name], base_rank[name], base_switch)
        try:
            rollups[name] = role_rollup(csvs[name]["columns_csv"])
        except Exception as e:
            rollups[name] = {"error": repr(e)}
        stages[f"shap_{name}"] = time.time() - t
        top = reports[name]["top_bases"][:5]
        print(f"[{name}] SHAP {res['seconds']}s additivity_err={res['additivity_max_abs_err']:.2e} "
              f"leak share={reports[name]['leak']['union']['share']} top5={[(d['base'], round(d['share'] or 0, 3)) for d in top]}",
              flush=True)

    t = time.time()
    leak_share = {g: ratio_bootstrap(shap_groups, {n: shap_res[n]["row_group"][g] for n in CONFIGS},
                                     {n: shap_res[n]["row_total"] for n in CONFIGS}, args.n_boot, args.seed)
                  for g in track}
    stages["leak_share_bootstrap"] = time.time() - t

    # 8. SHAP values against the definition
    t = time.time()
    verification = {"tolerance": VERIFY_TOLERANCE}
    rows_v = shap_rows[:max(args.verify_rows, args.verify_full_rows)]
    for name in CONFIGS:
        try:   # a failed check is recorded, never allowed to discard the fitted run
            verification[name] = verify_tree_shap(boosters[name], X[name], rows_v[:args.verify_rows], args.verify_trees,
                                                  args.verify_enum_max_features, 0, args.n_jobs)
            if args.verify_full_rows > 0:
                verification[name]["whole_ensemble"] = verify_tree_shap(
                    boosters[name], X[name], rows_v[:args.verify_full_rows], 0, -1, args.verify_full_rows, args.n_jobs)["whole_ensemble"]
                wf = verification[name]["whole_ensemble"]
                verification[name]["agrees"] = bool(verification[name]["agrees"]
                                                    and wf["max_abs_diff_phi_and_expected"] <= VERIFY_TOLERANCE)
        except Exception as e:
            verification[name] = {"error": repr(e), "agrees": False}
        verification[name]["shap_package"] = (shap_package_check(boosters[name], X[name][shap_rows[:args.verify_shap_rows]], args.n_jobs)
                                              if args.verify_shap_rows > 0 else {"skipped": True})
        print(f"[{name}] definition check agrees={verification[name]['agrees']} "
              f"max_abs_diff={verification[name].get('max_abs_diff', verification[name].get('error'))} "
              f"whole={verification[name].get('whole_ensemble')}", flush=True)
    aux_rows = np.flatnonzero(tr)[:max(args.verify_aux_rows, 0)]
    if len(aux_rows) >= 40 and len(np.unique(y[aux_rows])) == 2:
        aux_params = {"objective": "binary", "num_leaves": 6, "learning_rate": 0.1, "min_data_in_leaf": 10,
                      "feature_pre_filter": False, "verbose": -1, "num_threads": args.n_jobs, "seed": args.seed}
        try:
            aux = lgb.train(aux_params, lgb.Dataset(np.ascontiguousarray(X["clean"][aux_rows]), label=y[aux_rows]),
                            num_boost_round=8)
            verification["auxiliary_small_model"] = {
                "purpose": "every tree small enough for full subset enumeration; fitted on train rows of the clean matrix only",
                "params": aux_params, "n_train_rows": int(len(aux_rows)), "rounds": 8,
                **verify_tree_shap(aux, X["clean"], rows_v[:args.verify_rows], 8, args.verify_enum_max_features,
                                   len(rows_v[:args.verify_rows]), args.n_jobs)}
            del aux
        except Exception as e:
            verification["auxiliary_small_model"] = {"error": repr(e), "agrees": False}
    else:
        verification["auxiliary_small_model"] = {"skipped": True}
    verification["all_agree"] = bool(all(v.get("agrees", True) for k_, v in verification.items() if isinstance(v, dict)))
    stages["shap_verification"] = time.time() - t
    print(f"SHAP definition check all_agree={verification['all_agree']} ({stages['shap_verification']:.0f}s)", flush=True)

    from scipy.stats import spearmanr
    union_top = set(np.argsort(-base_vals["clean"], kind="stable")[:args.top_k].tolist()) | \
        set(np.argsort(-base_vals["leaky"], kind="stable")[:args.top_k].tolist())
    tot = {n: float(base_vals[n].sum()) for n in CONFIGS}

    def shift_row(b):
        return {"base": bases[b], "rank_clean": int(base_rank["clean"][b]), "rank_leaky": int(base_rank["leaky"][b]),
                "rank_gain_in_leaky": int(base_rank["clean"][b] - base_rank["leaky"][b]),
                "share_clean": float(base_vals["clean"][b] / tot["clean"]) if tot["clean"] else None,
                "share_leaky": float(base_vals["leaky"][b] / tot["leaky"]) if tot["leaky"] else None,
                "leak_switch": base_switch.get(bases[b])}

    nz = (base_vals["clean"] > 0) | (base_vals["leaky"] > 0)
    non_leak = nz & ~np.isin(np.arange(len(bases)), [base_id[b] for b in group_bases["union"]])
    top_c = set(np.argsort(-base_vals["clean"], kind="stable")[:args.top_k].tolist())
    top_l = set(np.argsort(-base_vals["leaky"], kind="stable")[:args.top_k].tolist())
    rank_shift = {
        "definition": "rank 1 = largest base mean |SHAP|; rank_gain_in_leaky = rank_clean - rank_leaky (positive = climbs under the leak)",
        "top_k_union": sorted((shift_row(b) for b in union_top), key=lambda d: d["rank_leaky"]),
        "leak_bases": sorted((shift_row(base_id[b]) for b in group_bases["union"]), key=lambda d: d["rank_leaky"]),
        "spearman_bases_nonzero": float(spearmanr(base_vals["clean"][nz], base_vals["leaky"][nz])[0]) if nz.sum() > 2 else None,
        "spearman_non_leak_bases_nonzero": float(spearmanr(base_vals["clean"][non_leak], base_vals["leaky"][non_leak])[0]) if non_leak.sum() > 2 else None,
        "top_k_jaccard": float(len(top_c & top_l) / max(len(top_c | top_l), 1)),
    }

    legacy = {"path": str(args.legacy_shap), "available": False}
    if args.legacy_shap and args.legacy_shap.exists():
        try:
            d = json.loads(args.legacy_shap.read_text(encoding="utf-8"))
            legacy = {"path": str(args.legacy_shap), "available": True,
                      "note": "v3-era corpus, market_lex label, leaky feature path, name-order fix applied; ranks there vs ranks here",
                      "top_bases": [{"legacy_rank": i, "base": b, "legacy_mean_abs_shap": float(v),
                                     "derived_leak_switch": base_switch.get(b),
                                     "rank_clean": int(base_rank["clean"][base_id[b]]) if b in base_id else None,
                                     "rank_leaky": int(base_rank["leaky"][base_id[b]]) if b in base_id else None}
                                    for i, (b, v) in enumerate(d.get("base_top", [])[:10], 1)]}
        except Exception as e:
            legacy = {"path": str(args.legacy_shap), "available": True, "error": repr(e)}

    preds_path = args.output_dir / f"{tag}.preds.npz"
    tmp_preds = preds_path.with_name(preds_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_preds,
        y_test=y_te, groups_test=g_te, engage_ts_test=engage_ts[te], scale_test=s_te,
        pred_clean=preds["clean"], pred_leaky=preds["leaky"],
        shap_rows_test_position=np.searchsorted(test_idx, shap_rows),
    )
    os.replace(tmp_preds, preds_path)

    deviations = [
        "SHAP values are path-dependent Tree SHAP on LightGBM's raw margin (log-odds): E[f(x)|x_S] is estimated from the "
        "trees' training cover. The interventional variant also described by Lundberg et al. (2020) is not used. With "
        "correlated inputs (time_norm moves with gold, level and item columns) attribution follows the tree paths, so a "
        "share describes this model, not a causal effect; the paired AUC difference is the causal ablation.",
        "Global importance is mean |SHAP| over a declared seeded sample of test-patch rows (--shap-rows), not every row.",
        "One model per configuration (one seed), not a multi-seed average as in the CoG Table VI. Leak-share CIs resample "
        "explained test matches and do not include refitting variance.",
        "Both matrices are built in-process from the source cache for a match subsample, because the leaky matrix exists "
        "nowhere else. The clean build is cross-checked row by row against the published v3.3 shards (shard_crosscheck).",
        "Protocol is the patch holdout of the learner comparison (train 15.14 / val 15.15 / test 15.16) on a "
        f"{args.n_matches}-match subsample, not the headline's match-grouped 5-fold OOF on the full corpus, so AUCs are "
        "neither the 0.6699 headline nor the full-corpus patch-holdout LightGBM.",
        "The explained model is the booster truncated to best_iteration (model_to_string(num_iteration=best)), so the "
        "probabilities, SHAP values and every check use exactly the trees behind the reported test AUC.",
        "Leak columns are derived empirically (value changes on sample rows when switches are flipped, all splits, "
        "label-free) and closed to whole bases by feature name. A column a switch could alter but did not alter on any "
        "sample row is not in the set.",
        "The role rollup applies analysis/shap_role_rollup.py's mapping to one CSV per configuration; there is no seed "
        "averaging, and the leak bases are all team-level (non-slotted), so they appear only in team_level_pct.",
        "Identifier columns are numeric inputs (no categorical treatment), as in the headline.",
        "shap is installed, but its TreeExplainer routes path-dependent LightGBM explanations to the same pred_contrib "
        "call, so the independent check is the from-definition computation on sampled trees and rows "
        "(shap_verification), not the shap package.",
    ]
    if args.standardize == "none":
        deviations.append("LightGBM is fitted on raw features; the learner comparison standardises with train-row "
                          "statistics first (a per-column affine map that can move histogram bin edges).")
    if len(shap_rows) >= len(test_idx):
        deviations[1] = "Global importance is mean |SHAP| over every test-patch row."
    if args.seed != SEED:
        deviations.append(f"seed {args.seed} instead of the declared 7")
    if args.label != "market_event":
        deviations.append(f"label {label_key} instead of the v3.3 primary y_market_event: {args.label} is the label of "
                          "the retracted v3-era artefact (features/shap_mlex.json) and a robustness column in v3.3; "
                          "the feature builds, split and learner are unchanged")

    results = {
        "item": "A7-shap-forensics",
        "purpose": "Leak forensics for the retraction: on identical engagements, does the leaky feature path put the model's "
                   "attribution on the leak columns, while the clean v3.3 path does not?",
        "reference": "Lundberg et al., Nature Machine Intelligence 2, 56-67 (2020): SHAP values, Tree SHAP (path-dependent), "
                     "mean |SHAP| global importance; EXPVALUE (Algorithm 1) of Lundberg, Erion & Lee, arXiv:1802.03888 "
                     "(2018) for the enumeration check",
        "provenance": {**git_state(), "python": sys.executable, "platform": platform.platform(),
                       "argv": list(argv) if argv is not None else sys.argv[1:],
                       "versions": {"numpy": np.__version__, "lightgbm": lgb.__version__,
                                    "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__},
                       "env": {k: os.environ.get(k) for k in ("LOL_OUTPUT_ROOT", "LOL_CFG_PRESET", "LOL_CFG_OVERRIDES",
                                                              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                       "seed": args.seed, "n_jobs": args.n_jobs, "index_workers": args.index_workers,
                       "started_unix": started, "wall_s": None, "stage_seconds": stages},
        "preset": {"name": args.preset, "values": preset_values},
        "configurations": {n: {"TIME_NORM_ABSOLUTE": sw[0], "ANCHORS_CAUSAL": sw[1], "other_fields": f"preset {args.preset}"}
                           for n, sw in CONFIGS.items()},
        "label": {"key": label_key, "label_type": args.label, "tie_policy": "drop",
                  "attribution": getattr(cfg, "LABEL_EVENT_ATTRIBUTION", None),
                  "row_population": f"row label {ROW_LABEL_TYPE} with tie policy random (the v3.3 shard row population), "
                                    f"then rows where {label_key} is a draw removed",
                  "n_refs": int(len(refs)), "n_draws_dropped": int((~keep).sum())},
        "sample": {"shards": str(args.shards), "seed": args.seed, "n_matches_requested": args.n_matches,
                   "n_matches_available": n_available, "n_matches_sampled": len(mids), "match_sample_sha1": sample_sha1,
                   "n_matches_with_rows": int(len(np.unique(groups))), "n_rows": int(len(y)), "n_features": int(len(names)),
                   "positive_rate": float(y.mean()), "teamfight_min": args.teamfight_min,
                   "draw": "np.random.default_rng(seed).choice(sorted shard groups), as run_deep_tabular_baselines.load_subsample"},
        "split": split,
        "row_identity": identity,
        "shard_crosscheck": crosscheck,
        "leak_columns": leak["record"],
        "learner": {"config": {**lgbm_config, "early_stopping_rounds": 50, "eval_metric": "auc", "random_state": args.seed,
                               "n_jobs": args.n_jobs},
                    "source": "run_deep_tabular_baselines.LGBM_PUBLISHED / run_lightgbm",
                    "standardize": args.standardize},
        "models": models,
        "test_auc": test_auc,
        "shap": {"algorithm": "LightGBM Booster.predict(pred_contrib=True), path-dependent Tree SHAP, raw margin",
                 "n_rows": int(len(shap_rows)), "row_sample": f"np.random.default_rng({args.seed}).choice(test rows), sorted",
                 "row_sample_sha1": hashlib.sha1(json.dumps([keys['clean'][i] for i in shap_rows]).encode("utf-8")).hexdigest(),
                 "by_scale_rows": {c: int((shap_class == c).sum()) for c in CLASSES},
                 **{n: {"seconds": shap_res[n]["seconds"], "expected_value": shap_res[n]["expected_value"],
                        "additivity_max_abs_err": shap_res[n]["additivity_max_abs_err"]} for n in CONFIGS}},
        "shap_verification": verification,
        "attribution": reports,
        "leak_share": leak_share,
        "leak_share_note": "share of total mean |SHAP| on the leak-affected bases (base closure) or on the affected columns "
                           "alone ('[affected columns]'). In the clean configuration the same bases hold their causal values "
                           "(absolute game time; anchors destroyed before the cutoff), so a clean share is legitimate use of "
                           "those features; the leaky-minus-clean difference is the attribution the leak adds.",
        "rank_shift": rank_shift,
        "role_rollup": rollups,
        "legacy_artefact_check": legacy,
        "leak_discipline": [
            "split by patch (15.14 / 15.15 / 15.16); LightGBM early stopping sees only the validation patch; no calibration step",
            "standardisation statistics come from train-patch rows only",
            "test-patch rows are used only for the reported AUCs, the SHAP explanation and the checks, after fitting",
            "the leak column set is a label-free property of the two feature builds (values only, no labels or predictions); "
            "it groups attributions after fitting and never selects model inputs",
            "the auxiliary verification model is fitted on train-patch rows only and is never evaluated for AUC",
            "the leaky configuration deliberately reads post-cutoff information; it is the forensic counterfactual, never a reported model",
        ],
        "deviations": deviations,
        "outputs": {"json": str(out_json), "preds": str(preds_path), "csv": csvs},
    }
    results["provenance"]["wall_s"] = round(time.time() - started, 1)
    write_json_atomic(out_json, results)
    u = results["attribution"]
    print(f"leak share (union): clean {u['clean']['leak']['union']['share']} leaky {u['leaky']['leak']['union']['share']}", flush=True)
    print("wrote", out_json, f"{results['provenance']['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
