"""Prove that every learner in the comparison received the same rows, columns, labels and split.

"All models saw identical inputs" is a claim the paper makes, so it has to be checked against the
artefacts rather than asserted from the code.  This reads the shards once and the two result files,
and reports PASS/FAIL for each thing that could differ:

  rows        the same 532,547 labelled engagements
  matches     the same 191,940 matches
  split       the same patch holdout, row for row
  label       the same y_market_event, same positive rate
  columns     which columns each pipeline used, and whether the ones dropped by the tree
              comparison really are constant across the corpus (a constant column carries no
              information for any learner, so dropping it must not change anything)
  agreement   the two independent LightGBM runs on the two column sets

It also checks the token view used by FT-Transformer and SAINT: v3.3 carries one unstructured
column (frame_age_s) after the 7 x 1,015 block, so the reshape has to place it in its own token
rather than failing or silently misaligning the suffix families.

--extended (ToG revision; cite this audit, not the 19-check one).  Without the flag the script
reproduces the published 19-check audit byte for byte.  One of those checks ("columns dropped by the
tree pipeline are constant across the corpus") compared a count with itself and could not fail, and
the published audit never opened the FT-Transformer or SAINT results or any prediction file.
--extended keeps every other check and adds:

  corpus       the tree run's corpus manifest equals the shards' manifest, the feature-name hash
               matches, no shard was written after the earliest result, and the tree run's merged
               matrix was last written before its first model could have started
  deep runs    the row / match / label / positive-rate / split / column checks for every deep JSON
  columns      the tautology is replaced: the tree run's merged matrix (memory-mapped read-only) is
               compared with the shards cell by cell; the columns it lacks are shown to hold one value
               in every shard row by an equality test that does not reuse the pipeline's min/max rule;
               and the kept set is compared with the columns non-constant over all patches, over the
               train patch alone, and over every patch except the test patch
  asymmetry    the tree pipeline picks columns by scanning every patch (test included) while the deep
               pipeline standardises on the train patch only; recorded with its size, not a failure
  predictions  every saved test-prediction file: y, groups and row order equal the test-patch rows of
               the shards element by element and are identical across files; every reported model has
               predictions that reproduce its reported AUC; within-match row order is tested against
               within-match shuffles; subsample runs are checked as subsets; every prediction file on
               disk is inventoried and what was never saved is listed
  row order    the element-wise checks fix a file's order up to permutations among rows that agree on
               everything the file stores beside its predictions (tree file: match, label, participation
               and presence class; deep file: match, label, smaller-side participant count).  Rows alone in
               such a stratum are fixed; the rest are tested by shuffling one file's predictions inside
               exactly those strata (19 shuffles, so beating all of them is p <= 1/20 per learner).  Both
               runners write predictions and metadata from the same row mask in one call, so a file cannot
               be misaligned with its own metadata; the tests are about order across files
  inventory    every *v33*.preds.npz in the features folder and every *.preds.npz / pred_*.npz under
               tog_revision/ (other revision items' outputs), classified; a file that covers the full test
               patch but is not audited is named, so a later learner-table run cannot go unaudited silently

--extended refuses to write over the published audit.

    python scripts/audit_model_comparison_inputs.py --extended
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np

N_SUFFIXES = 7
ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path("D:/LOL_Project/fusion_2615/features")
LEGACY_OUT = FEATURES / "model_comparison_input_audit.json"
EXTENDED_OUT = FEATURES / "tog_revision" / "A8-input-audit-and-cis" / "model_comparison_input_audit_extended.json"
SEED = 7
N_SHUFFLES = 19  # within-match permutation test: beating all 19 shuffles is p <= 1 / 20
NOT_SAVED = [
    "validation-patch (15.15) predictions: saved for no learner.  Both runners write test-patch rows only "
    "(worktree scripts/run_model_comparison_v33.py line 255, y=y[te]; main repo "
    "scripts/run_deep_tabular_baselines.py at commit 60945ed lines 610-612, y_test=y[te])",
    "training-seed replicates: every learner was fitted once with seed 7; no second-seed predictions exist",
    "fitted models: neither runner saves a model (no torch.save / save_model / joblib in either file), so the "
    "test predictions cannot be regenerated without refitting",
    "learner-table arms not yet run (docs/tog_manuscript/sec_learners.tex, table tab:learners): searched "
    "LightGBM, capacity-matched FT-Transformer, TabNet with the corrected sparsity sign, SAINT with "
    "contrastive pre-training - no result JSON or prediction file for them (see prediction_inventory)",
    "model_comparison_v33.preds.npz is the 5-fold match-grouped out-of-fold run over all patches, not a patch "
    "holdout; it holds lead_only and linear only, and its JSON reports only those two",
]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)
    return {"name": name, "pass": bool(ok), "detail": detail}


def merge_range(acc, rng):
    """Combine two per-column (min, max) pairs; None is the empty range."""
    if rng is None:
        return acc
    return rng if acc is None else (np.minimum(acc[0], rng[0]), np.maximum(acc[1], rng[1]))


def fold_range(acc, X):
    """Running per-column (min, max) over a row subset - the reduction merge_shards applies."""
    return acc if len(X) == 0 else merge_range(acc, (X.min(axis=0), X.max(axis=0)))


def non_constant(rng):
    return set() if rng is None else set(np.flatnonzero(rng[1] > rng[0]).tolist())


def iso(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def sha1_of(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path, digest=True):
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime": iso(st.st_mtime),
            **({"sha1": sha1_of(path)} if digest else {})}


def git_state():
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True,
                                  timeout=30).stdout
        except Exception as exc:  # provenance must never abort the audit
            return f"unavailable: {exc}"
    # porcelain lines are 'XY path'; the leading status column may be a space, so never strip the output
    return {"repo": str(ROOT), "commit": run("rev-parse", "HEAD").strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty_files": [ln[3:] for ln in run("status", "--porcelain").splitlines()
                            if ln[3:].startswith("scripts/")]}


def second_pass(matrix_path, shard_paths, keep, mode, y_key, const_cols, const_vals, watch, seed=SEED):
    """One more read of every shard.

    (1) The tree run's merged matrix against shards[:, keep], cell by cell.  merge_shards
        (run_scale_decomposition.py) writes every shard row, labelled or not, in shard order, so shard
        row r sits at the running offset.  'full' compares every cell; 'sample' the first 128 and 128
        random rows of each shard.  The file is only memory-mapped for reading and its mtime is taken
        before and after, so a concurrent rewrite shows as a failure.
    (2) On the labelled rows of the watched patches, how far the columns that are constant on the train
        patch move away from their train value - what a network standardised on the train patch meets
        at validation and test time.
    """
    mat = {"path": str(matrix_path), "mode": mode, "exists": matrix_path.exists(), "ok": False}
    dev = {pt: {"rows": 0, "rows_with_any_change": 0, "columns_changed": np.zeros(len(const_cols), dtype=bool),
                "max_abs_change": 0.0} for pt in watch}
    M = None
    if mode != "off" and mat["exists"]:
        before = matrix_path.stat().st_mtime
        M = np.load(matrix_path, mmap_mode="r")
        mat.update(shape=list(M.shape), rows_compared=0, cells_compared=0, mismatched_shards=[])
    rng = np.random.default_rng(seed)
    offset = 0
    for path in shard_paths:
        with np.load(path, allow_pickle=True) as z:
            X, y, p = z["X"], z[y_key], z["patch"]
        n = len(X)
        if M is not None and M.shape[1] == len(keep) and offset + n <= M.shape[0]:
            if mode == "full":
                block, ref = np.asarray(M[offset:offset + n]), X[:, keep]
            else:
                idx = np.unique(np.r_[np.arange(min(128, n)), rng.integers(n, size=min(128, n))])
                block, ref = np.asarray(M[offset + idx]), X[idx][:, keep]
            if not np.array_equal(block, ref):
                mat["mismatched_shards"].append(path.name)
            mat["rows_compared"] += int(len(block))
            mat["cells_compared"] += int(block.size)
            del block, ref
        if len(const_cols):
            for pt in watch:
                rows = np.flatnonzero((y >= 0) & (p == pt))
                if len(rows):
                    D = np.abs(X[np.ix_(rows, const_cols)].astype(np.float64) - const_vals)
                    moved = D > 0
                    d = dev[pt]
                    d["rows"] += int(len(rows))
                    d["rows_with_any_change"] += int(moved.any(axis=1).sum())
                    d["columns_changed"] |= moved.any(axis=0)
                    d["max_abs_change"] = max(d["max_abs_change"], float(D.max()))
        offset += n
    if M is not None:
        mat["rows_in_shards"] = offset
        mat["mtime_unchanged"] = matrix_path.stat().st_mtime == before
        mat["ok"] = bool(tuple(M.shape) == (offset, len(keep)) and not mat["mismatched_shards"]
                         and mat["mtime_unchanged"] and mat["rows_compared"] > 0)
    for d in dev.values():
        d["columns_changed"] = int(d["columns_changed"].sum())
    return mat, dev


def stratum_codes(*arrays):
    """One integer per row naming its combination of values across the given arrays."""
    codes = np.zeros(len(arrays[0]), dtype=np.int64)
    for arr in arrays:
        inv = np.unique(arr, return_inverse=True)[1].ravel().astype(np.int64)
        codes = np.unique(codes * (int(inv.max()) + 1) + inv, return_inverse=True)[1].ravel().astype(np.int64)
    return codes


def within_match_perms(groups, n, seed=SEED):
    """Permutations that shuffle rows inside each group and leave every group's rows in place.

    groups may be match ids or stratum_codes(); only group membership is used."""
    inv = np.unique(groups, return_inverse=True)[1]
    base = np.argsort(inv, kind="stable")
    rng = np.random.default_rng(seed)
    for _ in range(n):
        perm = np.empty(len(inv), dtype=np.int64)
        perm[base] = np.lexsort((rng.random(len(inv)), inv))
        yield perm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, default=Path("D:/LOL_Project/fusion_2615/corpus_shards_v33"))
    ap.add_argument("--tree-results", type=Path,
                    default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_v33_patch.json"))
    ap.add_argument("--deep-results", type=Path,
                    default=Path("D:/LOL_Project/fusion_2615/features/deep_tabular_v33_patch_full.json"))
    ap.add_argument("--y-key", default="y_market_event")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default {LEGACY_OUT} (the 19-check audit), or {EXTENDED_OUT} with --extended")
    ap.add_argument("--extended", action="store_true",
                    help="run the ToG-revision audit described above and write a new JSON")
    ap.add_argument("--extra-deep-results",
                    default=",".join(str(FEATURES / f"deep_tabular_v33_patch_{t}.json") for t in ("ft", "saint")),
                    help="[--extended] further deep-pipeline JSONs audited like --deep-results (comma-separated)")
    ap.add_argument("--subsample-results", default=str(FEATURES / "deep_tabular_v33_patch_fast.json"),
                    help="[--extended] deep runs on a match subsample (comma-separated); checked as subsets "
                         "of the test patch, never as learner-table rows")
    ap.add_argument("--tree-preds", type=Path, default=FEATURES / "model_comparison_v33_patch.preds.npz")
    ap.add_argument("--tree-matrix", type=Path, default=FEATURES / "model_comparison_v33_matrix.npy",
                    help="[--extended] the merged matrix the tree run fitted on; memory-mapped read-only")
    ap.add_argument("--matrix-check", choices=("full", "sample", "off"), default="full")
    ap.add_argument("--train-patch", default="15.14")
    ap.add_argument("--val-patch", default="15.15")
    ap.add_argument("--test-patch", default="15.16")
    a = ap.parse_args()
    ext = a.extended
    out = a.out or (EXTENDED_OUT if ext else LEGACY_OUT)
    if ext and out.resolve() == LEGACY_OUT.resolve():
        raise SystemExit(f"--extended never writes over the published 19-check audit {LEGACY_OUT}")
    started = time.time()
    results = []
    print("reading shards for the ground truth ...", flush=True)

    names = json.loads((a.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    col_min = col_max = None
    rows = labelled = 0
    matches, patches, pos = set(), {}, 0
    shard_paths = sorted(a.shards.glob("shard_*.npz"))
    cells, patches_all, nan_cells, ref_row, varies = {}, {}, 0, None, None
    test_rows = {k: [] for k in ("y", "groups", "cluster_blue", "cluster_red", "present_blue", "present_red")}
    for path in shard_paths:
        with np.load(path, allow_pickle=True) as z:
            X, y, g, p = z["X"], z[a.y_key], z["groups"], z["patch"]
            rows += len(y)
            keep = y >= 0
            labelled += int(keep.sum())
            pos += int((y[keep] == 1).sum())
            matches.update(np.unique(g[keep]).tolist())
            for patch, n in zip(*np.unique(p[keep], return_counts=True)):
                patches[str(patch)] = patches.get(str(patch), 0) + int(n)
            mn, mx = X.min(axis=0), X.max(axis=0)
            col_min = mn if col_min is None else np.minimum(col_min, mn)
            col_max = mx if col_max is None else np.maximum(col_max, mx)
            if ext:
                nan_cells += int(np.isnan(X).sum())
                if ref_row is None:
                    ref_row, varies = X[0].copy(), np.zeros(X.shape[1], dtype=bool)
                varies |= (X != ref_row).any(axis=0)   # equality with one row, not the min/max rule
                for patch, n in zip(*np.unique(p, return_counts=True)):
                    patches_all[str(patch)] = patches_all.get(str(patch), 0) + int(n)
                    for lab in (True, False):
                        m = (p == patch) & (keep == lab)
                        if m.any():
                            cells[(str(patch), lab)] = fold_range(cells.get((str(patch), lab)), X[m])
                te = keep & (p == a.test_patch)
                test_rows["y"].append(y[te])
                test_rows["groups"].append(g[te])
                for k in ("cluster_blue", "cluster_red", "present_blue", "present_red"):
                    test_rows[k].append(z[k][te])
    constant = np.flatnonzero(col_max <= col_min)
    non_constant_n = int(len(names) - len(constant))
    truth = {"rows_total": rows, "rows_labelled": labelled, "matches": len(matches),
             "positive_rate": pos / labelled, "patches": patches,
             "columns_total": len(names), "columns_constant": int(len(constant)),
             "columns_non_constant": non_constant_n}
    print(json.dumps(truth, indent=2), flush=True)

    tree = json.loads(a.tree_results.read_text(encoding="utf-8")) if a.tree_results.exists() else None
    deep = json.loads(a.deep_results.read_text(encoding="utf-8")) if a.deep_results.exists() else None
    deep_runs = {"full": (a.deep_results, deep)}
    if ext:
        for s in (s.strip() for s in a.extra_deep_results.split(",")):
            if s:
                deep_runs[Path(s).stem.rsplit("_", 1)[-1]] = (
                    Path(s), json.loads(Path(s).read_text(encoding="utf-8")) if Path(s).exists() else None)

    manifest, provenance_inputs, timing = {}, {}, {}
    if ext:
        print("\ncorpus identity")
        manifest = json.loads((a.shards / "manifest.json").read_text(encoding="utf-8"))
        results.append(check("the tree run's corpus manifest equals the shards' manifest.json",
                             tree is not None and tree.get("corpus_manifest") == manifest,
                             f"run_id {manifest.get('run_id')}, corpus commit {manifest.get('git_commit')}, "
                             f"{manifest.get('num_shards')} shards, complete {manifest.get('complete')}"))
        names_sha1 = sha1_of(a.shards / "feature_names.json")
        results.append(check("feature_names.json hash equals the manifest",
                             names_sha1 == manifest.get("feature_names_sha1"), names_sha1))
        result_files = [q for q in (a.tree_results, a.tree_preds,
                                    *(pp for path, _ in deep_runs.values()
                                      for pp in (path, path.with_suffix(".preds.npz")))) if q.exists()]
        shard_latest = max(q.stat().st_mtime for q in shard_paths)
        result_earliest = min(q.stat().st_mtime for q in result_files)
        results.append(check("no shard was written after the earliest learner result",
                             shard_latest < result_earliest,
                             f"latest shard {iso(shard_latest)}, earliest of {len(result_files)} result files "
                             f"{iso(result_earliest)}"))
        ok_time, detail = False, "tree matrix, tree predictions or tree results missing"
        if tree and a.tree_matrix.exists() and a.tree_preds.exists():
            fit_s = sum(float(m.get("seconds") or 0.0) for m in tree["models"].values())
            pred_t, mat_t = a.tree_preds.stat().st_mtime, a.tree_matrix.stat().st_mtime
            ok_time = mat_t <= pred_t - fit_s
            timing = {"matrix_mtime": iso(mat_t), "tree_predictions_mtime": iso(pred_t),
                      "tree_fit_seconds_total": fit_s, "latest_possible_first_fit_start": iso(pred_t - fit_s)}
            detail = (f"matrix written {iso(mat_t)}; predictions written {iso(pred_t)} after {fit_s:.0f} s of "
                      f"fitting, so the first fit began no later than {iso(pred_t - fit_s)} (merge_shards writes "
                      "the matrix before any fit).  An upper bound, not a date: the summed fit seconds leave out "
                      "scoring, saving and the in-run bootstrap, so the first fit may have begun earlier; that the "
                      "matrix holds the shards' values is shown by the cell-by-cell check, not by this one")
        results.append(check("the tree matrix on disk was last written no later than the latest possible start of the "
                             "tree run's first fit (prediction-file mtime minus summed fit seconds)",
                             ok_time, detail))
        results.append(check("no NaN in the shards (a NaN would make the min/max rule drop its column)",
                             nan_cells == 0, f"{nan_cells} NaN cells"))
        results.append(check("tree pipeline label key", tree is not None and tree.get("y_key") == a.y_key,
                             str((tree or {}).get("y_key"))))

    print("\nrows, matches, label")
    if tree:
        results.append(check("tree pipeline row count", tree["n"] == labelled, f"{tree['n']} vs {labelled}"))
        results.append(check("tree pipeline match count", tree["n_matches"] == len(matches),
                             f"{tree['n_matches']} vs {len(matches)}"))
    if deep:
        results.append(check("deep pipeline row count", deep["n_rows"] == labelled, f"{deep['n_rows']} vs {labelled}"))
        results.append(check("deep pipeline match count", deep["n_matches"] == len(matches),
                             f"{deep['n_matches']} vs {len(matches)}"))
        results.append(check("deep pipeline label key", deep["split"].get("y_key") == a.y_key,
                             str(deep["split"].get("y_key"))))
    if ext:
        for tag, (path, run) in deep_runs.items():
            if tag != "full":
                results.append(check(f"deep pipeline ({tag}) results present", run is not None, str(path)))
                if run is None:
                    continue
                results.append(check(f"deep pipeline ({tag}) row count", run["n_rows"] == labelled,
                                     f"{run['n_rows']} vs {labelled}"))
                results.append(check(f"deep pipeline ({tag}) match count", run["n_matches"] == len(matches),
                                     f"{run['n_matches']} vs {len(matches)}"))
                results.append(check(f"deep pipeline ({tag}) label key", run["split"].get("y_key") == a.y_key,
                                     str(run["split"].get("y_key"))))
            elif run is None:
                continue
            sp = run["split"]
            results.append(check(f"deep pipeline ({tag}) positive rate equals the shards",
                                 abs(run["positive_rate"] - pos / labelled) < 1e-12,
                                 f"{run['positive_rate']:.6f} vs {pos / labelled:.6f}"))
            for part in ("train", "val", "test"):
                patch = sp.get(f"{part}_patch")
                results.append(check(f"deep pipeline ({tag}) {part} patch and rows match the shards and the tree run",
                                     sp.get("kind") == "patch" and sp.get(part) == patches.get(patch)
                                     and (tree is None or (tree["split"][part], tree["split"]["rows"][part])
                                          == (patch, sp.get(part))),
                                     f"{patch}: {sp.get(part)} vs shard count {patches.get(patch)}"))
            results.append(check(f"deep pipeline ({tag}) sees every column and the same token bases",
                                 run["n_features"] == len(names)
                                 and run.get("n_columns_seen_by_every_model", len(names)) == len(names)
                                 and run.get("n_bases") == len(names) // N_SUFFIXES,
                                 f"{run['n_features']} cols, {run.get('n_bases')} bases vs {len(names)}"))

    print("\npatch split")
    if tree and deep:
        t, d = tree["split"]["rows"], deep["split"]
        for part, key in (("train", "train"), ("val", "val"), ("test", "test")):
            results.append(check(f"{part} rows identical", t[part] == d[key], f"tree {t[part]} vs deep {d[key]}"))
        for part, patch in (("train", tree["split"]["train"]), ("val", tree["split"]["val"]),
                            ("test", tree["split"]["test"])):
            results.append(check(f"{part} patch rows match the shards", t[part] == patches.get(patch),
                                 f"{t[part]} vs shard count {patches.get(patch)} for {patch}"))

    print("\ncolumns")
    if tree:
        results.append(check("tree pipeline column count equals non-constant columns",
                             tree["n_features"] == non_constant_n, f"{tree['n_features']} vs {non_constant_n}"))
    if deep:
        results.append(check("deep pipeline uses every column", deep["n_features"] == len(names),
                             f"{deep['n_features']} vs {len(names)}"))
    column_sets, matrix, asymmetry = {}, {}, {}
    if not ext:
        results.append(check("columns dropped by the tree pipeline are constant across the corpus",
                             len(constant) == len(names) - non_constant_n,
                             f"{len(constant)} constant columns, e.g. {[names[j] for j in constant[:3]]}"))
    else:
        all_cols = np.arange(len(names))
        kept = np.flatnonzero(col_max > col_min)            # run_scale_decomposition.merge_shards, line 110
        kept_set = set(kept.tolist())
        dropped = np.setdiff1d(all_cols, kept)
        whole = lambda keys: reduce(merge_range, (cells.get(k) for k in keys), None)
        lab_train = f"labelled rows, train patch {a.train_patch} only"
        not_test = f"every row, every patch except the test patch {a.test_patch}"
        sets = {
            "every row, all patches, by equality with one row (the merge rule recomputed without min/max)":
                set(np.flatnonzero(varies).tolist()),
            "labelled rows, all patches": non_constant(whole([k for k in cells if k[1]])),
            lab_train: non_constant(whole([(a.train_patch, True)])),
            f"labelled rows, validation patch {a.val_patch} only": non_constant(whole([(a.val_patch, True)])),
            f"labelled rows, test patch {a.test_patch} only": non_constant(whole([(a.test_patch, True)])),
            not_test: non_constant(whole([k for k in cells if k[0] != a.test_patch])),
        }
        eq_label = next(iter(sets))
        for label, s in sets.items():
            column_sets[label] = {"non_constant": len(s), "tree_kept_not_in_set": len(kept_set - s),
                                  "in_set_not_tree_kept": len(s - kept_set),
                                  "examples_tree_kept_not_in_set": [names[j] for j in sorted(kept_set - s)[:10]]}
            print(f"    {label}: {column_sets[label]}", flush=True)
        train_range = cells.get((a.train_patch, True))
        const_train = np.setdiff1d(all_cols, sorted(sets[lab_train]))
        const_vals = train_range[0][const_train].astype(np.float64) if train_range is not None else np.zeros(0)
        print(f"  second pass: tree matrix vs shards ({a.matrix_check}), train-constant columns on "
              f"{a.val_patch} and {a.test_patch} ...", flush=True)
        matrix, moved = second_pass(a.tree_matrix, shard_paths, kept, a.matrix_check, a.y_key,
                                    const_train, const_vals, (a.val_patch, a.test_patch))
        results.append(check("the tree run's merged matrix equals the shards restricted to the kept columns, cell by cell",
                             matrix["ok"],
                             f"matrix {matrix.get('shape')} vs {matrix.get('rows_in_shards')} shard rows x "
                             f"{len(kept)} columns; {matrix.get('cells_compared', 0):,} cells compared "
                             f"({a.matrix_check}); mismatched shards {matrix.get('mismatched_shards')}; "
                             f"mtime unchanged {matrix.get('mtime_unchanged')}"))
        vals, counts = np.unique(col_min[dropped], return_counts=True)
        top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:5]
        results.append(check("columns dropped by the tree pipeline are constant across the corpus",
                             matrix["ok"] and not varies[dropped].any(),
                             f"replaces the published check, which compared a count with itself: none of the "
                             f"{len(dropped)} columns missing from the verified tree matrix takes a second value "
                             f"in any of the {rows:,} shard rows, tested by equality with one row rather than by "
                             f"the pipeline's min/max rule; their values: "
                             + ", ".join(f"{v:g} in {c} columns" for c, v in top)
                             + f"; e.g. {[names[j] for j in dropped[:3]]}"))
        eq = column_sets[eq_label]
        lab = column_sets["labelled rows, all patches"]
        results.append(check("tree-kept columns equal the columns non-constant over all patches",
                             matrix["ok"] and eq["tree_kept_not_in_set"] == 0 and eq["in_set_not_tree_kept"] == 0,
                             f"{len(kept)} in the tree matrix vs {eq['non_constant']} that vary over every shard row "
                             f"(kept-only {eq['tree_kept_not_in_set']}, set-only {eq['in_set_not_tree_kept']}); over "
                             f"the labelled rows alone {lab['non_constant']} vary, so {lab['tree_kept_not_in_set']} "
                             "tree columns vary only on unlabelled rows"))
        tr_ = column_sets[lab_train]
        results.append(check(f"every column non-constant on the train patch ({a.train_patch}) is in the tree matrix",
                             matrix["ok"] and tr_["in_set_not_tree_kept"] == 0,
                             f"{tr_['non_constant']} columns vary on the labelled {a.train_patch} rows; missing from "
                             f"the tree matrix {tr_['in_set_not_tree_kept']}; in the tree matrix but constant on "
                             f"{a.train_patch} {tr_['tree_kept_not_in_set']}.  Implied by the check above, because the "
                             f"{a.train_patch} rows are a subset of every row; stated for the learners' training rows"))
        nt = column_sets[not_test]
        asymmetry = {
            "tree_selection": "run_scale_decomposition.merge_shards keeps a column when its max exceeds its min over "
                              "every shard row, all patches (test included), labelled or not; labels are never read",
            "deep_standardisation": "run_deep_tabular_baselines.py standardises with mean and sd of the labelled "
                                    f"train-patch rows, sd < 1e-6 set to 1 (commit 60945ed lines 505-507); all "
                                    f"{len(names)} columns are carried",
            "tree_columns_constant_on_train_labelled": tr_["tree_kept_not_in_set"],
            "tree_columns_that_vary_only_on_the_test_patch": nt["tree_kept_not_in_set"],
            "examples_varying_only_on_the_test_patch": nt["examples_tree_kept_not_in_set"],
            "columns_constant_on_train_labelled": int(len(const_train)),
            "those_columns_on_labelled_rows": moved,
        }
        mv, mt = moved.get(a.val_patch, {}), moved.get(a.test_patch, {})
        results.append(check("asymmetry recorded: tree column selection scans every patch (test included), "
                             "deep standardisation uses the train patch only",
                             matrix["ok"] and tr_["in_set_not_tree_kept"] == 0,
                             f"recorded, not a failure.  {tr_['tree_kept_not_in_set']} of the {len(kept)} tree columns "
                             f"are constant on the labelled {a.train_patch} rows and {nt['tree_kept_not_in_set']} vary "
                             f"only on {a.test_patch}, so without {a.test_patch} the rule would keep {nt['non_constant']}; "
                             f"a tree fitted on {a.train_patch} cannot split on a column constant there, so the test "
                             "patch decides only whether such a column is carried, never what a tree learns.  The deep "
                             f"pipeline carries all {len(names)} columns standardised on {a.train_patch}: the "
                             f"{len(const_train)} columns constant there move on {mv.get('rows_with_any_change', 0):,} of "
                             f"{mv.get('rows', 0):,} {a.val_patch} rows and {mt.get('rows_with_any_change', 0):,} of "
                             f"{mt.get('rows', 0):,} {a.test_patch} rows ({mt.get('columns_changed', 0)} columns on "
                             f"{a.test_patch}, max |change| {mt.get('max_abs_change', 0.0):g}); weights on those "
                             "inputs got no gradient from training data, so on those rows they can add only noise, "
                             "never label information"))

    print("\ntoken view (FT-Transformer / SAINT)")
    n_bases = len(names) // N_SUFFIXES
    structured = N_SUFFIXES * n_bases
    extras = names[structured:]
    results.append(check("suffix families are contiguous and suffix-major",
                         all(names[i * n_bases].rsplit("__", 1)[-1] ==
                             names[i * n_bases + 5].rsplit("__", 1)[-1] for i in range(N_SUFFIXES)),
                         str([names[i * n_bases].rsplit("__", 1)[-1] for i in range(N_SUFFIXES)])))
    results.append(check("unstructured columns get their own token, none dropped",
                         structured + len(extras) == len(names),
                         f"{n_bases} structured tokens + {len(extras)} extra {extras}"))
    probe = np.arange(len(names), dtype=np.float32)[None, :]
    tok = probe[:, :structured].reshape(1, N_SUFFIXES, n_bases).transpose(0, 2, 1)
    pad = np.zeros((1, len(extras), N_SUFFIXES), dtype=tok.dtype)
    pad[:, :, 0] = probe[:, structured:]
    tok = np.concatenate([tok, pad], axis=1)
    expected = [0 + i * n_bases for i in range(N_SUFFIXES)]
    results.append(check("token 0 carries base 0 under all 7 statistics",
                         np.allclose(tok[0, 0], expected), f"{tok[0, 0].tolist()} vs {expected}"))
    results.append(check("every column reaches exactly one token",
                         sorted(tok[0][tok[0] != 0].tolist() + [0.0]) == sorted(float(i) for i in range(len(names))),
                         f"{tok.shape[1]} tokens x {N_SUFFIXES}"))

    print("\nindependent LightGBM agreement across the two column sets")
    if tree and deep and "lgbm_paper" in tree.get("models", {}) and "lightgbm" in deep.get("models", {}):
        a1 = tree["models"]["lgbm_paper"]["overall_auc"]
        a2 = deep["models"]["lightgbm"]["test_auc"]
        results.append(check("same learner, both column sets, AUC within 0.002",
                             abs(a1 - a2) < 0.002,
                             f"{a1:.4f} ({tree['models']['lgbm_paper']['n_features']} cols) vs "
                             f"{a2:.4f} ({deep['n_features']} cols), diff {a1 - a2:+.4f}"))

    prediction_files, row_order, inventory, subsample = {}, {}, [], {}
    unaudited_full, learner_files = [], {}
    if ext:
        from sklearn.metrics import roc_auc_score
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_scale_decomposition import scale_class  # the function the tree run classified rows with

        print(f"\nprediction files, row for row (test patch {a.test_patch})")
        tt = {k: np.concatenate(v) for k, v in test_rows.items()}
        files = {"tree": (a.tree_preds, tree, "overall_auc")}
        files.update({f"deep:{tag}": (path.with_suffix(".preds.npz"), run, "test_auc")
                      for tag, (path, run) in deep_runs.items()})
        loaded = {}
        for label, (path, run, auc_key) in files.items():
            results.append(check(f"{label} predictions saved", path.exists(), str(path)))
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            if label == "tree":
                y_, g_, cls, prs = (arrays.pop(k) for k in ("y", "groups", "classes", "present"))
                order_ok = (np.array_equal(cls, scale_class(tt["cluster_blue"], tt["cluster_red"], 4).astype(str))
                            and np.array_equal(prs, scale_class(tt["present_blue"], tt["present_red"], 4).astype(str)))
                order_what, preds_ = "participation and presence classes", arrays
            else:
                y_, g_, mp = (arrays.pop(k) for k in ("y_test", "groups_test", "min_participants_test"))
                order_ok = np.array_equal(mp, np.minimum(tt["cluster_blue"], tt["cluster_red"]))
                order_what = "smaller-side participant counts"
                preds_ = {k[len("pred_"):]: v for k, v in arrays.items() if k.startswith("pred_")}
            reported = (run or {}).get("models", {})
            aucs = {k: float(roc_auc_score(y_, v)) for k, v in preds_.items()}
            results.append(check(f"{label} has predictions for every model its JSON reports",
                                 bool(preds_) and set(preds_) == set(reported),
                                 f"preds {sorted(preds_)} vs json {sorted(reported)}"))
            results.append(check(f"{label} y equals the shard test patch element-wise",
                                 np.array_equal(y_.astype(np.int64), tt["y"].astype(np.int64)), f"{len(y_)} rows"))
            results.append(check(f"{label} groups equal the shard test patch element-wise",
                                 np.array_equal(g_, tt["groups"]), f"{len(np.unique(g_))} matches"))
            results.append(check(f"{label} row order carries the shards' {order_what}", order_ok))
            gap = max((abs(aucs[k] - reported[k][auc_key]) for k in aucs if k in reported), default=float("nan"))
            results.append(check(f"{label} reproduces the AUC its JSON reports", gap < 1e-9,
                                 f"max |diff| {gap:.1e}; " + ", ".join(f"{k} {v:.4f}" for k, v in aucs.items())))
            loaded[label] = (y_, g_, preds_)
            prediction_files[label] = {**file_info(path), "n_rows": int(len(y_)),
                                       "n_matches": int(len(np.unique(g_))), "test_auc": aucs,
                                       "dtypes": {k: str(v.dtype) for k, v in preds_.items()}}
        if "tree" in loaded:
            y0, g0, p0 = loaded["tree"]
            same = all(np.array_equal(yy.astype(np.int64), y0.astype(np.int64)) and np.array_equal(gg, g0)
                       for yy, gg, _ in loaded.values())
            results.append(check("every prediction file has identical y, groups and row order",
                                 same and len(loaded) == len(files), f"{len(loaded)} of {len(files)} files"))
            if "lgbm_paper" in p0:
                ref = p0["lgbm_paper"].astype(np.float64)
                perms = list(within_match_perms(g0, N_SHUFFLES))
                corr = lambda u, v: float(np.corrcoef(u, v)[0, 1])
                for label, (_, _, pr) in loaded.items():
                    if label == "tree":
                        continue
                    stats = {}
                    for k, v in pr.items():
                        v = v.astype(np.float64)
                        shuffled = [corr(ref, v[q]) for q in perms]
                        stats[k] = {"r_aligned": corr(ref, v), "r_shuffled_max": max(shuffled),
                                    "r_shuffled_mean": float(np.mean(shuffled))}
                    row_order[label] = stats
                    results.append(check(
                        f"{label} row order inside matches: every learner's correlation with lgbm_paper beats "
                        f"{N_SHUFFLES} within-match shuffles",
                        bool(stats) and all(s["r_aligned"] > s["r_shuffled_max"] for s in stats.values()),
                        "; ".join(f"{k} r {s['r_aligned']:.4f} vs shuffled max {s['r_shuffled_max']:.4f}"
                                  for k, s in stats.items())
                        + " (a shuffle keeps every row in its own match, so a row-order slip inside a match would "
                          "fall to the shuffled value)"))

                # Each file's order equals the shards' order up to a permutation that its element-wise checks
                # cannot see: one that only exchanges rows agreeing on everything the file stores next to its
                # predictions.  The tree file stores match, label, participation class and presence class; a
                # deep file stores match, label and the smaller-side participant count.  A row alone in its
                # stratum is therefore fixed by those checks; the rows sharing a stratum are tested here by
                # shuffling that file's side only, inside those strata and nowhere else.
                sides = {
                    "deep": ("deep predictions", "match, label, smaller-side participants",
                             stratum_codes(g0, y0.astype(np.int64),
                                           np.minimum(tt["cluster_blue"], tt["cluster_red"]).astype(np.int64))),
                    "tree": ("lgbm_paper predictions", "match, label, participation class, presence class",
                             stratum_codes(g0, y0.astype(np.int64),
                                           scale_class(tt["cluster_blue"], tt["cluster_red"], 4).astype(str),
                                           scale_class(tt["present_blue"], tt["present_red"], 4).astype(str))),
                }
                for side, (what, key, strata) in sides.items():
                    sizes = np.bincount(strata)
                    movable = int((sizes[strata] > 1).sum())
                    sperms = list(within_match_perms(strata, N_SHUFFLES, seed=SEED + 1))
                    moved_rows = int(np.mean([(q != np.arange(len(q))).sum() for q in sperms]))
                    strat = {"shuffled": what, "stratum": key, "n_strata": int(len(sizes)),
                             "rows_in_multi_row_strata": movable, "rows_moved_per_shuffle_mean": moved_rows,
                             "learners": {}}
                    for label, (_, _, pr) in loaded.items():
                        if label == "tree":
                            continue
                        for k, v in pr.items():
                            v = v.astype(np.float64)
                            shuffled = [corr(ref, v[q]) if side == "deep" else corr(ref[q], v) for q in sperms]
                            strat["learners"][f"{label}:{k}"] = {"r_aligned": corr(ref, v),
                                                                 "r_shuffled_max": max(shuffled),
                                                                 "r_shuffled_mean": float(np.mean(shuffled))}
                    row_order[f"within_strata_invisible_to_{side}_file_checks"] = strat
                    sl = strat["learners"]
                    results.append(check(
                        f"{side}-file row order among rows its element-wise checks cannot tell apart: every deep "
                        f"learner's correlation with lgbm_paper beats {N_SHUFFLES} shuffles of the {what} within "
                        f"({key}) strata",
                        bool(sl) and movable > 0 and all(s["r_aligned"] > s["r_shuffled_max"] for s in sl.values()),
                        f"{movable:,} of {len(g0):,} test rows share their stratum with another row "
                        f"({len(sizes):,} strata; a shuffle moves {moved_rows:,} rows on average); "
                        + "; ".join(f"{k} r {s['r_aligned']:.4f} vs shuffled max {s['r_shuffled_max']:.4f}"
                                    for k, s in sl.items())
                        + ".  Every other row is alone in its stratum, so the element-wise checks already fix "
                          "its position"))

        for s in (s.strip() for s in a.subsample_results.split(",")):
            if not s:
                continue
            jp = Path(s)
            pp = jp.with_suffix(".preds.npz")
            if not (jp.exists() and pp.exists()):
                subsample[str(jp)] = "absent"
                continue
            run = json.loads(jp.read_text(encoding="utf-8"))
            with np.load(pp, allow_pickle=False) as z:
                ys, gs, ms = z["y_test"], z["groups_test"], z["min_participants_test"]
                ps = {k[len("pred_"):]: z[k] for k in z.files if k.startswith("pred_")}
            sel = np.isin(tt["groups"], np.unique(gs))
            same_rows = (np.array_equal(ys.astype(np.int64), tt["y"][sel].astype(np.int64))
                         and np.array_equal(gs, tt["groups"][sel])
                         and np.array_equal(ms, np.minimum(tt["cluster_blue"], tt["cluster_red"])[sel]))
            aucs = {k: float(roc_auc_score(ys, v)) for k, v in ps.items()}
            gap = max((abs(aucs[k] - run["models"][k]["test_auc"]) for k in aucs if k in run["models"]),
                      default=float("nan"))
            subsample[str(jp)] = {**file_info(pp), "n_rows_run": run["n_rows"], "n_matches_run": run["n_matches"],
                                  "n_test_rows": int(len(ys)), "test_auc": aucs}
            results.append(check(
                f"subsample run {jp.stem} (not a learner-table row): its test rows are the shard test-patch rows of "
                "its own matches, in order, and reproduce its AUCs",
                same_rows and gap < 1e-9 and run["split"].get("y_key") == a.y_key,
                f"{len(ys):,} of {len(tt['y']):,} test rows from {len(np.unique(gs)):,} test matches; run on "
                f"{run['n_matches']:,} matches / {run['n_rows']:,} rows; max AUC |diff| {gap:.1e}"))

        used = {str(path) for path, _, _ in files.values()}
        feat_dir = a.tree_preds.parent
        found = sorted({*feat_dir.glob("deep_tabular_v33_patch_*.preds.npz"),
                        *feat_dir.glob("model_comparison_v33*.preds.npz")})
        # everything else that could hold v3.3 predictions: other top-level v3.3 runs and every revision item's
        # outputs.  Files for earlier corpora carry no 'v33' in their name and are not listed.
        found_more = sorted({*feat_dir.glob("*v33*.preds.npz"), *(feat_dir / "tog_revision").rglob("*.preds.npz"),
                             *(feat_dir / "tog_revision").rglob("pred_*.npz")} - set(found))
        meta_keys = {"y", "y_test", "groups", "groups_test", "match_id", "classes", "present", "scale", "presence",
                     "patch", "engage_ts", "engage_ts_test", "cluster_blue", "cluster_red", "min_participants_test"}
        for pp in found + found_more:
            jp = pp.with_name(pp.name.replace(".preds.npz", ".json"))
            run = None
            try:
                run = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else None
            except (OSError, ValueError):
                pass
            try:
                with np.load(pp, allow_pickle=False) as z:
                    keys = list(z.files)
                    yk = next((k for k in ("y_test", "y") if k in keys), None)
                    gk = next((k for k in ("groups_test", "groups", "match_id") if k in keys), None)
                    n_rows = int(len(z[yk])) if yk else None
                    gg = z[gk] if gk else None
            except Exception as exc:  # another item may be writing the file right now
                inventory.append({"file": str(pp), "readable": False, "error": f"{type(exc).__name__}: {exc}",
                                  "role": "unreadable at audit time; not a learner-table row"})
                continue
            learners = ([k[len("pred_"):] for k in keys if k.startswith("pred_")]
                        or [k for k in keys if k not in meta_keys and not k.endswith("_index")])
            full_test = bool(gg is not None and len(gg) == len(tt["groups"]) and np.array_equal(gg, tt["groups"]))
            if str(pp) in used:
                role = "learner comparison: patch holdout, the full test patch"
            elif run is not None and "folds" in run:
                role = (f"{run['folds']}-fold match-grouped out-of-fold predictions over all patches; not a patch "
                        f"holdout, not a learner-table row; its JSON reports {sorted(run.get('models', {}))}")
            elif full_test:
                role = ("COVERS THE FULL TEST PATCH but is not audited here: if it is a learner-table row, re-run "
                        "with it in --extra-deep-results")
            elif n_rows is not None and n_rows >= labelled:
                role = (f"one prediction per labelled row of every patch (label {(run or {}).get('y_key', '?')}, "
                        f"{n_rows:,} rows): out-of-fold over all patches, not a patch holdout, not a learner-table row")
            elif run is not None and run.get("n_matches") not in (None, len(matches)):
                role = (f"subsample run on {run['n_matches']:,} matches ({run.get('n_rows', 0):,} rows); not a "
                        "learner-table row")
            elif "tog_revision" in pp.parts:
                item = pp.parts[pp.parts.index("tog_revision") + 1]
                role = f"revision item {item}: {n_rows if n_rows is not None else '?'} rows, not the full test patch; " \
                       "not a learner-table row"
            else:
                role = "not used by the learner comparison"
            inventory.append({"file": str(pp), "json": str(jp) if jp.exists() else None, "n_rows": n_rows,
                              "learners": learners, "covers_full_test_patch": full_test, "role": role,
                              "mtime": iso(pp.stat().st_mtime)})
        unaudited_full = [r["file"] for r in inventory if r.get("covers_full_test_patch") and r["file"] not in used]
        print(f"  inventory: {len(inventory)} prediction files ({len(found)} learner-comparison candidates, "
              f"{len(found_more)} elsewhere); covering the full test patch but not audited: {unaudited_full or 'none'}",
              flush=True)
        table = {label: sorted(v["test_auc"]) for label, v in prediction_files.items()}
        reported = {label: sorted((run or {}).get("models", {})) for label, (_, run, _) in files.items()}
        results.append(check("every learner in the patch-holdout comparison has saved test predictions",
                             len(prediction_files) == len(files) and table == reported,
                             f"{sum(len(v) for v in table.values())} learners: "
                             + "; ".join(f"{k} {v}" for k, v in table.items())
                             + f"; {len(found)} prediction files found under {feat_dir}, "
                               f"{sum(1 for r in inventory if str(r['file']) in used)} used; "
                               f"{len(inventory)} v3.3 prediction files inventoried in all, "
                               f"{len(unaudited_full)} of the others cover the full test patch"))
        learner_files = {f"{label}:{k}": v["path"] for label, v in prediction_files.items() for k in v["test_auc"]}

    payload = {"truth": truth, "checks": results,
               "verdict": "PASS" if all(r["pass"] for r in results) else "FAIL",
               "note": "the tree pipeline drops columns that are constant across the whole corpus; "
                       "a constant column is uninformative for every learner, and the two "
                       "independent LightGBM runs confirm it changes nothing measurable"}
    if ext:
        payload.update({
            "audit": "extended (ToG revision); supersedes the 19-check model_comparison_input_audit.json",
            "n_checks": len(results),
            "n_failed": sum(1 for r in results if not r["pass"]),
            "replaced_tautology": "the published check 'columns dropped by the tree pipeline are constant across the "
                                  "corpus' tested len(constant) == len(names) - non_constant, where non_constant was "
                                  "defined as len(names) - len(constant); it keeps its name here with a real test",
            "truth_extended": {"rows_by_patch_all_rows": patches_all, "rows_test_patch_labelled": int(len(tt["y"])),
                               "columns_varying_by_equality": int(varies.sum())},
            "column_sets": column_sets, "asymmetry": asymmetry, "tree_matrix": {**matrix, **timing},
            "prediction_files": prediction_files, "row_order_within_matches": row_order,
            "subsample_runs": subsample, "prediction_inventory": inventory,
            "saved_test_predictions": {
                "learners": learner_files,
                "summary": f"all {len(learner_files)} patch-holdout learners have saved test-patch predictions, in "
                           f"{len(prediction_files)} files; what was not saved is listed under not_saved",
                "full_test_patch_files_not_audited": unaudited_full},
            "not_saved": NOT_SAVED,
            "run_code_provenance": [
                "tree run (model_comparison_v33_patch.*) finished before worktree commit 3fb00c3 "
                "(2026-09-11 02:22:51), which carries scripts/run_model_comparison_v33.py",
                "deep full run (deep_tabular_v33_patch_full.*, 01:53) predates main-repo commit 60945ed "
                "(2026-09-11 02:22:29); its JSON lacks n_columns_seen_by_every_model, which the ft and saint "
                "JSONs (03:31, 04:49) carry",
            ],
            "provenance": {"script": str(Path(__file__).resolve()), "script_sha1": sha1_of(Path(__file__).resolve()),
                           "git": git_state(), "argv": sys.argv[1:],
                           "preset": os.environ.get("LOL_CFG_PRESET", "unset (corpus manifest below)"),
                           "corpus_run_id": manifest.get("run_id"), "corpus_git_commit": manifest.get("git_commit"),
                           "detector": manifest.get("detector"), "label_key": a.y_key,
                           "split": {"kind": "patch holdout", "train": a.train_patch, "val": a.val_patch,
                                     "test": a.test_patch},
                           "rows_labelled": labelled, "matches": len(matches), "seed": SEED,
                           "inputs": {"tree_results": file_info(a.tree_results),
                                      **{f"deep_{t}": file_info(p_) for t, (p_, r_) in deep_runs.items() if r_},
                                      "tree_matrix": file_info(a.tree_matrix, digest=False)
                                      if a.tree_matrix.exists() else None,
                                      "manifest": file_info(a.shards / "manifest.json")},
                           "wall_clock_s": round(time.time() - started, 1),
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [r["name"] for r in results if not r["pass"]]
    print(f"\nverdict: {payload['verdict']}" + (f" - failing: {failed}" if failed else ""), flush=True)
    if ext:
        print(f"{len(results)} checks; wrote {out} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
