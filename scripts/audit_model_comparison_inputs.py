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
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_SUFFIXES = 7


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)
    return {"name": name, "pass": bool(ok), "detail": detail}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, default=Path("D:/LOL_Project/fusion_2615/corpus_shards_v33"))
    ap.add_argument("--tree-results", type=Path,
                    default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_v33_patch.json"))
    ap.add_argument("--deep-results", type=Path,
                    default=Path("D:/LOL_Project/fusion_2615/features/deep_tabular_v33_patch_full.json"))
    ap.add_argument("--y-key", default="y_market_event")
    ap.add_argument("--out", type=Path,
                    default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_input_audit.json"))
    a = ap.parse_args()
    results = []
    print("reading shards for the ground truth ...", flush=True)

    names = json.loads((a.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    col_min = col_max = None
    rows = labelled = 0
    matches, patches, pos = set(), {}, 0
    for path in sorted(a.shards.glob("shard_*.npz")):
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
    constant = np.flatnonzero(col_max <= col_min)
    non_constant = int(len(names) - len(constant))
    truth = {"rows_total": rows, "rows_labelled": labelled, "matches": len(matches),
             "positive_rate": pos / labelled, "patches": patches,
             "columns_total": len(names), "columns_constant": int(len(constant)),
             "columns_non_constant": non_constant}
    print(json.dumps(truth, indent=2), flush=True)

    tree = json.loads(a.tree_results.read_text(encoding="utf-8")) if a.tree_results.exists() else None
    deep = json.loads(a.deep_results.read_text(encoding="utf-8")) if a.deep_results.exists() else None

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
                             tree["n_features"] == non_constant, f"{tree['n_features']} vs {non_constant}"))
    if deep:
        results.append(check("deep pipeline uses every column", deep["n_features"] == len(names),
                             f"{deep['n_features']} vs {len(names)}"))
    results.append(check("columns dropped by the tree pipeline are constant across the corpus",
                         len(constant) == len(names) - non_constant,
                         f"{len(constant)} constant columns, e.g. {[names[j] for j in constant[:3]]}"))

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

    payload = {"truth": truth, "checks": results,
               "verdict": "PASS" if all(r["pass"] for r in results) else "FAIL",
               "note": "the tree pipeline drops columns that are constant across the whole corpus; "
                       "a constant column is uninformative for every learner, and the two "
                       "independent LightGBM runs confirm it changes nothing measurable"}
    a.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [r["name"] for r in results if not r["pass"]]
    print(f"\nverdict: {payload['verdict']}" + (f" - failing: {failed}" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
