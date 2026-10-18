"""Participation and presence count distributions of corpus v3.3, and where the teamfight cut sits.

The v3.3 teamfight cut ``n_min = min(blue, red) >= 4`` was read from the v2 corpus
(docs/DEFINITION_EVIDENCE.md, scale section; features/scale_participation_dist.json): the smaller-side
marginal had no trough, while the k-v-k diagonal peaked at 2v2 and 5v5 with a trough at 4v4.  This script
recomputes that evidence on the v3.3 shards (G 13.7 s, D 4,264 u, R 1,600 u, B 15 s) for both count pairs:

  participation  ``cluster_blue/red``: kill-cluster participants plus in-radius interaction actors
  presence       ``present_blue/red``: alive champions within R of the first kill at the cutoff (>= M = 2)

and two populations: ``labelled`` (rows with a label for --y-key, i.e. the headline 532,547 with draws
dropped) and ``all_rows`` (every detected engagement row in the shards).  It reads only the per-row count
arrays of each shard, never X.

Reported per count pair: the joint (blue, red) matrix, its diagonal, the smaller-side, larger-side,
per-team and total marginals, |blue - red|, class shares at cuts 3/4/5 with positive rate and median
engagement minute, the {c..5} x {c..5} block mass, shares by patch, and the presence x participation
cross-tab.  Class shares, block mass, diagonal shares and smaller-side shares carry match-clustered
percentile intervals from the same replicates as the troughs.  The v2 distribution is recomputed from its
JSON for a side-by-side reading.

A -1 participation count is ``int(0 or -1)`` from data/index_split._fight_to_ref_row (gameplay/fights.py
always writes an int count), so ``--negative-count-as zero`` (default) reads it as 0; ``unknown`` keeps
those rows (42 labelled, 59 detected) out of every class as run_scale_decomposition.scale_class does.

Trough, defined here for a discrete distribution p(k) over its supported range k_lo..k_hi (the range with
non-zero mass in the full data): an interior k with p(k) < p(k-1) and p(k) < p(k+1).  Its depth is
1 - p(k) / min(max_{j<k} p(j), max_{j>k} p(j)); the primary trough is the deepest one.  Location and depth
get match-clustered percentile bootstrap intervals (Efron & Tibshirani 1993 ch. 13; clusters resampled
whole as in Field & Welsh 2007, JRSS-B 69(3)): matches are drawn with replacement and every distribution is
recomputed with match multiplicities as weights.

    .venv/Scripts/python.exe scripts/scale_participation_v33.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARDS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
DEFAULT_V2 = Path("D:/LOL_Project/fusion_2615/features/scale_participation_dist.json")
PAIRS = {"participation": ("cluster_blue", "cluster_red"), "presence": ("present_blue", "present_red")}
CUTS = (3, 4, 5)
_I, _J = np.indices((6, 6))
_MIN, _MAX, _TOT = np.minimum(_I, _J).ravel(), np.maximum(_I, _J).ravel(), (_I + _J).ravel()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def distributions(joint36: np.ndarray) -> dict:
    """Every marginal used below, derived from the flattened 6x6 joint (blue rows, red cols)."""
    j = np.asarray(joint36, dtype=float)
    return {"joint": j.reshape(6, 6), "diagonal": j.reshape(6, 6).diagonal().copy(),
            "smaller_side": np.bincount(_MIN, weights=j, minlength=6),
            "larger_side": np.bincount(_MAX, weights=j, minlength=6),
            "per_team": j.reshape(6, 6).sum(axis=1) + j.reshape(6, 6).sum(axis=0),
            "total": np.bincount(_TOT, weights=j, minlength=11)}


def support_of(p: np.ndarray) -> tuple:
    nz = np.flatnonzero(np.asarray(p) > 0)
    return (int(nz[0]), int(nz[-1])) if len(nz) else (0, -1)


def troughs(p: np.ndarray, support: tuple) -> list:
    lo, hi = support
    vals = [float(p[k]) for k in range(lo, hi + 1)]
    out = []
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            left, right = max(vals[:i]), max(vals[i + 1:])
            ref = min(left, right)
            out.append({"k": lo + i, "depth": (1.0 - vals[i] / ref) if ref > 0 else 0.0,
                        "p": vals[i], "left_peak": left, "right_peak": right})
    return out


def peaks(p: np.ndarray, support: tuple) -> list:
    lo, hi = support
    vals = [float(p[k]) for k in range(lo, hi + 1)]
    out = []
    for i, v in enumerate(vals):
        left = vals[i - 1] if i > 0 else -np.inf
        right = vals[i + 1] if i + 1 < len(vals) else -np.inf
        if v > left and v > right:
            out.append(lo + i)
    return out


def primary_trough(p: np.ndarray, support: tuple):
    t = troughs(p, support)
    return max(t, key=lambda d: d["depth"]) if t else None


def shares(counts: np.ndarray) -> dict:
    tot = float(np.sum(counts))
    return {str(k): {"n": int(round(v)), "share": (float(v / tot) if tot else None)} for k, v in enumerate(counts)}


def analyse_pair(b, r, y, groups, patch, minute, boot_mod, n_boot: int, seed: int,
                 negative_count_as: str = "unknown") -> dict:
    negative = (b < 0) | (r < 0)
    raw_range = {"range_blue": [int(b.min()), int(b.max())], "range_red": [int(r.min()), int(r.max())]}
    if negative_count_as == "zero":  # a -1 is int(0 or -1): a team with zero counted champions
        b, r = np.maximum(b, 0), np.maximum(r, 0)
    elif negative_count_as != "unknown":
        raise ValueError(negative_count_as)
    ok = (b >= 0) & (r >= 0)
    if ok.any() and (b[ok].max() > 5 or r[ok].max() > 5):
        raise SystemExit("per-team counts above 5")
    code = np.where(ok, b * 6 + r, -1)
    joint = np.bincount(code[ok], minlength=36).astype(float)
    d = distributions(joint)
    n_ok = float(ok.sum())
    out = {"n": int(len(b)), "negative_count_as": negative_count_as, "rows_with_negative_count": int(negative.sum()),
           "sentinel_rows": int((~ok).sum()), **raw_range,
           "joint_blue_rows_red_cols": d["joint"].astype(int).tolist(),
           "joint_share": (d["joint"] / n_ok).round(6).tolist(),
           "diagonal": shares(d["diagonal"]), "diagonal_share_of_population": (d["diagonal"] / n_ok).round(6).tolist(),
           "smaller_side_marginal": shares(d["smaller_side"]), "larger_side_marginal": shares(d["larger_side"]),
           "per_team_marginal": shares(d["per_team"]), "total_marginal": shares(d["total"])}
    asym = np.abs(b - r)[ok]
    out["asymmetry_abs_blue_minus_red"] = shares(np.bincount(asym, minlength=6).astype(float))

    # troughs: point estimates and match-clustered bootstrap of location and depth
    analyses = ("diagonal", "smaller_side", "per_team", "total")
    supports = {a: support_of(d[a]) for a in analyses}
    tr = {}
    for a in analyses:
        tr[a] = {"support": list(supports[a]), "peaks": peaks(d[a], supports[a]),
                 "troughs": troughs(d[a], supports[a]), "primary": primary_trough(d[a], supports[a])}
    boot = boot_mod.MatchBootstrap(groups, seed)
    loc = {a: [] for a in analyses}
    depth = {a: [] for a in analyses}
    diag4 = []
    rows_ok = np.flatnonzero(ok)
    inv_ok = boot.inverse[rows_ok]
    code_ok = code[rows_ok]
    inv_bad = boot.inverse[np.flatnonzero(~ok)]
    diag_draws, small_draws = [], []
    class_draws = {str(c): {k: [] for k in ("pick", "skirmish", "teamfight", "unknown", "block_mass_c_to_5")} for c in CUTS}
    for _ in range(int(n_boot)):
        counts, _sample = boot.draw()
        jb = np.bincount(code_ok, weights=counts[inv_ok], minlength=36)
        db = distributions(jb)
        for a in analyses:
            pt = primary_trough(db[a], supports[a])
            loc[a].append(pt["k"] if pt else None)
            if pt:
                depth[a].append(pt["depth"])
        dg = db["diagonal"]
        diag4.append(bool(dg[4] < dg[3] and dg[4] < dg[5]))
        # shares with the same denominators as the point estimates: known rows for the pmfs, all rows for classes
        w_bad = float(counts[inv_bad].sum()) if len(inv_bad) else 0.0
        n_known = float(jb.sum())
        n_all = n_known + w_bad
        ss = db["smaller_side"]
        diag_draws.append(dg / n_known)
        small_draws.append(ss / n_known)
        for c in CUTS:
            cd = class_draws[str(c)]
            cd["pick"].append(ss[:2].sum() / n_all)
            cd["skirmish"].append(ss[2:c].sum() / n_all)
            cd["teamfight"].append(ss[c:].sum() / n_all)
            cd["unknown"].append(w_bad / n_all)
            cd["block_mass_c_to_5"].append(db["joint"][c:, c:].sum() / n_all)

    def pct(draws) -> dict:
        a = np.asarray(draws, dtype=float)
        return {"ci_2.5": np.percentile(a, 2.5, axis=0).round(6).tolist(),
                "ci_97.5": np.percentile(a, 97.5, axis=0).round(6).tolist(), "n_boot": int(len(a))}

    out["diagonal_share_of_population_ci"] = pct(diag_draws)
    out["smaller_side_share_ci"] = pct(small_draws)
    for a in analyses:
        ks = [k for k in loc[a] if k is not None]
        share = {str(k): float(np.mean([v == k for v in loc[a]])) for k in sorted(set(ks))}
        share["none"] = float(np.mean([v is None for v in loc[a]]))
        dep = np.asarray(depth[a], dtype=float)
        tr[a]["bootstrap"] = {"n_boot": int(n_boot), "primary_location_share": share,
                              "primary_depth": ({"mean": float(dep.mean()), "ci_2.5": float(np.percentile(dep, 2.5)),
                                                 "ci_97.5": float(np.percentile(dep, 97.5)), "n": int(len(dep))}
                                                if len(dep) else None)}
    tr["diagonal"]["bootstrap"]["share_4v4_below_3v3_and_5v5"] = float(np.mean(diag4))
    out["troughs"] = tr

    n_min = np.minimum(b, r)
    labelled = y >= 0
    cuts = {}
    for c in CUTS:
        cls = {"pick": ok & (n_min <= 1), "skirmish": ok & (n_min >= 2) & (n_min < c), "teamfight": ok & (n_min >= c),
               "unknown": ~ok}
        cd = class_draws[str(c)]
        cuts[str(c)] = {name: {"n": int(m.sum()), "share": float(m.mean()),
                               "share_ci": [float(np.percentile(cd[name], 2.5)), float(np.percentile(cd[name], 97.5))],
                               "positive_rate_labelled": (float(y[m & labelled].mean()) if (m & labelled).any() else None),
                               "median_engage_minute": (float(np.median(minute[m])) if m.any() else None)}
                        for name, m in cls.items()}
        cuts[str(c)]["block_mass_c_to_5"] = float((ok & (b >= c) & (r >= c)).mean())
        cuts[str(c)]["block_mass_c_to_5_ci"] = [float(np.percentile(cd["block_mass_c_to_5"], 2.5)),
                                                float(np.percentile(cd["block_mass_c_to_5"], 97.5))]
    out["cuts"] = cuts
    out["by_patch"] = {}
    for p in sorted(set(patch.tolist())):
        pm = patch == p
        out["by_patch"][str(p)] = {"n": int(pm.sum()), **{
            str(c): {"pick": float(np.mean(n_min[pm & ok] <= 1)),
                     "skirmish": float(np.mean((n_min[pm & ok] >= 2) & (n_min[pm & ok] < c))),
                     "teamfight": float(np.mean(n_min[pm & ok] >= c))} for c in CUTS}}
    return out


def v2_reference(path: Path) -> dict | None:
    if not path.exists():
        return None
    v2 = json.loads(path.read_text(encoding="utf-8"))
    ref = {"source": str(path), "n": v2.get("n"), "patches": v2.get("patches"),
           "note": "v2 corpus (G 18 s, D 4,000 u, R 1,800 u, B 10 s), recomputed from the stored joint matrices"}
    for pair, key in (("participation", "participation_joint_blue_red_0_5"), ("presence", "presence_joint_blue_red_0_5")):
        if key not in v2:
            continue
        d = distributions(np.asarray(v2[key], dtype=float).ravel())
        n_ok = float(d["joint"].sum())
        ref[pair] = {"diagonal_share_of_population": (d["diagonal"] / n_ok).round(6).tolist(),
                     "smaller_side_marginal": shares(d["smaller_side"]),
                     "troughs": {a: {"support": list(support_of(d[a])), "peaks": peaks(d[a], support_of(d[a])),
                                     "troughs": troughs(d[a], support_of(d[a]))}
                                 for a in ("diagonal", "smaller_side", "per_team", "total")},
                     "block_mass_c_to_5": {str(c): float(d["joint"][c:, c:].sum() / n_ok) for c in CUTS}}
    return ref


def main(argv=None) -> int:
    boot_mod = load_sibling("scale_cut_sensitivity_v33")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    ap.add_argument("--y-key", default="y_market_event", help="label defining the headline population (draws < 0)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--v2-json", type=Path, default=DEFAULT_V2)
    ap.add_argument("--negative-count-as", choices=("zero", "unknown"), default="zero",
                    help="reading of a -1 participation count (int(0 or -1) in data/index_split._fight_to_ref_row)")
    ap.add_argument("--output", type=Path, default=boot_mod.DEFAULT_OUT_DIR / "scale_participation_v33.json")
    args = ap.parse_args(argv)
    started = time.time()

    keys = [args.y_key, "groups", "patch", "engage_ts", "cluster_blue", "cluster_red", "present_blue", "present_red"]
    A = boot_mod.load_shard_arrays(args.shards, keys)
    y_all = A[args.y_key].astype(np.int64)
    populations = {"labelled": y_all >= 0, "all_rows": np.ones(len(y_all), dtype=bool)}
    res = {"item": boot_mod.ITEM,
           "what": "v3.3 participation / presence count distributions, troughs and class shares at cuts 3/4/5",
           "provenance": {**boot_mod.git_provenance(), "script": "scripts/scale_participation_v33.py",
                          "python": sys.executable, "corpus": boot_mod.corpus_provenance(args.shards),
                          "preset": "v3.3", "label_key": args.y_key, "split": "none (descriptive, whole corpus)",
                          "seed": int(args.seed), "n_boot": int(args.n_boot), "wall_clock_s": None},
           "definitions": {
               "n_min": "min(blue, red)", "classes": "pick n_min <= 1, skirmish 2..c-1, teamfight n_min >= c; "
                                                     "unknown = negative count on either side when read as unknown",
               "negative_counts": (f"participation -1 counts read as {args.negative_count_as}: gameplay/fights.py writes "
                                   "an int count and _fight_to_ref_row stores int(count or -1); presence counts are "
                                   ">= M by the gate and never negative"),
               "intervals": "match-clustered percentile bootstrap (same replicates as the troughs)",
               "trough": "interior k with p(k) < p(k-1) and p(k) < p(k+1) on the supported range; depth = "
                         "1 - p(k) / min(max left, max right); primary = deepest",
               "populations": {"labelled": f"{args.y_key} >= 0 (headline rows, draws dropped)",
                               "all_rows": "every detected engagement row stored in the shards"}},
           "populations": {}}
    for pop, sel in populations.items():
        idx = np.flatnonzero(sel)
        groups = A["groups"][idx]
        minute = A["engage_ts"][idx].astype(np.float64) / 60000.0
        block = {"n": int(len(idx)), "n_matches": int(len(np.unique(groups)))}
        for pair, (kb, kr) in PAIRS.items():
            t0 = time.time()
            block[pair] = analyse_pair(A[kb][idx].astype(np.int64), A[kr][idx].astype(np.int64), y_all[idx], groups,
                                       A["patch"][idx], minute, boot_mod, args.n_boot, args.seed,
                                       negative_count_as=(args.negative_count_as if pair == "participation" else "unknown"))
            print(f"[{pop}/{pair}] done in {time.time() - t0:.0f}s", flush=True)
        pm, pm_ok = boot_mod.smaller_side_count(A["present_blue"][idx], A["present_red"][idx], "unknown")
        qm, qm_ok = boot_mod.smaller_side_count(A["cluster_blue"][idx], A["cluster_red"][idx], args.negative_count_as)
        ct = np.zeros((6, 6), dtype=np.int64)
        okc = pm_ok & qm_ok
        np.add.at(ct, (pm[okc], qm[okc]), 1)
        block["crosstab_presence_nmin_rows_participation_nmin_cols"] = ct.tolist()
        res["populations"][pop] = block
    res["v2_reference"] = v2_reference(args.v2_json)
    res["deviations"] = [
        "Troughs are read from the empirical pmf without smoothing (the v2 reading was by eye from the same tables).",
        "The bootstrap resamples matches and keeps the trough support fixed at the full-data supported range.",
        f"Participation -1 counts are read as {args.negative_count_as}; the v2 joint matrices exclude negative counts, "
        "so the v2 side-by-side is on known counts only (42 labelled / 59 detected v3.3 rows are affected).",
    ]
    res["provenance"]["wall_clock_s"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=1), encoding="utf-8")

    for pop, block in res["populations"].items():
        print(f"\n== {pop}: n={block['n']:,} matches={block['n_matches']:,}")
        for pair in PAIRS:
            a = block[pair]
            dg = a["diagonal_share_of_population"]
            sm = [a["smaller_side_marginal"][str(k)]["share"] or 0.0 for k in range(6)]
            print(f"[{pair}] sentinel={a['sentinel_rows']} diagonal kvk % (0..5): " + " ".join(f"{v * 100:.2f}" for v in dg))
            print(f"[{pair}] smaller-side % (0..5): " + " ".join(f"{v * 100:.2f}" for v in sm))
            for an, t in a["troughs"].items():
                prim = t["primary"]
                bs = t["bootstrap"]["primary_location_share"]
                print(f"   trough[{an}] support={t['support']} peaks={t['peaks']} primary="
                      f"{(prim['k'], round(prim['depth'], 3)) if prim else None} bootstrap location share={bs}")
            print(f"   4v4 below 3v3 and 5v5 in {a['troughs']['diagonal']['bootstrap']['share_4v4_below_3v3_and_5v5'] * 100:.1f}% of replicates")
            for c in CUTS:
                cc = a["cuts"][str(c)]
                ci = lambda k: f"[{cc[k]['share_ci'][0] * 100:.1f}, {cc[k]['share_ci'][1] * 100:.1f}]"
                print(f"   cut {c}: pick {cc['pick']['share'] * 100:.1f}% {ci('pick')} skirmish {cc['skirmish']['share'] * 100:.1f}% "
                      f"{ci('skirmish')} teamfight {cc['teamfight']['share'] * 100:.1f}% {ci('teamfight')} "
                      f"unknown {cc['unknown']['share'] * 100:.3f}% block {cc['block_mass_c_to_5'] * 100:.1f}%")
    if res["v2_reference"] and "participation" in res["v2_reference"]:
        v2p = res["v2_reference"]["participation"]
        print("\n[v2 participation] diagonal %: " + " ".join(f"{v * 100:.2f}" for v in v2p["diagonal_share_of_population"])
              + f" | troughs {v2p['troughs']['diagonal']['troughs']}")
    print(f"\nwrote {args.output} ({res['provenance']['wall_clock_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
