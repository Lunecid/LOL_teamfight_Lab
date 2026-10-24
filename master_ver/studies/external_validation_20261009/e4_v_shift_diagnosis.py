"""External validation, step 4: why does the frozen final V break at 16.13?  (diagnosis; DECISION_LOG_KO.md 10-09 21:56)

No training and no new V inference: stored nominal V predictions, the projection inputs, the item bank and the frozen
V vocabularies only.  Sets: TEST 15.16 (main projection, reference), KR 15.18-15.22, KR 16.13, NA1 16.13; supported
nominal V rows; y = match winner; match weight 1/m.  Per 5-minute game-time bin:
  (A) model-free predictors of the winner: AUC of the blue-minus-red team total gold, XP, mean level, CS (lane + jungle)
      and team kills, beside V;
  (B) V: AUC, calibration slope / intercept, Brier, mean |logit|, mean predicted versus observed blue win rate;
  (C) input shift: standardized mean difference (pooled SD) of every numeric V input channel against TEST 15.16 on a
      fixed random subsample (node frame 46 fields averaged over the 10 players, team growth 6 fields as team mean and
      blue - red, current gold, team kills, objective counts and "category seen" shares, team KDA totals);
  (D) unseen tokens: V metrics on 16.13 rows split by static unseen tokens in the match (champion / runes / spells /
      bans) and unseen current-item entries in the row.
Writes reports/external_validation_20261009/V_SHIFT_DIAGNOSIS_20261009.json once.

Usage: python -B studies/external_validation_20261009/e4_v_shift_diagnosis.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

DATA = Path(r"D:\LOL_External_20261009")
MAIN_PROJ = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
V_TEST_PRED = Path(r"D:\LOL_Training_20261004\v_report_1\V_TEST_PRED_final")
OUT = ROOT / "reports/external_validation_20261009/V_SHIFT_DIAGNOSIS_20261009.json"
SETS = ("test_15_16", "kr_15_18_22", "kr_16_13", "na1_16_13")
BINS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 10_000))
SUBSAMPLE, SEED = 120_000, 20261009
TG_FIELDS = ("level_mean", "xp_sum", "curGold_sum", "totalGold_sum", "laneCS_sum", "jgCS_sum")
OBJECTIVES = ("dragon", "baron", "herald", "atakhan", "horde", "tower", "inhibitor", "plate", "dragon_soul")
TEST_ROLE = 5


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def bin_label(lo: float, hi: float) -> str:
    return f"[{lo},{hi})" if hi < 10_000 else f"[{lo},inf)"


def load_set(name: str) -> dict:
    if name == "test_15_16":
        proj = MAIN_PROJ
        rows = np.load(proj / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        matches = np.load(proj / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        mq = np.load(proj / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
        mi = np.asarray(rows["match_index"], dtype=np.int64)
        keep = (np.asarray(matches["role"])[mi] == TEST_ROLE) & np.asarray(rows["supported"], dtype=bool)
        q = np.asarray(mq["first_query"], dtype=np.int64)[mi[keep]] + np.asarray(rows["query_ref"], dtype=np.int64)[keep]
        order = np.argsort(q, kind="stable")
        q, m = q[order], mi[keep][order]
        parts = sorted(V_TEST_PRED.glob("part_*.npz"))
        pieces = [np.load(p, allow_pickle=False) for p in parts]
        qi = np.concatenate([p["query_index"] for p in pieces])
        if not np.array_equal(qi, q):
            raise RuntimeError("TEST nominal predictions do not cover the nominal query list")
        raw_p = np.concatenate([p["raw_p"] for p in pieces]).astype(np.float64)
        logit = np.concatenate([p["logit"] for p in pieces]).astype(np.float64)
        items = None
    else:
        proj = DATA / "projection" / name
        pred = np.load(DATA / "labels" / name / "PRED_nominal_final.npz", allow_pickle=False)
        q = np.asarray(pred["query_index"], dtype=np.int64)
        m = np.asarray(pred["match_index"], dtype=np.int64)
        raw_p = np.asarray(pred["raw_p"], dtype=np.float64)
        logit = np.asarray(pred["logit"], dtype=np.float64)
        matches = np.load(proj / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        items = DATA / "items" / name
    y = np.asarray(matches["target_Blue_win"], dtype=np.float64)[m]
    minutes = np.asarray(np.load(proj / "ELAPSED_MINUTES.npy", mmap_mode="r"), dtype=np.float64)[q]
    return {"name": name, "proj": proj, "items": items, "q": q, "m": m, "y": y, "p": raw_p, "logit": logit, "minutes": minutes}


def weighted_auc(y: np.ndarray, score: np.ndarray, match: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(score)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
        return None
    _, inverse, counts = np.unique(match[ok], return_inverse=True, return_counts=True)
    return float(roc_auc_score(y[ok], score[ok], sample_weight=1.0 / counts[inverse]))


def predictors(s: dict) -> dict[str, np.ndarray]:
    tg = np.asarray(np.load(s["proj"] / "TEAM_GROWTH.npy", mmap_mode="r")[s["q"]], dtype=np.float64)
    tk = np.asarray(np.load(s["proj"] / "TEAM_KILLS.npy", mmap_mode="r")[s["q"]], dtype=np.float64)
    return {"gold_diff": tg[:, 0, 3] - tg[:, 1, 3], "xp_diff": tg[:, 0, 1] - tg[:, 1, 1], "level_diff": tg[:, 0, 0] - tg[:, 1, 0],
            "cs_diff": (tg[:, 0, 4] + tg[:, 0, 5]) - (tg[:, 1, 4] + tg[:, 1, 5]), "kill_diff": tk[:, 0] - tk[:, 1]}


def part_ab(s: dict) -> dict:
    from studies.q_production_20261005 import qeval

    preds = predictors(s)
    out = {}
    for lo, hi in BINS:
        mask = (s["minutes"] >= lo) & (s["minutes"] < hi)
        if mask.sum() < 500:
            continue
        y, m, p, lg = s["y"][mask], s["m"][mask], s["p"][mask], s["logit"][mask]
        ev = qeval.evaluate(y, p, m)["match_weighted"]
        w = qeval.match_weights(m)
        out[bin_label(lo, hi)] = {
            "rows": int(mask.sum()), "matches": int(len(np.unique(m))),
            "simple_auc": {k: weighted_auc(y, v[mask], m) for k, v in preds.items()},
            "v": {"auc": ev["auc"], "brier": ev["brier"], "calibration_slope": ev["calibration_slope"],
                  "calibration_intercept": ev["calibration_intercept"], "mean_abs_logit": float(np.average(np.abs(lg), weights=w)),
                  "mean_p": float(np.average(p, weights=w)), "blue_win_rate": float(np.average(y, weights=w))}}
    return out


def channels(s: dict, rows: np.ndarray) -> dict[str, np.ndarray]:
    proj, q = s["proj"], s["q"][rows]
    fields = [f["name"] for f in json.loads((MAIN_PROJ / "FIELD_MANIFEST.json").read_text(encoding="utf-8"))["node_frame_fields"]]
    node = np.asarray(np.load(proj / "NODE_FRAME.npy", mmap_mode="r")[q], dtype=np.float64)          # (n, 10, 46)
    out = {f"node:{name}": np.nanmean(node[:, :, i], axis=1) for i, name in enumerate(fields)}
    tg = np.asarray(np.load(proj / "TEAM_GROWTH.npy", mmap_mode="r")[q], dtype=np.float64)
    for i, name in enumerate(TG_FIELDS):
        out[f"team_mean:{name}"] = (tg[:, 0, i] + tg[:, 1, i]) / 2
        out[f"team_diff:{name}"] = tg[:, 0, i] - tg[:, 1, i]
    cg = np.asarray(np.load(proj / "CACHED_GOLD.npy", mmap_mode="r")[q], dtype=np.float64)
    out["cached_gold:mean"], out["cached_gold:diff"] = (cg[:, 0] + cg[:, 1]) / 2, cg[:, 0] - cg[:, 1]
    tk = np.asarray(np.load(proj / "TEAM_KILLS.npy", mmap_mode="r")[q], dtype=np.float64)
    out["team_kills:total"], out["team_kills:diff"] = tk.sum(axis=1), tk[:, 0] - tk[:, 1]
    oc = np.asarray(np.load(proj / "OBJECTIVE_COUNTS.npy", mmap_mode="r")[q], dtype=np.float64)
    for i, name in enumerate(OBJECTIVES):
        out[f"objective_count:{name}"] = oc[:, i]
        out[f"objective_seen:{name}"] = (oc[:, i] > 0).astype(np.float64)
    kda = np.asarray(np.load(proj / "PLAYER_KDA.npy", mmap_mode="r")[q], dtype=np.float64)
    for i, name in enumerate(("kills", "deaths", "assists")):
        out[f"kda_total:{name}"] = kda[:, :, i].sum(axis=1)
    out["elapsed_minutes"] = s["minutes"][rows]
    return out


def smd(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100:
        return None
    sd = np.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / sd) if sd > 0 else (0.0 if a.mean() == b.mean() else None)


def part_c(sets: dict) -> dict:
    rng = np.random.default_rng(SEED)
    sampled = {}
    for name, s in sets.items():
        rows = np.sort(rng.choice(len(s["q"]), size=min(SUBSAMPLE, len(s["q"])), replace=False))
        sampled[name] = (rows, channels(s, rows))
    out = {}
    ref_rows, ref = sampled["test_15_16"]
    ref_minutes = sets["test_15_16"]["minutes"][ref_rows]
    for name in SETS[1:]:
        rows, ch = sampled[name]
        minutes = sets[name]["minutes"][rows]
        out[name] = {}
        for lo, hi in BINS[:6]:
            a_mask, b_mask = (ref_minutes >= lo) & (ref_minutes < hi), (minutes >= lo) & (minutes < hi)
            values = {k: smd(ref[k][a_mask], ch[k][b_mask]) for k in ch}
            ranked = sorted(((k, v) for k, v in values.items() if v is not None), key=lambda kv: -abs(kv[1]))
            out[name][bin_label(lo, hi)] = {"rows_reference": int(a_mask.sum()), "rows": int(b_mask.sum()),
                                            "top20": [{"channel": k, "smd": round(v, 4)} for k, v in ranked[:20]],
                                            "all": {k: (round(v, 4) if v is not None else None) for k, v in values.items()},
                                            "channels_abs_smd_gt_0.2": int(sum(abs(v) > 0.2 for _, v in ranked))}
    presence = {}
    for name, (rows, ch) in sampled.items():
        presence[name] = {o: float(ch[f"objective_seen:{o}"][sets[name]["minutes"][rows] >= 25].mean()) for o in OBJECTIVES}
    return {"subsample_per_set": SUBSAMPLE, "seed": SEED, "smd_vs_test_15_16": out, "objective_seen_share_at_25min_or_later": presence}


def static_unseen_per_match(s: dict, vocab: dict) -> dict[str, np.ndarray]:
    from studies.observable_state_20261002.schema import STATIC_NAMES

    static = np.load(s["proj"] / "STATIC_IDS.npy")
    bans = np.load(s["proj"] / "BANS.npy")
    flags = {}
    for index, name in enumerate(STATIC_NAMES):
        values = static[:, :, index]
        known = np.isin(values, np.asarray(sorted(v for v in vocab[f"static:{name}"] if isinstance(v, int)), dtype=np.int64))
        flags[name] = (~known & (values >= 0)).any(axis=1)
    b = bans.reshape(len(bans), -1)
    known = np.isin(b, np.asarray(sorted(v for v in vocab["static:champion_id"] if isinstance(v, int)), dtype=np.int64))
    flags["bans"] = (~known & (b >= 0)).any(axis=1)
    return flags


def item_unseen_per_query(s: dict, vocab: dict) -> np.ndarray:
    offsets = np.load(s["items"] / "NODE_OFFSETS.npy")
    ids = np.fromfile(s["items"] / "ITEM_IDS.bin", dtype="<i8")
    total_q = len(offsets)
    node_of = np.searchsorted(offsets[:, 1:].reshape(-1), np.arange(len(ids), dtype=np.int64), side="right")
    element_query = node_of // 10
    item_vocab = np.asarray(sorted(v for v in vocab["item"] if isinstance(v, int)), dtype=np.int64)
    unseen = ~np.isin(ids, item_vocab) & (ids >= 0)
    return np.bincount(element_query[unseen], minlength=total_q)


def part_d(sets: dict) -> dict:
    from studies.external_validation_20261009.e2a_unk import _vocab_sets
    from studies.observable_state_20261002.schema import STATIC_NAMES
    from studies.q_production_20261005 import qeval

    vocab = _vocab_sets()
    out = {}
    for name in SETS[1:]:
        s = sets[name]
        flags = static_unseen_per_match(s, vocab)
        any_static = np.zeros(len(flags["bans"]), dtype=bool)
        for f in flags.values():
            any_static |= f
        row_static = any_static[s["m"]]
        row_champion = flags["champion_id"][s["m"]] if "champion_id" in flags else np.zeros(len(s["m"]), dtype=bool)
        row_item = item_unseen_per_query(s, vocab)[s["q"]] > 0

        def metrics(mask):
            if mask.sum() < 500:
                return {"rows": int(mask.sum())}
            ev = qeval.evaluate(s["y"][mask], s["p"][mask], s["m"][mask])["match_weighted"]
            return {"rows": int(mask.sum()), "matches": int(len(np.unique(s["m"][mask]))), "auc": ev["auc"],
                    "calibration_slope": ev["calibration_slope"], "brier": ev["brier"]}
        out[name] = {"matches_with_static_unseen_share": float(any_static.mean()),
                     "static_unseen_share_by_field": {k: float(v.mean()) for k, v in flags.items() if v.any()},
                     "rows_with_item_unseen_share": float(row_item.mean()),
                     "groups": {"no_static_no_item": metrics(~row_static & ~row_item), "static_only": metrics(row_static & ~row_item),
                                "item_only": metrics(~row_static & row_item), "both": metrics(row_static & row_item),
                                "no_champion_unseen": metrics(~row_champion), "champion_unseen": metrics(row_champion)}}
        if not STATIC_NAMES:
            raise RuntimeError("static field names missing")
    return out


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    sets = {name: load_set(name) for name in SETS}
    report = {"kind": "V_SHIFT_DIAGNOSIS", "utc": datetime.now(timezone.utc).isoformat(), "sets": {}, "code_sha256": sha_file(Path(__file__))}
    for name, s in sets.items():
        report["sets"][name] = {"rows": int(len(s["q"])), "matches": int(len(np.unique(s["m"]))), "by_bin": part_ab(s)}
    report["input_shift"] = part_c(sets)
    report["unseen_tokens"] = part_d(sets)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, r in report["sets"].items():
        print("==", name, r["rows"])
        for b, e in r["by_bin"].items():
            sa = {k: (round(v, 3) if v is not None else None) for k, v in e["simple_auc"].items()}
            v = e["v"]
            print(f"  {b:9s} n={e['rows']:7d} V auc {v['auc']:.3f} slope {v['calibration_slope']:.2f} |logit| {v['mean_abs_logit']:.2f} "
                  f"p {v['mean_p']:.3f} y {v['blue_win_rate']:.3f} | simple {sa}")
    for name, bins in report["input_shift"]["smd_vs_test_15_16"].items():
        for b in ("[5,10)", "[15,20)"):
            e = bins.get(b)
            if e:
                print("  shift", name, b, ">0.2:", e["channels_abs_smd_gt_0.2"], [(t["channel"], t["smd"]) for t in e["top20"][:8]])
    print("  objective seen at >=25 min", json.dumps(report["input_shift"]["objective_seen_share_at_25min_or_later"]))
    for name, d in report["unseen_tokens"].items():
        print("  unseen", name, round(d["matches_with_static_unseen_share"], 3), round(d["rows_with_item_unseen_share"], 3),
              {g: (v.get("rows"), round(v["auc"], 3) if "auc" in v else None, round(v["calibration_slope"], 2) if "calibration_slope" in v else None)
               for g, v in d["groups"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
