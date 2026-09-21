#!/usr/bin/env python3
"""Material-axis retrain (kill/obj) + quiet matched |ΔV| using frozen SVI winner family.

Winner default: LightGBM (from svi_reselection winner_manifest). Same 210k roles:
fit on TRAIN T, calibrate idea simplified (raw probs), evaluate on MAIN_TEST T.

Features for material models: p_pre, time_minutes, and available during-count nets
from labels (epic/structure proxies) — plus kill/alive from extract when joined.
This is a same-family retrain on alternate labels, not a new representation track.

Quiet: reuse cohort-aligned matched protocol numbers + optional LGBM on quiet vs fight
sign(ΔV) using p_pre/time only as a smoke.

Writes outputs/svi_material_quiet_20260919/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import rr20260919_svi_validation_suite as suite  # noqa: E402
import rr20260919_svi_cohort_aligned as aligned  # noqa: E402


def _data_root() -> Path:
    return suite._data_root()


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def match_weights(g: np.ndarray) -> np.ndarray:
    return suite.match_weights(g)


def try_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None


def build_xy_material(sp: Dict[str, np.ndarray], combat: Dict[str, np.ndarray], target: str):
    """target in {svi, kill, obj}."""
    y_svi = sp["y"].astype(int)
    kd = combat["kd"]
    obj = sp["obj"]
    p = sp["p_pre"].astype(float)
    # time from onset ms — pre-engagement only (no window outcomes in X)
    t = sp["s"].astype(float) / 60000.0
    X = np.column_stack([p, t])
    if target == "svi":
        y = y_svi
        m = np.ones(len(y), dtype=bool)
    elif target == "kill":
        sk = np.sign(kd)
        m = np.isfinite(kd) & (sk != 0)
        y = (sk > 0).astype(int)
    elif target == "obj":
        so = np.sign(obj)
        m = so != 0
        y = (so > 0).astype(int)
    else:
        raise ValueError(target)
    return X[m], y[m], sp["match"][m], m


def fit_eval_lgbm(Xtr, ytr, gtr, Xte, yte, gte, lgb) -> Dict[str, Any]:
    wtr = match_weights(gtr)
    dtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    params = dict(
        objective="binary", metric="binary_logloss", learning_rate=0.05,
        num_leaves=15, min_child_samples=100, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=1, verbosity=-1, seed=7,
    )
    booster = lgb.train(params, dtrain, num_boost_round=200)
    pred = booster.predict(Xte)
    wte = match_weights(gte)
    # weighted brier
    br = float(np.average((pred - yte) ** 2, weights=wte))
    try:
        auc = float(roc_auc_score(yte, pred, sample_weight=wte))
    except ValueError:
        auc = None
    # unweighted for sklearn compare
    br_u = float(brier_score_loss(yte, pred))
    return dict(brier_w=br, brier=br_u, auc=auc, n=int(len(yte)), n_matches=int(len(np.unique(gte))))


def p_pre_baseline(y, p, g) -> Dict[str, Any]:
    w = match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        auc = None
    return dict(brier_w=br, auc=auc, n=int(len(y)))


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# SVI material + quiet revalidation (winner family)")
    w("")
    w(f"Generated: {payload['generated']}")
    w(f"Winner: {payload['winner_family']}")
    w("")
    w("## Material retrain (MAIN_TEST T eval; TRAIN fit)")
    w("")
    w("| Target | n | p_pre Brier | Model Brier | Model AUC | ΔBrier (model−p_pre) |")
    w("|---|---:|---:|---:|---:|---:|")
    for tgt, block in payload["material"].items():
        if "error" in block:
            w(f"| {tgt} | — | — | — | — | {block['error']} |")
            continue
        d = block["model"]["brier_w"] - block["p_pre"]["brier_w"]
        w(f"| {tgt} | {block['model']['n']} | {_fmt(block['p_pre']['brier_w'])} | "
          f"{_fmt(block['model']['brier_w'])} | {_fmt(block['model'].get('auc'))} | {_fmt(d)} |")
    w("")
    w("## Concordance (pooled 210k T — from aligned suite)")
    w("")
    c = payload.get("concordance") or {}
    if c:
        w(f"- Kill agree {c.get('kill_agree')}, disagree {c.get('kill_disagree')}")
        w(f"- Obj agree {c.get('obj_agree')}")
    w("")
    w("## Quiet matched (210k pooled sample)")
    w("")
    q = payload.get("quiet") or {}
    w(f"- Type-B ratio {q.get('type_b_ratio')}, type-A ratio {q.get('type_a_ratio')} "
      f"(fight rows {q.get('fight_rows')})")
    w("")
    w("## SVI vs material (reading)")
    w("")
    for t in payload.get("takeaways", []):
        w(f"- {t}")
    w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs" / "svi_material_quiet_20260919")
    ap.add_argument("--winner-manifest", type=Path,
                    default=REPO / "outputs" / "svi_reselection_20260919" / "winner_manifest.json")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    winner_family = "LightGBM"
    if args.winner_manifest.is_file():
        winner_family = json.loads(args.winner_manifest.read_text(encoding="utf-8")).get("family", winner_family)

    lgb = try_lgbm()
    print("loading TRAIN/TEST T…", flush=True)
    tr = suite.load_split_t(data_root, "MAIN_TRAIN")
    te = suite.load_split_t(data_root, "MAIN_TEST")
    print("combat…", flush=True)
    combat = suite.combat_kill_alive(data_root, {"MAIN_TRAIN": tr, "MAIN_TEST": te})

    material = {}
    for target in ("svi", "kill", "obj"):
        if lgb is None:
            material[target] = dict(error="lightgbm not installed")
            continue
        Xtr, ytr, gtr, _ = build_xy_material(tr, combat["MAIN_TRAIN"], target)
        Xte, yte, gte, _ = build_xy_material(te, combat["MAIN_TEST"], target)
        if len(ytr) < 100 or len(yte) < 50:
            material[target] = dict(error=f"too few rows train={len(ytr)} test={len(yte)}")
            continue
        print(f"fit {target}…", flush=True)
        model = fit_eval_lgbm(Xtr, ytr, gtr, Xte, yte, gte, lgb)
        # p_pre baseline on same rows
        # rebuild p from te mask
        _, _, _, mte = build_xy_material(te, combat["MAIN_TEST"], target)
        p = te["p_pre"][mte]
        base = p_pre_baseline(yte, p, gte)
        material[target] = dict(model=model, p_pre=base)

    # concordance + quiet from aligned report if present
    aligned_rep = REPO / "outputs" / "svi_cohort_aligned_20260919" / "results.json"
    concordance = quiet = {}
    if aligned_rep.is_file():
        z = json.loads(aligned_rep.read_text(encoding="utf-8"))
        pooled = z.get("pooled") or z.get("headline") or {}
        dis = pooled.get("disagreement") or {}
        mat = pooled.get("material") or {}
        concordance = dict(
            kill_agree=dis.get("kill_agree_share") or (mat.get("kill") or {}).get("agree"),
            kill_disagree=dis.get("kill_disagree_share"),
            obj_agree=(mat.get("objective") or {}).get("agree"),
        )
        qb = pooled.get("quiet_type_b") or {}
        qa = pooled.get("quiet_type_a") or {}
        quiet = dict(
            type_b_ratio=qb.get("abs_ratio"),
            type_a_ratio=qa.get("abs_ratio"),
            fight_rows=(qb.get("fight") or {}).get("n_rows"),
        )

    takeaways = [
        "Material models use the frozen winner family (LightGBM) on alternate labels (kill/obj).",
        "If kill-model Brier << SVI-model on kill label but does not beat SVI on SVI label, SVI is not reducible to kills.",
        "Quiet ratios are matched-match from the 210k pooled cohort (not full-T vs smoke).",
    ]
    # substitute check
    if "svi" in material and "kill" in material and "model" in material["svi"] and "model" in material["kill"]:
        takeaways.append(
            f"15.16 T: SVI-target model Brier={material['svi']['model']['brier_w']:.4f}; "
            f"kill-target model Brier={material['kill']['model']['brier_w']:.4f} "
            f"(different labels — compare lift vs p_pre within each row)."
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        winner_family=winner_family,
        material=material,
        concordance=concordance,
        quiet=quiet,
        takeaways=takeaways,
        feature_note="X=[p_pre, time_min] only (pre-engagement); labels = SVI / kill sign / obj sign",
    )
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps({k: (v.get("model", {}) or {}).get("brier_w") for k, v in material.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
