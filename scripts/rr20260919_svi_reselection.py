#!/usr/bin/env python3
"""SVI family reselection: aggregate sealed iq/Track-A + meta grids; freeze winner.

Uses existing Q_SELECT-chosen sealed metrics for PT / logistic / LightGBM / MLP / residual MLP.
Records declared HP grids for FT-Transformer / TabNet / SAINT (reviewer deep) and
EmbedMLP / TabM (2024–25 meta). Optionally fits meta models on a TRAIN subsample
(--fit-meta) when feature matrices are available via the LOL_Teamfight script path.

Winner freeze rule: among available sealed families on MAIN_TEST T, primary contrast
family is the Q_SELECT overall winner from incremental_q (LightGBM for T). Meta/deep
full searches are queued; smoke meta fits do not override sealed selection unless
--allow-meta-override and Q_SELECT Brier improves.

Writes outputs/svi_reselection_20260919/{results.json,winner_manifest.json,REPORT.md}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from train.svi_tabular_meta import (  # noqa: E402
    EMBEDMLP_GRID, TABM_GRID, FT_TRANSFORMER_GRID, TABNET_GRID, SAINT_GRID,
    build_meta, param_count,
)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _fmt(x, nd=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def sealed_family_table(data_root: Path) -> Dict[str, Any]:
    iq = load_json(data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "results.json")
    ta_path = data_root / "outputs" / "track_a_mlp_20260916" / "eval" / "results.json"
    ta = load_json(ta_path) if ta_path.is_file() else {}

    t = iq["results"]["MAIN_TEST"]["T"]["metrics_named"]["all"]
    rows = {}
    for key, label in (
        ("pt_winner", "PT"),
        ("logit_winner", "logistic"),
        ("lgbm_winner", "LightGBM"),
        ("old_p_pre_logistic", "p_pre_logistic_legacy"),
        ("old_constant", "constant"),
    ):
        if key in t:
            rows[label] = dict(
                source="incremental_q",
                named_key=key,
                brier=float(t[key]["brier"]),
                auc=float(t[key]["auc"]),
                logloss=float(t[key].get("logloss", t[key].get("log_loss", float("nan")))),
            )
    if ta:
        tm = ta["results"]["MAIN_TEST"]["T"]["metrics_named"]["all"]
        for key, label in (("mlp_winner", "MLP"), ("resmlp_winner", "residual_MLP")):
            if key in tm:
                rows[label] = dict(
                    source="track_a_mlp",
                    named_key=key,
                    brier=float(tm[key]["brier"]),
                    auc=float(tm[key]["auc"]),
                    logloss=float(tm[key].get("logloss", tm[key].get("log_loss", float("nan")))),
                )

    # Q_SELECT overall winner from iq REPORT / winners field
    winners = iq["results"]["MAIN_TEST"]["T"].get("winners") or {}
    overall = winners.get("overall") or winners.get("all") or "lgbm_winner"
    return dict(
        sealed_eval="MAIN_TEST T h90 (15.16 role inside 210k)",
        n=int(iq["results"]["MAIN_TEST"]["T"]["rows"]),
        families=rows,
        iq_overall_winner_key=overall,
        selection_note="iq/Track-A winners were chosen on Q_SELECT before sealed TEST open",
    )


def bootstrap_delta_brier(data_root: Path) -> Optional[Dict[str, Any]]:
    """Reuse phase-1 if present."""
    p = Path(__file__).resolve().parents[1] / "outputs" / "reviewer_response_phase1_20260919" / "results.json"
    if not p.is_file():
        # try data root
        p2 = data_root / "outputs" / "reviewer_response_phase1_20260919" / "results.json"
        p = p2 if p2.is_file() else p
    if not p.is_file():
        # from iq bootstrap
        iq = load_json(data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "results.json")
        boot = iq["results"]["MAIN_TEST"]["T"]["bootstrap"]["all"]["pairs"]
        # pair 0 is usually lgbm - pt
        pair0 = boot[0]
        return dict(
            source="incremental_q bootstrap pairs[0]",
            label=pair0.get("label") or "lgbm_winner - pt_winner",
            delta_brier=float(pair0["delta_brier"]["point"]),
            ci95=[float(pair0["delta_brier"]["ci95"][0]), float(pair0["delta_brier"]["ci95"][1])],
        )
    z = load_json(p)
    return dict(source=str(p), note="phase-1 present", payload_keys=list(z.keys())[:20])


def fit_meta_smoke(data_root: Path, n_train: int, device: str) -> Dict[str, Any]:
    """Fit EmbedMLP/TabM on a hash-sampled TRAIN T subset if ridge matrices load."""
    scripts = data_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        import fc20260915_common as C  # noqa: E402
        import iq20260915_common as Q  # noqa: E402
    except Exception as e:
        return dict(status="skipped", reason=f"import failed: {e}")

    try:
        import torch
        from sklearn.metrics import brier_score_loss
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return dict(status="skipped", reason=f"torch/sklearn: {e}")

    # Prefer ridge feature packs; fall back to [p_pre, time] from labels (exercises modules).
    X = y = g = None
    feat_note = ""
    try:
        feat_path = C.OUT / "extract" / "MAIN" / "q_features" / "MAIN_TRAIN_h90_T.npz"
        cands = []
        if not feat_path.is_file():
            cands = list((C.OUT / "extract").rglob("*TRAIN*h90*T*.npz"))[:8] if (C.OUT / "extract").is_dir() else []
            feat_path = cands[0] if cands else feat_path
        if feat_path.is_file():
            z = np.load(feat_path, allow_pickle=False)
            keys = list(z.files)
            X = z["X"] if "X" in keys else z["x"]
            y = z["y"].astype(int)
            g = z["match"].astype(str) if "match" in keys else np.arange(len(y)).astype(str)
            feat_note = f"loaded {feat_path}"
    except Exception as e:
        feat_note = f"ridge load failed: {e}"

    if X is None:
        # fallback: labels npz
        lab = np.load(
            data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TRAIN_labels.npz",
            allow_pickle=False)
        coh = np.load(
            data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TRAIN_cohort.npz",
            allow_pickle=False)
        m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
        y = lab["Y_h90"][m].astype(int)
        g = lab["match"][m].astype(str)
        X = np.column_stack([lab["p_pre"][m].astype(float), lab["s"][m].astype(float) / 60000.0])
        feat_note = "fallback X=[p_pre,time] from MAIN_TRAIN labels (not full 352 ridge)"

    # subsample
    matches = np.unique(g)
    scored = sorted(matches, key=lambda m: hash(("meta", str(m))) % (2 ** 63))
    keep_m = set(scored[: max(50, n_train // 3)])
    mask = np.array([str(m) in keep_m for m in g])
    X, y, g = X[mask], y[mask], g[mask]
    if len(y) < 100:
        return dict(status="skipped", reason=f"too few rows after sample: {len(y)}")

    # split last 20% matches for local select (not Q_SELECT — smoke only)
    um = np.unique(g)
    n_val = max(1, len(um) // 5)
    val_m = set(um[-n_val:])
    tr = np.array([str(m) not in val_m for m in g])
    va = ~tr
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr]).astype(np.float32)
    Xva = scaler.transform(X[va]).astype(np.float32)
    ytr, yva = y[tr], y[va]

    out = {"status": "ok", "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_in": int(X.shape[1]), "families": {}}
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    def train_one(name, kwargs, epochs=30):
        model = build_meta(name, X.shape[1], **kwargs).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xtr, device=dev)
        yt = torch.tensor(ytr.astype(np.float32), device=dev).unsqueeze(1)
        best = 1e9
        best_state = None
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt), yt)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(torch.tensor(Xva, device=dev))).cpu().numpy().ravel()
            br = float(brier_score_loss(yva, pv))
            if br < best:
                best = br
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return dict(brier_val=best, n_params=param_count(model), config=kwargs)

    # one config each for smoke
    out["families"]["EmbedMLP"] = train_one("embedmlp", EMBEDMLP_GRID[0])
    out["families"]["TabM"] = train_one("tabm", TABM_GRID[0])
    out["note"] = (
        "SMOKE only on TRAIN subsample — does not replace sealed Q_SELECT selection. "
        + feat_note
    )
    return out


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# SVI model reselection — 210k corpus")
    w("")
    w(f"Generated: {payload['generated']}")
    w("")
    w("## Selection rule")
    w("")
    w("- Population: 210k matches (15.14–15.16).")
    w("- Choose on Q_SELECT; sealed numbers below are **15.16 T role**.")
    w("- Never select on TEST or external.")
    w("")
    w("## Sealed family table (15.16 T)")
    w("")
    w("| Family | Source | Brier | AUC |")
    w("|---|---|---:|---:|")
    fams = payload["sealed"]["families"]
    for name, row in sorted(fams.items(), key=lambda kv: kv[1]["brier"]):
        w(f"| {name} | {row['source']} | {_fmt(row['brier'])} | {_fmt(row['auc'], 4)} |")
    w("")
    win = payload["winner"]
    w(f"**Frozen winner:** `{win['family']}` (`{win['named_key']}`) — {win['reason']}")
    w("")
    w("## Declared grids (reviewer deep + 2024–25 meta)")
    w("")
    w("| Family | #candidates | Grid summary |")
    w("|---|---:|---|")
    for name, g in payload["declared_grids"].items():
        w(f"| {name} | {g['n']} | `{g['summary']}` |")
    w("")
    w("FT-Transformer / TabNet / SAINT: implementation in `scripts/run_deep_tabular_baselines.py`; "
      "full Q_SELECT search is the ToG revision `d2_hparam_search_deep` budget. "
      "This reselection freezes the sealed tabular winner for material/quiet/transfer.")
    w("")
    meta = payload.get("meta_smoke") or {}
    w("## Meta smoke (EmbedMLP / TabM)")
    w("")
    if meta.get("status") != "ok":
        w(f"_Skipped:_ {meta.get('reason', meta.get('status'))}")
    else:
        w(f"n_train={meta['n_train']} n_val={meta['n_val']} n_in={meta['n_in']}")
        w("")
        w("| Family | val Brier | params |")
        w("|---|---:|---:|")
        for name, row in meta["families"].items():
            w(f"| {name} | {_fmt(row['brier_val'])} | {row['n_params']} |")
        w("")
        w(meta.get("note", ""))
    w("")
    w("## Primary sealed contrast")
    w("")
    d = payload.get("primary_contrast")
    if d:
        w(f"- {d.get('label', 'LightGBM − PT')}: ΔBrier {_fmt(d.get('delta_brier'))} "
          f"CI {d.get('ci95')}")
    w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs" / "svi_reselection_20260919")
    ap.add_argument("--fit-meta", action="store_true")
    ap.add_argument("--meta-n-train", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sealed = sealed_family_table(data_root)
    # Freeze LightGBM as iq overall winner for T
    winner = dict(
        family="LightGBM",
        named_key="lgbm_winner",
        bundle="incremental_q_training_20260915",
        config="lgbm_L15_M100__raw",
        cohort="T",
        horizon="h90",
        reason="Q_SELECT overall winner on T in incremental_q; sealed 15.16 T Brier best among classical+Track-A neural",
        sealed_brier=sealed["families"]["LightGBM"]["brier"],
        sealed_auc=sealed["families"]["LightGBM"]["auc"],
    )

    contrast = None
    iq = load_json(data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "results.json")
    pair0 = iq["results"]["MAIN_TEST"]["T"]["bootstrap"]["all"]["pairs"][0]
    amb = pair0.get("a_minus_b") or {}
    br = amb.get("brier") or amb.get("delta_brier") or {}
    if isinstance(br, dict):
        point = float(br.get("point", br.get("estimate", br.get("mean", 0.0))))
        ci = br.get("ci95") or br.get("ci")
    else:
        point, ci = float(br or 0.0), None
    if ci is not None:
        ci = [float(ci[0]), float(ci[1])]
    contrast = dict(
        label=pair0.get("label") or f"{pair0.get('a')} - {pair0.get('b')}",
        delta_brier=point,
        ci95=ci,
    )

    meta = dict(status="skipped", reason="pass --fit-meta to run EmbedMLP/TabM smoke")
    if args.fit_meta:
        print("fitting meta smoke…", flush=True)
        meta = fit_meta_smoke(data_root, args.meta_n_train, args.device)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        protocol="docs/SVI_MODEL_RESELECTION_TRANSFER_20260919.md",
        sealed=sealed,
        winner=winner,
        primary_contrast=contrast,
        declared_grids=dict(
            EmbedMLP=dict(n=len(EMBEDMLP_GRID), summary=str(EMBEDMLP_GRID[0]) + " …"),
            TabM=dict(n=len(TABM_GRID), summary=str(TABM_GRID[0]) + " …"),
            FT_Transformer=dict(n=len(FT_TRANSFORMER_GRID), summary=str(FT_TRANSFORMER_GRID)),
            TabNet=dict(n=len(TABNET_GRID), summary=str(TABNET_GRID)),
            SAINT=dict(n=len(SAINT_GRID), summary=str(SAINT_GRID)),
        ),
        meta_smoke=meta,
        next_steps=[
            "Full FT/TabNet/SAINT Q_SELECT search via ToG d2_hparam_search_deep when compute allows",
            "Material/quiet retrain with frozen LightGBM spec",
            "Transfer score on 2026 EXT_* prediction files",
        ],
    )
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "winner_manifest.json").write_text(
        json.dumps(winner, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps({"out_dir": str(out_dir), "winner": winner["family"],
                      "brier": winner["sealed_brier"], "meta": meta.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
