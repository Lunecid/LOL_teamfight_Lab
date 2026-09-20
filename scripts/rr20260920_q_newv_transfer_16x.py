#!/usr/bin/env python3
"""Score-only transfer of frozen new-V q on EXT 16.x (KR/NA1) — no refit.

Corpus contract:
  - Fit/select: 210k KR 15.14–15.16 role split
  - Transfer: 16.x score-only after freeze (PAPER_COHORT_CONTRACT §4)

Uses:
  - frozen fit85 MLP V for EXT SVI labels (p_pre/p_post/Y)
  - frozen q from q_newv_fit85 selection_freeze.json (logit_state + PT/b_p)
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
QOUT = REPO / "outputs" / "q_newv_fit85_20260920"
OUT = QOUT / "transfer_16x"
ROLE = "TRANSFER_SCORE_ONLY_NO_REFIT"

COHORTS = (
    ("KR_16.13", "EXT_KR_16.13", "KR 16.13", False),
    ("NA1_16.13", "EXT_NA1_16.13", "NA1 16.13", False),
    ("KR_16.15", "EXT_KR_16.15", "KR 16.15", False),
    ("KR_16.14_pilot", "EXT_KR_16.14_pilot", "KR 16.14 pilot", True),
)

from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated  # noqa: E402


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics(y, p, g) -> Dict[str, Any]:
    w = match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(n=int(len(y)), n_matches=int(len(np.unique(g))), brier=br, logloss=ll, auc=auc)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def expanded_pre(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    return X[:, keep].astype(np.float64, copy=False), cols


def load_ext_t(data_root: Path, D, L, set_id: str, coh_name: str) -> Dict[str, Any]:
    E = D.load_engagements(L, set_id, None, states=True, counts=False)
    coh_path = (
        data_root
        / "outputs"
        / "cohort_role_training_20260915"
        / "cohorts"
        / f"{coh_name}_cohort.npz"
    )
    lab_keys = None
    if coh_path.is_file():
        coh = np.load(coh_path, allow_pickle=False)
        # prefer pairing with engagement keys
        if "match" in coh.files and "s" in coh.files:
            m = coh["cohort"] == 1
            lab_keys = set(
                zip(coh["match"][m].astype(str).tolist(), coh["s"][m].astype(np.int64).tolist())
            )
    em, es = E["match"].astype(str), E["s"].astype(np.int64)
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    if lab_keys is not None:
        keep &= np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    X, cols = expanded_pre(E["X_pre"][keep], list(E["names"]))
    return dict(
        X=X,
        cols=cols,
        X_pre_raw=E["X_pre"][keep],
        X_post=E["X_post_h90"][keep],
        match=em[keep],
        s=es[keep],
        tmin=es[keep].astype(float) / 60000.0,
        names=list(E["names"]),
    )


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    args = ap.parse_args(argv)

    data_root = _data_root()
    _setup(data_root)
    import joblib
    import fc20260915_data as D
    import pandas as pd

    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")
    freeze_path = QOUT / "selection_freeze.json"
    if not freeze_path.is_file():
        raise SystemExit(f"missing {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    winner = freeze["selected_q"]

    OUT.mkdir(parents=True, exist_ok=True)
    L = D.Layout(False)
    ev = load_evaluator(BUNDLE)

    # load frozen q models
    bp = joblib.load(QOUT / "models" / "bp.joblib")
    pt = joblib.load(QOUT / "models" / "pt.joblib")
    logit_pack = joblib.load(QOUT / "models" / "logit_state.joblib")
    lgbm_pack = joblib.load(QOUT / "models" / "lgbm_state.joblib")

    results = {}
    for set_id, coh_name, label, pilot in COHORTS:
        print(f"transfer {label}…", flush=True)
        pack = load_ext_t(data_root, D, L, set_id, coh_name)
        # labels from frozen V (score-only)
        p_pre = predict_calibrated(ev, pack["X_pre_raw"])
        p_post = predict_calibrated(ev, pack["X_post"])
        dV = p_post - p_pre
        y = (dV > 0).astype(np.int8)
        g = pack["match"]
        tmin = pack["tmin"]
        b40 = (p_pre >= 0.40) & (p_pre <= 0.60)

        num_ix = logit_pack["num_ix"]
        Xnum = np.column_stack([pack["X"][:, num_ix], p_pre.reshape(-1, 1)])
        pred = {
            "constant": np.full(len(y), float(np.mean(y))),  # EXT base rate diagnostic only — NOT used as primary
            "b_p": bp.predict_proba(p_pre.reshape(-1, 1))[:, 1],
            "PT": pt.predict_proba(np.column_stack([p_pre, tmin]))[:, 1],
            "logit_state": logit_pack["pipe"].predict_proba(Xnum)[:, 1],
        }
        # frozen constant from TRAIN rate if available
        # prefer TRAIN constant from primary_table
        primary = json.loads((QOUT / "primary_table.json").read_text(encoding="utf-8"))
        # recover train constant from TEST constant≈0.5; use selection models
        # overwrite constant with MAIN train rate ≈ from primary census not stored — use 0.5-ish from TRAIN meta
        train_meta = json.loads((QOUT / "labels" / "TRAIN_oof_h90_meta.json").read_text(encoding="utf-8"))
        pred["constant"] = np.full(len(y), float(train_meta.get("P_SVI", 0.5)))

        feat_cols = lgbm_pack["feat_cols"]
        champ_ix = lgbm_pack["champ_ix"]

        def make_df(X, p):
            df = pd.DataFrame(np.column_stack([X, p.reshape(-1, 1)]), columns=feat_cols)
            for j in champ_ix:
                df[feat_cols[j]] = df[feat_cols[j]].astype("category")
            return df

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred["lgbm_state"] = lgbm_pack["model"].predict_proba(make_df(pack["X"], p_pre))[:, 1]

        scores = {n: metrics(y, pred[n], g) for n in pred}
        b40_scores = {
            n: metrics(y[b40], pred[n][b40], g[b40]) for n in pred if int(b40.sum()) >= 50
        }
        d_brier = scores[winner]["brier"] - scores["PT"]["brier"]
        results[set_id] = dict(
            label=label,
            pilot=pilot,
            n=int(len(y)),
            n_matches=int(len(np.unique(g))),
            B40_n=int(b40.sum()),
            P_SVI=float(np.mean(y)),
            scores=scores,
            B40=b40_scores,
            selected_q=winner,
            delta_brier_q_minus_PT=float(d_brier),
        )
        print(
            f"  n={len(y)} P(SVI=1)={np.mean(y):.3f} "
            f"{winner} Brier={scores[winner]['brier']:.4f} PT={scores['PT']['brier']:.4f} "
            f"Δ={d_brier:.5f}",
            flush=True,
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        contract="docs/PAPER_COHORT_CONTRACT_20260919.md §4 + Q_PREDICTION_DESIGN_CONTRACT",
        note=(
            "210k KR 15.14–15.16 = fit/select/eval roles. "
            "16.x = score-only transfer; V/q/PT/b(p) frozen; no refit on EXT."
        ),
        frozen_v="A_MLP_expanded_fit85",
        frozen_q=winner,
        cohorts=results,
    )
    (OUT / "results.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Transfer — frozen new-V q on 16.x (score-only)",
        "",
        f"Generated: {payload['generated']}",
        f"**Frozen q:** `{winner}` · **Frozen V:** fit85 MLP Expanded",
        "",
        "Per [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md): "
        "**210k KR 15.14–15.16** for fit/select/eval; **16.x KR/NA1 = transfer only (no refit)**.",
        "",
        "## Teamfight T",
        "",
        "| Cohort | n | P(SVI=1) | q Brier | PT Brier | ΔBrier (q−PT) | q AUC | Pilot? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for set_id, _, label, pilot in COHORTS:
        r = results[set_id]
        sq = r["scores"][winner]
        sp = r["scores"]["PT"]
        lines.append(
            f"| {label} | {r['n']} | {fmt(r['P_SVI'], 3)} | {fmt(sq['brier'])} | {fmt(sp['brier'])} | "
            f"{fmt(r['delta_brier_q_minus_PT'], 5)} | {fmt(sq['auc'])} | {pilot} |"
        )
    lines += ["", "## B40 (new p_pre from frozen V)", ""]
    lines += ["| Cohort | B40 n | q Brier | PT Brier | q AUC |", "|---|---:|---:|---:|---:|"]
    for set_id, _, label, _ in COHORTS:
        r = results[set_id]
        if not r["B40"]:
            continue
        sq, sp = r["B40"][winner], r["B40"]["PT"]
        lines.append(
            f"| {label} | {r['B40_n']} | {fmt(sq['brier'])} | {fmt(sp['brier'])} | {fmt(sq['auc'])} |"
        )
    md = REPO / "docs" / "Q_NEWV_FIT85_TRANSFER_16X_20260920.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "results.json", md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
