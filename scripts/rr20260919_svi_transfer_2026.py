#!/usr/bin/env python3
"""Score frozen SVI winner on 2026 external/latest patches (no retrain).

Reads sealed EXT_* prediction NPZs from incremental_q_training_20260915.
Reports LightGBM / PT / logistic on T (and N appendix) for:
  EXT_KR_16.13, EXT_NA1_16.13, EXT_KR_16.15, EXT_KR_16.14_pilot

Writes outputs/svi_transfer_2026_20260919/
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]

COHORTS = (
    ("EXT_KR_16.13", "KR 16.13", "same-region newer patch", False),
    ("EXT_NA1_16.13", "NA1 16.13", "other region", False),
    ("EXT_KR_16.15", "KR 16.15", "newer KR", False),
    ("EXT_KR_16.14_pilot", "KR 16.14 pilot", "pilot only", True),
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


def metrics_from_npz(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    y = z["y"].astype(int)
    g = z["match"].astype(str)
    # equal match weight
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    w = (1.0 / c[inv]).astype(np.float64)
    out = dict(n=int(len(y)), n_matches=int(len(np.unique(g))))
    for name, col in (
        ("lgbm", "named__lgbm_winner"),
        ("pt", "named__pt_winner"),
        ("logit", "named__logit_winner"),
        ("p_pre", "p_pre"),
    ):
        if col not in z.files:
            continue
        p = z[col].astype(float)
        br = float(np.average((p - y) ** 2, weights=w))
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, p, sample_weight=w))
        except Exception:
            auc = None
        out[name] = dict(brier=br, auc=auc)
    if "lgbm" in out and "pt" in out:
        out["lgbm_minus_pt_brier"] = out["lgbm"]["brier"] - out["pt"]["brier"]
    return out


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# SVI transfer — 2026 external / latest patches")
    w("")
    w(f"Generated: {payload['generated']}")
    w("")
    w("Frozen winner from 210k selection (no retrain). Sealed `incremental_q` EXT predictions.")
    w("")
    w("## Teamfight T")
    w("")
    w("| Cohort | n | PT Brier | LGBM Brier | LGBM−PT | LGBM AUC | Pilot? |")
    w("|---|---:|---:|---:|---:|---:|---|")
    for key, label, _, pilot in COHORTS:
        r = payload["T"].get(key, {})
        w(f"| {label} | {r.get('n', 'NA')} | {_fmt((r.get('pt') or {}).get('brier'))} | "
          f"{_fmt((r.get('lgbm') or {}).get('brier'))} | {_fmt(r.get('lgbm_minus_pt_brier'))} | "
          f"{_fmt((r.get('lgbm') or {}).get('auc'), 4)} | {pilot} |")
    w("")
    w("## Other engagements N (appendix)")
    w("")
    w("| Cohort | n | PT Brier | LGBM Brier | LGBM−PT |")
    w("|---|---:|---:|---:|---:|")
    for key, label, _, pilot in COHORTS:
        r = payload["N"].get(key, {})
        w(f"| {label} | {r.get('n', 'NA')} | {_fmt((r.get('pt') or {}).get('brier'))} | "
          f"{_fmt((r.get('lgbm') or {}).get('brier'))} | {_fmt(r.get('lgbm_minus_pt_brier'))} |")
    w("")
    w("## Notes")
    w("")
    for t in payload["takeaways"]:
        w(f"- {t}")
    w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs" / "svi_transfer_2026_20260919")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    pred_dir = data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    T, N = {}, {}
    for key, label, role, pilot in COHORTS:
        for coh, bucket in (("T", T), ("N", N)):
            path = pred_dir / f"{key}_h90_{coh}.npz"
            print(f"score {path.name}…", flush=True)
            if not path.is_file():
                bucket[key] = dict(error="missing", path=str(path))
                continue
            bucket[key] = metrics_from_npz(path)
            bucket[key]["label"] = label
            bucket[key]["role"] = role
            bucket[key]["pilot"] = pilot

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        winner="LightGBM (incremental_q lgbm_winner, frozen from 210k)",
        T=T,
        N=N,
        takeaways=[
            "Transfer = score-only; selection stayed on 210k Q_SELECT.",
            "KR 16.14 pilot is not a headline cell.",
            "EUW1 has 0 complete pairs in the 2026 collection — omitted.",
            "Negative LGBM−PT means LightGBM better than PT (lower Brier).",
        ],
    )
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    summary = {k: T.get(k, {}).get("lgbm_minus_pt_brier") for k, *_ in COHORTS}
    print(json.dumps({"out_dir": str(out_dir), "lgbm_minus_pt_T": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
