"""Export the telemetry-only tabular baseline matrix for the fusion study.

Runs the released CoG pipeline's exact tabular path (``build_tabular_Xy``,
the paper's strongest representation) over every detected engagement in the
current cache, and saves X, y, feature names, and join keys
(match_id, engage_ts ms) so vision features extracted from the captured
pre-fight windows can be joined on the same engagements.

Environment must point at the fusion corpus before running:
    LOL_DETAIL_DIR / LOL_TIMELINE_DIR / LOL_OUTPUT_ROOT -> fusion_2615

Example:
    python scripts/export_fusion_baseline.py --output D:/LOL_Project/fusion_2615/features/telemetry_baseline.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import cfg  # noqa: E402
from data.index_split import build_fight_index  # noqa: E402
from train.baseline import build_tabular_Xy  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-set", default=str(getattr(cfg, "FEATURE_SET", "full")))
    args = parser.parse_args(argv)

    refs = build_fight_index()
    print(f"fight refs: {len(refs)}")
    X, y, feat_names, used = build_tabular_Xy(refs, feature_set=args.feature_set)
    print(f"X: {X.shape} | y: {y.shape} | positives: {int(y.sum())} ({y.mean():.3f})")
    keys = [
        {"match_id": r.match_id, "engage_ts_ms": int(r.t_start_ts), "patch": r.patch}
        for r in used
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y)
    sidecar = args.output.with_suffix(".keys.json")
    sidecar.write_text(
        json.dumps(
            {"feature_set": args.feature_set, "n_features": len(feat_names),
             "feature_names": feat_names, "keys": keys},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("wrote", args.output, "and", sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
