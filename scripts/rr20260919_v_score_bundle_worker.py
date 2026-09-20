#!/usr/bin/env python3
"""Subprocess worker: score fixed X from evaluator_bundle ONLY (no TRAIN reload).

Usage:
  python rr20260919_v_score_bundle_worker.py --bundle PATH --X PATH.npy --out PATH.npy [--calibrated]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from v_redesign_evaluator_bundle import (  # noqa: E402
    load_evaluator,
    predict_calibrated,
    predict_raw_mlp,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--X", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()
    ev = load_evaluator(args.bundle)
    X = np.load(args.X)
    p = predict_calibrated(ev, X) if args.calibrated else predict_raw_mlp(ev, X)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, p.astype(np.float64))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
