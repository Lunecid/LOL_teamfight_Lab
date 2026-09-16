"""Evaluate a frozen objective-aware value model and engagement predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.state_value_experiment import run_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=250)
    a = ap.parse_args()
    if (a.out_dir / "results.json").exists():
        raise SystemExit("Results already exist; use a new output directory.")
    result = run_experiment(a.dataset, a.out_dir, folds=a.folds, trees=a.trees)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "value_validation", "engagement_test", "B_minus_A")}, indent=2))
