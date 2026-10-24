"""Run raw projection with a single numerical worker and immutable outputs."""
import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from .run import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-matches", type=int)
    args = parser.parse_args()
    result = build(args.output, max_matches=args.max_matches)
    print(json.dumps({key: result[key] for key in ("complete", "full_scope", "counts", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
