import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from .repair import repair_run
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = repair_run(args.parent, args.output)
    print({k: result[k] for k in ("complete", "accepted", "counts", "elapsed_seconds")})
