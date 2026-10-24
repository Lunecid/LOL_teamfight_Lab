import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from .run import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print({name: result[name] for name in ("complete", "accepted_preparation", "training_ready", "counts", "elapsed_seconds")})
