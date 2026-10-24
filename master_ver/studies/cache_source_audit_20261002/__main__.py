"""CLI entry point for the cache source audit."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from .run_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
