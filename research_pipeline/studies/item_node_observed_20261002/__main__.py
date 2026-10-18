"""Build the recipient-confirmed observed-item bank."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _set_numeric_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def main(argv: list[str] | None = None) -> int:
    _set_numeric_threads()
    from .audit import DEFAULT_OUTPUT, run_bank
    from .bank import DEFAULT_PREPARATION_ROOT, DEFAULT_SOURCE_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION_ROOT)
    parser.add_argument("--max-matches", type=int, default=None, help="bounded smoke run; never accepting")
    parser.add_argument("--workers", type=int, default=0, help="ordered replay workers, 0 for the single-process smoke path (maximum 3)")
    parser.add_argument("--no-hash-verification", action="store_true", help="skip source archive comparisons for development only")
    parser.add_argument("--allow-partial", action="store_true", help="do not require full 209999/4384026/2618905 coverage")
    parser.add_argument("--disk-reserve-bytes", type=int, default=8 << 30)
    args = parser.parse_args(argv)
    manifest = run_bank(
        args.output,
        source_root=args.source,
        preparation_root=args.preparation,
        max_matches=args.max_matches,
        workers=args.workers,
        verify_hashes=not args.no_hash_verification,
        strict_full=not args.allow_partial and args.max_matches is None,
        disk_reserve_bytes=args.disk_reserve_bytes,
    )
    print({name: manifest.get(name) for name in ("complete", "accepted_bank", "full_corpus_verified", "counts", "elapsed_seconds")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
