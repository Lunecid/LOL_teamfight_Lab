"""Command line entry point for the item source audit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


def main() -> int:
    from .audit import DEFAULT_OUTPUT, run_audit
    from .source import DEFAULT_PREPARATION_ROOT, DEFAULT_SOURCE_ROOT

    parser = argparse.ArgumentParser(description="Audit the accepted source item transactions with the direct ledger.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION_ROOT)
    parser.add_argument("--max-matches", type=int, default=None,
                        help="bounded smoke run; the output is explicitly non-accepting")
    parser.add_argument("--no-hash-verification", action="store_true",
                        help="skip expected shard hash comparisons for local development only")
    parser.add_argument("--allow-partial", action="store_true",
                        help="do not require the 209999-match/4384026-query/2618905-row full-corpus gate")
    args = parser.parse_args()
    result = run_audit(
        args.output,
        source_root=args.source,
        preparation_root=args.preparation,
        max_matches=args.max_matches,
        verify_hashes=not args.no_hash_verification,
        strict_full=not args.allow_partial and args.max_matches is None,
    )
    print({name: result.get(name) for name in ("complete", "accepted_audit", "full_corpus_verified", "counts", "elapsed_seconds")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
