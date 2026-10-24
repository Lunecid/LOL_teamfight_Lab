"""CLI for the full TRAIN population adoption projection."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config as C
from .projection import project_partition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=C.DEFAULT_SOURCE_RUN)
    parser.add_argument("--contract", type=Path, default=C.CONTRACT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-artifact-hash", action="store_true", help="use recorded hashes without rereading artifact bytes")
    args = parser.parse_args(argv)
    manifest = project_partition(
        args.source_run,
        contract_path=args.contract,
        output_dir=args.output,
        hash_artifacts=not args.skip_artifact_hash,
    )
    print({"output": str(args.output.resolve()), "rows": manifest["counts"]["n_rows"], "projection_sha256": manifest["projection_sha256"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
