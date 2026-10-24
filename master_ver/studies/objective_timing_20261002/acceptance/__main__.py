"""CLI: python -m studies.objective_timing_20261002.acceptance"""
from __future__ import annotations

from . import thread_bootstrap  # noqa: F401 — set thread caps before numpy

import argparse
import sys
from pathlib import Path

from .pipeline import run_acceptance


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=(
            "Scientific acceptance for objective_timing_20261002. "
            "Does not rewrite extraction RUN_MANIFEST complete flags. "
            "Run before treating extracted statistics as accepted."
        )
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed objective_timing run directory (read-only)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="NEW verification output directory (must not already exist non-empty)",
    )
    p.add_argument(
        "--skip-raw-source",
        action="store_true",
        help="Skip full raw cache audit (not for official acceptance)",
    )
    args = p.parse_args(argv)
    try:
        result = run_acceptance(
            args.run_dir,
            args.out_dir,
            argv=["python", "-m", "studies.objective_timing_20261002.acceptance", *argv],
            skip_raw_source=bool(args.skip_raw_source),
        )
    except FileExistsError as exc:
        # Collision / non-empty out_dir: exit only; do not write there
        print({"error": "out_dir_collision", "detail": str(exc)}, flush=True)
        return 2
    except ValueError as exc:
        # Unsafe path (target run / descendant): exit only; do not write there
        print({"error": "unsafe_out_dir", "detail": str(exc)}, flush=True)
        return 2
    print(
        {
            "accepted": result.get("accepted"),
            "complete": result.get("complete"),
            "stage": result.get("stage"),
            "error": result.get("error"),
            "out_dir": result.get("out_dir"),
            "manifest": str(Path(result.get("out_dir") or args.out_dir) / "ACCEPTANCE_MANIFEST.json"),
        },
        flush=True,
    )
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
