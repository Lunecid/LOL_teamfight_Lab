"""CLI: python -m studies.objective_timing_20261002"""
from __future__ import annotations

from . import thread_limits as TL  # noqa: F401

import argparse
from pathlib import Path

from .run import reaggregate_from, run
from .write_report import write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Objective timing study 20261002")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--max-matches", type=int, default=None, help="Smoke/partial only; complete=false")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--skip-report", action="store_true")
    p.add_argument(
        "--reaggregate-from",
        type=Path,
        default=None,
        help="Reuse immutable objective_events from a prior complete run",
    )
    p.add_argument(
        "--skip-source-schema",
        action="store_true",
        help="Skip full read-only events.json schema verification (not for acceptance)",
    )
    args = p.parse_args(argv)

    if args.reaggregate_from is not None:
        man = reaggregate_from(
            args.reaggregate_from,
            run_dir=args.run_dir,
            workers=args.workers,
            skip_source_schema=args.skip_source_schema,
        )
    else:
        man = run(run_dir=args.run_dir, max_matches=args.max_matches, workers=args.workers)
    report = None
    if not args.skip_report:
        report = write_report(man)
    print(
        {
            "complete": man.get("complete"),
            "run_dir": man.get("run_dir"),
            "report": str(report) if report else None,
        }
    )
    if args.max_matches is not None:
        return 0
    return 0 if man.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
