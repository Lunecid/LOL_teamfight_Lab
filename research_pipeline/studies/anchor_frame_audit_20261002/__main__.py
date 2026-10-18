"""CLI entry: python -m studies.anchor_frame_audit_20261002"""
from __future__ import annotations

from . import thread_limits as TL  # noqa: F401  — before heavy imports in subcommands

import argparse
from pathlib import Path

from .run_audit import run
from .v_oof_audit import run_code_audit
from .write_report import write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Anchor frame audit 20261002")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--max-matches", type=int, default=None, help="Partial diagnostic only; complete=false")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--skip-corpus", action="store_true", help="Only code audit + report stub")
    args = p.parse_args(argv)

    if args.skip_corpus:
        from . import config as C

        run_dir = args.run_dir or C.new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        code_audit = run_code_audit(run_dir / "code_audit.json")
        print({"code_audit": str(run_dir / "code_audit.json"), "sha256": code_audit.get("sha256")})
        return 0

    man = run(run_dir=args.run_dir, max_matches=args.max_matches, workers=args.workers)
    code_audit = run_code_audit(Path(man["run_dir"]) / "code_audit.json")
    report = write_report(man, code_audit)
    print({"complete": man.get("complete"), "run_dir": man.get("run_dir"), "report": str(report)})
    if args.max_matches is not None:
        return 0
    return 0 if man.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
