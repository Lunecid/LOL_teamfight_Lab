"""E2b/E2c driver.

README
  python -B studies/external_validation_20261009/e2bc_run.py e2b --dataset parity_15_16 --workers 2
  python -B studies/external_validation_20261009/e2bc_run.py e2b-parity             # E2b_PARITY_REPORT.json
  python -B studies/external_validation_20261009/e2bc_run.py e2c --dataset parity_15_16 --workers 2
  python -B studies/external_validation_20261009/e2bc_run.py e2c-parity             # E2c_PARITY_REPORT.json
  python -B studies/external_validation_20261009/e2bc_run.py e2b --dataset kr_15_18_22 --workers 6   (then kr_16_13, na1_16_13)
  python -B studies/external_validation_20261009/e2bc_run.py e2c --dataset kr_15_18_22 --workers 6
  python -B studies/external_validation_20261009/e2bc_run.py lock                   # EXTERNAL_SCORE_LOCK.json
  The external sets run only after both parity reports say PASS.  Worker counts are capped by
  e2bc_common.cpu_workers_allowed (2 until boundary_flags_1/RUN_FLAGS.log shows 'evaluate rc', then 6).
  Stage times: reports/external_validation_20261009/E2bc_STAGE_TIMES.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402


def _gate_external(dataset: str, which: str) -> None:
    if dataset == C.DATASET_PARITY:
        return
    for path, name in ((B.E2B_PARITY, "E2b"), (B.E2C_PARITY, "E2c")):
        if which == "e2b" and name == "E2c":
            continue
        if not path.exists() or C.read_json(path).get("status") != "PASS":
            raise SystemExit(f"BLOCKED: the {name} parity gate has not passed; external sets are not run")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("e2b", "e2c"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=B.EXTERNAL_DATASETS)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--device", default="cuda")
    sub.add_parser("e2b-parity")
    sub.add_parser("e2c-parity")
    sub.add_parser("lock")
    args = parser.parse_args(argv)
    B.env_runtime()
    if args.command == "e2b":
        from studies.external_validation_20261009 import e2b_labels

        _gate_external(args.dataset, "e2b")
        log = C.Log(C.LOGS / f"E2b_{args.dataset}.log")
        log(f"[run] e2b {args.dataset} workers={args.workers} device={args.device}")
        e2b_labels.run_e2b(args.dataset, args.workers, args.device, log)
    elif args.command == "e2b-parity":
        from studies.external_validation_20261009 import e2b_labels

        log = C.Log(C.LOGS / "E2b_parity.log")
        with B.Timer(C.DATASET_PARITY, "e2b_parity_compare", log):
            report = e2b_labels.parity_labels(log)
        return 0 if report["status"] == "PASS" else 2
    elif args.command == "e2c":
        from studies.external_validation_20261009 import e2c_scores

        _gate_external(args.dataset, "e2c")
        log = C.Log(C.LOGS / f"E2c_{args.dataset}.log")
        log(f"[run] e2c {args.dataset} workers={args.workers} device={args.device}")
        e2c_scores.run_e2c(args.dataset, args.workers, args.device, log)
    elif args.command == "e2c-parity":
        from studies.external_validation_20261009 import e2c_scores

        log = C.Log(C.LOGS / "E2c_parity.log")
        with B.Timer(C.DATASET_PARITY, "e2c_parity_compare", log):
            report = e2c_scores.parity_scores(log)
        return 0 if report["status"] == "PASS" else 2
    else:
        from studies.external_validation_20261009 import e2_lock

        log = C.Log(C.LOGS / "E2_lock.log")
        with B.Timer("all", "external_score_lock", log):
            e2_lock.write_lock(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
