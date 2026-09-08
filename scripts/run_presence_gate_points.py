"""Re-run the LightGBM engagement baseline at alternative presence-gate points (R, B).

The presence gate (>= M alive champions per team within R of the first kill, B before it)
cannot be estimated from Match-V5, so its leverage on the *prediction* result is measured
directly: every point re-runs the released pipeline end to end (fight index -> patch holdout
split 15.14 / 15.15 / 15.16 -> 100k per split -> LightGBM baseline) with only
``TF2_VALIDITY_RADIUS`` and ``TF2_ENGAGE_PRE_KILL_MS`` changed, passed through
``LOL_CFG_OVERRIDES`` so the fight-index worker processes see them too.  All other detector
constants stay at the released values (G 18 s, D 4,000 u, M 2), so the comparison isolates
the gate.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_presence_gate_points.py \\
        --points ref_R1800_B10:1800:10000 rule_R1600_B15:1600:15000 --seed 7

Outputs: <OUTPUT_ROOT>/runs_presence_gate/<point>/run_*/models/lgbm/baseline/{report.json,
minute_report_test.json, situation_report_test.json, pred_*.csv}, a log per point next to
it, and manifest.json listing the points and return codes.  Summarise with
``scripts/summarize_presence_gate_points.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = ["ref_R1800_B10:1800:10000", "rule_R1600_B15:1600:15000"]


def parse_point(s: str):
    name, r, b = s.split(":")
    return name, float(r), int(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", default=DEFAULT_POINTS, help="name:R_units:B_ms")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-matches", type=int, default=210000)
    ap.add_argument("--workers", type=int, default=8, help="fight-index worker processes")
    ap.add_argument("--models", default="lgbm")
    ap.add_argument("--train-patches", default="15.14")
    ap.add_argument("--val-patches", default="15.15")
    ap.add_argument("--test-patches", default="15.16")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_root = Path(os.environ.get("LOL_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))
    group = out_root / "runs_presence_gate"
    group.mkdir(parents=True, exist_ok=True)
    manifest_path = group / "manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else {"points": {}}

    for spec in args.points:
        name, R, B = parse_point(spec)
        overrides = {"RUN_DIRNAME": f"runs_presence_gate/{name}", "TF2_VALIDITY_RADIUS": R,
                     "TF2_ENGAGE_PRE_KILL_MS": B, "FIGHT_INDEX_NUM_WORKERS": int(args.workers)}
        env = dict(os.environ)
        env["LOL_CFG_OVERRIDES"] = json.dumps(overrides)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        cmd = [sys.executable, "runner.py", "--mode", "train", "--max_matches", str(args.max_matches),
               "--seed", str(args.seed), "--split_mode", "patch_holdout",
               "--train_patches", args.train_patches, "--val_patches", args.val_patches,
               "--test_patches", args.test_patches, "--models", args.models]
        log_path = group / f"{name}.log"
        print(f"[{name}] R={R:.0f} u, B={B} ms -> {group / name}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} overrides={json.dumps(overrides)}\n")
            log.write("# " + " ".join(cmd) + "\n")
            log.flush()
            rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - t0
        manifest["points"][name] = {"R_u": R, "B_ms": B, "seed": args.seed, "max_matches": args.max_matches,
                                    "models": args.models, "returncode": rc, "elapsed_s": round(elapsed, 1),
                                    "run_dir": str(group / name), "log": str(log_path),
                                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1)
        print(f"[{name}] rc={rc} elapsed={elapsed / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
