"""Build corpus v3 and run the scale decomposition on it.

Corpus v3 is the settled definition (docs/DEFINITION_EVIDENCE.md section 17):

    G = 13.7 s   kill-interval valley           TF2_KILL_CLUSTER_GAP_MS = 13700
    D = 4,264 u  sharing-rate crossover         CLUSTER_MAX_DIAMETER    = 4264
    R = 1,600 u  champion-death XP share radius TF2_VALIDITY_RADIUS     = 1600
    B = 15 s     kill/assist credit window      TF2_ENGAGE_PRE_KILL_MS  = 15000
    M = 2        (unchanged)

with the ToG label ``market_lex`` (gold swing beyond a 300 g dead zone, refined by
cluster kills -> survivors -> structures, genuine draws dropped) over the label window
[cutoff, max(last kill, cutoff + 35 s)].  The 35 s horizon keeps the minimum aftermath
at 20 s after the first kill, as in the CoG corpus (10 s lead + 30 s horizon); with a
15 s lead that is cutoff + 35 s.

The corpus is built as shards (``scripts/build_corpus_shard.py``) with the detector
constants passed through ``LOL_CFG_OVERRIDES`` so every shard process sees them, then
``scripts/run_scale_decomposition.py`` fits the tabular baseline under match-grouped
folds with the v3 class cut (teamfight = smaller side >= 4).

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/build_corpus_v3.py \\
        --out-dir D:/LOL_Project/fusion_2615/corpus_shards_v3_mlex \\
        --output D:/LOL_Project/fusion_2615/features/scale_decomposition_v3_mlex.json

``--wait-for-glob/--wait-pattern`` lets the build start after another run finishes
(e.g. the runner-based v3 LightGBM run), so the two do not compete for CPUs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V3_DETECTOR = {
    "TF2_KILL_CLUSTER_GAP_MS": 13700,
    "CLUSTER_MAX_DIAMETER": 4264.0,
    "TF2_VALIDITY_RADIUS": 1600.0,
    "TF2_ENGAGE_PRE_KILL_MS": 15000,
    "FIGHT_HORIZON_SEC": 35,
}
V3_LABEL = {"label_type": "market_lex", "tie_policy": "drop", "gold_deadzone": 300.0}
V3_SCALE = {"pick_max": 1, "skirmish_min": 2, "teamfight_min": 4}


def wait_for(pattern_glob: str, pattern: str, timeout_s: float, poll_s: float = 60.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for path in glob.glob(pattern_glob):
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if pattern in text:
                return f"found {pattern!r} in {path}"
            if "Traceback" in text:
                return f"traceback in {path}; proceeding"
        time.sleep(poll_s)
    return "timeout; proceeding"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="scale-decomposition JSON")
    ap.add_argument("--num-shards", type=int, default=32)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-matches", type=int, default=None, help="sample size (default: every cached match)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--wait-for-glob", default=None)
    ap.add_argument("--wait-pattern", default="[DONE]")
    ap.add_argument("--wait-timeout-h", type=float, default=6.0)
    ap.add_argument("--skip-shards", action="store_true", help="shards already built; only decompose")
    ap.add_argument("--skip-decomposition", action="store_true")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    env["LOL_CFG_OVERRIDES"] = json.dumps(V3_DETECTOR)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LOKY_MAX_CPU_COUNT", "8")

    if args.wait_for_glob:
        print(f"waiting for {args.wait_pattern!r} in {args.wait_for_glob} ...", flush=True)
        print(wait_for(args.wait_for_glob, args.wait_pattern, args.wait_timeout_h * 3600), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"detector": V3_DETECTOR, "label": V3_LABEL, "scale": V3_SCALE, "seed": args.seed,
               "num_shards": args.num_shards, "n_matches": args.n_matches,
               "definition": "docs/DEFINITION_EVIDENCE.md section 17"},
              open(args.out_dir / "v3_definition.json", "w", encoding="utf-8"), indent=1)

    if not args.skip_shards:
        log = open(args.out_dir / "build.log", "a", encoding="utf-8")
        t0 = time.time()
        pending = list(range(args.num_shards))
        running: list = []
        while pending or running:
            while pending and len(running) < args.parallel:
                i = pending.pop(0)
                cmd = [sys.executable, "scripts/build_corpus_shard.py", "--shard", str(i), "--num-shards", str(args.num_shards),
                       "--seed", str(args.seed), "--label-type", V3_LABEL["label_type"], "--tie-policy", V3_LABEL["tie_policy"],
                       "--out-dir", str(args.out_dir)]
                if args.n_matches:
                    cmd += ["--n-matches", str(args.n_matches)]
                running.append((i, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)))
                print(f"[shard {i}] started ({len(running)} running, {len(pending)} pending)", flush=True)
            time.sleep(10)
            still = []
            for i, proc in running:
                rc = proc.poll()
                if rc is None:
                    still.append((i, proc))
                else:
                    print(f"[shard {i}] rc={rc} at {(time.time() - t0) / 60:.1f} min", flush=True)
            running = still
        log.close()
        print(f"shards done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not args.skip_decomposition:
        cmd = [sys.executable, "scripts/run_scale_decomposition.py", "--shards", str(args.out_dir), "--output", str(args.output),
               "--teamfight-min", str(V3_SCALE["teamfight_min"]), "--n-boot", str(args.n_boot)]
        print(" ".join(cmd), flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode
        print(f"decomposition rc={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
