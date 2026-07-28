from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_SCRIPT = PROJECT_ROOT / "scripts" / "collect_current_season.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect an exact patch to a fixed quota for sequential regions"
    )
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exact-api-patch", required=True)
    parser.add_argument("--platforms", nargs="+", default=["kr", "na1", "euw1"])
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["CHALLENGER", "GRANDMASTER", "MASTER"],
        choices=["CHALLENGER", "GRANDMASTER", "MASTER"],
    )
    parser.add_argument("--target-per-platform", type=int, default=10_000)
    parser.add_argument("--kr-state-dir", default=None)
    parser.add_argument("--max-storage-gb", type=float, default=15.0)
    parser.add_argument("--min-free-gb", type=float, default=150.0)
    parser.add_argument("--key-refresh-command", default=os.getenv("RIOT_KEY_REFRESH_COMMAND"))
    return parser


def state_dir_for(output_root: Path, platform: str, patch: str, kr_state_dir: Optional[str]) -> Path:
    if platform.lower() == "kr" and kr_state_dir:
        return Path(kr_state_dir).expanduser().resolve()
    tag = patch.replace(".", "_")
    return output_root / f"_collector_state_{platform.lower()}_{tag}"


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_per_platform < 1:
        raise SystemExit("--target-per-platform must be positive")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan_dir = output_root / f"_region_plan_{args.exact_api_patch.replace('.', '_')}"
    status_path = plan_dir / "status.json"

    for platform in args.platforms:
        platform = platform.lower()
        state_dir = state_dir_for(
            output_root, platform, args.exact_api_patch, args.kr_state_dir
        )
        write_status(
            status_path,
            {
                "state": "RUNNING",
                "platform": platform,
                "patch": args.exact_api_patch,
                "target": args.target_per_platform,
                "updated_at": int(time.time()),
            },
        )
        command = [
            sys.executable,
            str(COLLECTOR_SCRIPT),
            "--key-file",
            str(Path(args.key_file).expanduser().resolve()),
            "--output-root",
            str(output_root),
            "--state-dir",
            str(state_dir),
            "--platform",
            platform,
            "--tiers",
            *args.tiers,
            "--min-api-patch",
            args.exact_api_patch,
            "--exact-api-patch",
            args.exact_api_patch,
            "--max-complete-matches",
            str(args.target_per_platform),
            "--cycle-seconds",
            "600",
            "--work-budget-seconds",
            "540",
            "--max-storage-gb",
            str(args.max_storage_gb),
            "--min-free-gb",
            str(args.min_free_gb),
        ]
        if args.key_refresh_command:
            command.extend(["--key-refresh-command", args.key_refresh_command])
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            write_status(
                status_path,
                {
                    "state": "ERROR",
                    "platform": platform,
                    "patch": args.exact_api_patch,
                    "return_code": result.returncode,
                    "updated_at": int(time.time()),
                },
            )
            return int(result.returncode)

    write_status(
        status_path,
        {
            "state": "COMPLETE",
            "platforms": [str(value).lower() for value in args.platforms],
            "patch": args.exact_api_patch,
            "target_per_platform": args.target_per_platform,
            "updated_at": int(time.time()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
