from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acquisition.client import AuthenticationRejected, RiotClient  # noqa: E402
from acquisition.collector import CollectionAgent  # noqa: E402
from acquisition.config import CollectorConfig, parse_patch  # noqa: E402
from acquisition.key_provider import EnvFileKeyProvider, KeyUnavailable  # noqa: E402
from acquisition.raw_store import RawStore  # noqa: E402
from acquisition.state import CollectionState  # noqa: E402
from acquisition.supervisor import AlreadyRunning, CollectorSupervisor  # noqa: E402
from acquisition.supervisor import SingleInstanceLock  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously collect current-season ranked Match-V5 detail/timeline data"
    )
    parser.add_argument(
        "--key-file",
        default=os.getenv("RIOT_API_KEY_FILE"),
        help="user-owned env file containing RIOT_API_KEY (or RIOT_API_KEY_FILE)",
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("LOL_ACQUISITION_ROOT"),
        help="new-season raw root; writes <root>/<platform>/{detail,timeline}",
    )
    parser.add_argument(
        "--state-dir",
        default=os.getenv("LOL_ACQUISITION_STATE_DIR"),
        help="SQLite/status/log directory (default: <output-root>/_collector_state)",
    )
    parser.add_argument("--platform", default="kr")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["CHALLENGER", "GRANDMASTER", "MASTER"],
        choices=["CHALLENGER", "GRANDMASTER", "MASTER"],
    )
    parser.add_argument("--min-api-patch", default="16.13")
    parser.add_argument(
        "--exact-api-patch",
        default=None,
        help="retain only this exact API patch, for example 16.13",
    )
    parser.add_argument(
        "--max-complete-matches",
        type=int,
        default=0,
        help="stop cleanly after this many complete matches for the platform/patch",
    )
    parser.add_argument("--rank-refresh-seconds", type=int, default=3600)
    parser.add_argument("--cycle-seconds", type=int, default=3600)
    parser.add_argument(
        "--work-budget-seconds",
        type=int,
        default=3000,
        help="bound each hourly work slice so rank/key checks cannot be starved",
    )
    parser.add_argument("--initial-lookback-days", type=int, default=14)
    parser.add_argument("--scan-overlap-minutes", type=int, default=15)
    parser.add_argument("--players-per-cycle", type=int, default=250)
    parser.add_argument("--match-pages-per-player", type=int, default=3)
    parser.add_argument("--matches-per-cycle", type=int, default=5000)
    parser.add_argument("--short-limit", type=int, default=18)
    parser.add_argument("--long-limit", type=int, default=95)
    parser.add_argument("--long-window-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-storage-gb",
        type=float,
        default=0.0,
        help="pause raw downloads after this many GiB (0 disables the quota)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="pause raw downloads when the destination has this little free GiB",
    )
    parser.add_argument(
        "--key-refresh-command",
        default=os.getenv("RIOT_KEY_REFRESH_COMMAND"),
        help=(
            "optional external provisioner that atomically replaces the key file; "
            "Riot does not provide an official development-key refresh API"
        ),
    )
    parser.add_argument("--once", action="store_true", help="run one collection cycle")
    parser.add_argument("--probe-only", action="store_true", help="only validate API authentication")
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> CollectorConfig:
    if not args.key_file:
        parser.error("--key-file or RIOT_API_KEY_FILE is required")
    if not args.output_root:
        parser.error("--output-root or LOL_ACQUISITION_ROOT is required")
    output_root = Path(args.output_root).expanduser().resolve()
    state_dir = (
        Path(args.state_dir).expanduser().resolve()
        if args.state_dir
        else output_root / "_collector_state"
    )
    return CollectorConfig(
        key_file=Path(args.key_file).expanduser().resolve(),
        output_root=output_root,
        state_dir=state_dir,
        platform=args.platform,
        tiers=tuple(args.tiers),
        min_api_patch=parse_patch(args.min_api_patch),
        exact_api_patch=parse_patch(args.exact_api_patch),
        max_complete_matches=(
            int(args.max_complete_matches) if args.max_complete_matches > 0 else None
        ),
        rank_refresh_seconds=args.rank_refresh_seconds,
        cycle_interval_seconds=args.cycle_seconds,
        cycle_work_budget_seconds=args.work_budget_seconds,
        initial_lookback_seconds=args.initial_lookback_days * 24 * 60 * 60,
        scan_overlap_seconds=args.scan_overlap_minutes * 60,
        players_per_cycle=args.players_per_cycle,
        match_pages_per_player=args.match_pages_per_player,
        matches_per_cycle=args.matches_per_cycle,
        short_limit=args.short_limit,
        long_limit=args.long_limit,
        long_window_seconds=args.long_window_seconds,
        key_refresh_command=args.key_refresh_command,
        max_storage_bytes=(
            int(args.max_storage_gb * 1024**3) if args.max_storage_gb > 0 else None
        ),
        min_free_bytes=int(args.min_free_gb * 1024**3),
    )


def configure_logging(state_dir: Path, level: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        state_dir / "collector.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, console_handler],
    )


async def run(args: argparse.Namespace, config: CollectorConfig) -> int:
    provider = EnvFileKeyProvider(config.key_file)
    state = CollectionState(config.database_path)
    store = RawStore(config.output_root, config.platform)
    client = RiotClient(config, provider)
    agent = CollectionAgent(config, client, state, store)
    supervisor = CollectorSupervisor(config, provider, client, state, agent)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, supervisor.stop)
        except (NotImplementedError, RuntimeError):
            # Proactor loops on some Windows/Python combinations do not expose
            # add_signal_handler. Ctrl+C still cancels asyncio.run normally.
            pass

    try:
        if args.probe_only:
            await client.probe()
            print("Riot API authentication probe succeeded (key redacted).")
            return 0
        if args.once:
            with SingleInstanceLock(config.lock_path):
                try:
                    await client.probe()
                    summary = await agent.run_cycle()
                except KeyUnavailable as exc:
                    logging.error("%s", exc)
                    return 3
                except AuthenticationRejected as exc:
                    logging.error("Riot API authentication rejected (HTTP %s)", exc.status_code)
                    return 3
                print(summary.to_dict())
                return 0
        await supervisor.run_forever()
        return 0
    except AlreadyRunning as exc:
        logging.error("%s", exc)
        return 4
    finally:
        await client.close()
        state.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args, parser)
    configure_logging(config.state_dir, args.log_level)
    try:
        return asyncio.run(run(args, config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
