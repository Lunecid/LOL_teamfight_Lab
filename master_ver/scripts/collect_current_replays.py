from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acquisition.replay import (  # noqa: E402
    LeagueClientUnavailable,
    ReplayCollector,
    ReplayCollectorConfig,
)
from acquisition.supervisor import AlreadyRunning, SingleInstanceLock  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive League Client replay files that match completed Match-V5 pairs"
        )
    )
    parser.add_argument(
        "--source-db",
        default=os.getenv("LOL_ACQUISITION_DB"),
        help="read-only collector.sqlite3 containing completed match metadata",
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("LOL_REPLAY_ROOT"),
        help="replay archive root; writes <root>/<platform>/rofl",
    )
    parser.add_argument(
        "--league-lockfile",
        default=os.getenv("LOL_LEAGUE_LOCKFILE"),
        help="League Client lockfile; the client must be open and logged in",
    )
    parser.add_argument("--platform", default="kr")
    parser.add_argument("--max-replays", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--max-storage-gb", type=float, default=8.0)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--download-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    parser.add_argument(
        "--keep-client-copy",
        action="store_true",
        help="keep the League Client replay copy after archiving it",
    )
    parser.add_argument(
        "--newest-first",
        action="store_true",
        help="process newest matches first instead of expiry-risk oldest first",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def config_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> ReplayCollectorConfig:
    if not args.source_db:
        parser.error("--source-db or LOL_ACQUISITION_DB is required")
    if not args.output_root:
        parser.error("--output-root or LOL_REPLAY_ROOT is required")
    if not args.league_lockfile:
        parser.error("--league-lockfile or LOL_LEAGUE_LOCKFILE is required")
    if args.max_storage_gb <= 0:
        parser.error("--max-storage-gb must be positive")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    return ReplayCollectorConfig(
        source_database=Path(args.source_db).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        league_lockfile=Path(args.league_lockfile).expanduser().resolve(),
        platform=args.platform,
        max_replays=args.max_replays,
        max_candidates=args.max_candidates,
        max_storage_bytes=int(args.max_storage_gb * 1024**3),
        min_free_bytes=int(args.min_free_gb * 1024**3),
        poll_seconds=args.poll_seconds,
        download_timeout_seconds=args.download_timeout_seconds,
        request_delay_seconds=args.request_delay_seconds,
        delete_client_copy=not args.keep_client_copy,
        oldest_first=not args.newest_first,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args, parser)
    try:
        with SingleInstanceLock(config.state_dir / "replay_collector.lock"):
            collector = ReplayCollector(config)
            try:
                summary = collector.collect(dry_run=args.dry_run)
            finally:
                collector.close()
    except AlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except LeagueClientUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
