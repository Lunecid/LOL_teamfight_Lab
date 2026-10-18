from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


PLATFORM_TO_REGION = {
    "kr": "asia",
    "jp1": "asia",
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


def parse_patch(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None or not value.strip():
        return None
    parts = value.strip().split(".")
    if len(parts) < 2:
        raise ValueError("patch must look like '16.13'")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError("patch must contain numeric major/minor values") from exc


def patch_from_game_version(value: str) -> Optional[Tuple[int, int]]:
    try:
        parts = str(value).split(".")
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError, IndexError):
        return None


def public_patch_label(api_patch: Optional[Tuple[int, int]]) -> Optional[str]:
    """Map Riot's 2026 API build major (16.x) to the public patch label (26.x).

    The raw ``gameVersion`` is always retained as the source of truth. This
    convenience mapping is deliberately narrow so a future numbering change
    cannot silently relabel data.
    """

    if api_patch is None:
        return None
    major, minor = api_patch
    if major == 16:
        return f"26.{minor}"
    return None


@dataclass(frozen=True)
class CollectorConfig:
    key_file: Path
    output_root: Path
    state_dir: Path
    platform: str = "kr"
    region: Optional[str] = None
    queue: str = "RANKED_SOLO_5x5"
    queue_id: int = 420
    map_id: int = 11
    game_mode: str = "CLASSIC"
    tiers: Tuple[str, ...] = ("CHALLENGER", "GRANDMASTER", "MASTER")
    min_api_patch: Optional[Tuple[int, int]] = (16, 13)
    exact_api_patch: Optional[Tuple[int, int]] = None
    max_complete_matches: Optional[int] = None
    rank_refresh_seconds: int = 3600
    cycle_interval_seconds: int = 3600
    cycle_work_budget_seconds: int = 3000
    initial_lookback_seconds: int = 14 * 24 * 60 * 60
    scan_overlap_seconds: int = 15 * 60
    players_per_cycle: int = 250
    match_pages_per_player: int = 3
    matches_per_cycle: int = 5000
    request_timeout_seconds: float = 30.0
    max_retries: int = 5
    auth_poll_seconds: float = 15.0
    error_retry_seconds: float = 60.0
    short_limit: int = 18
    short_window_seconds: float = 1.0
    long_limit: int = 95
    long_window_seconds: float = 120.0
    key_refresh_command: Optional[str] = None
    max_missing_puuid_fraction: float = 0.01
    max_storage_bytes: Optional[int] = None
    min_free_bytes: int = 0

    def __post_init__(self) -> None:
        platform = self.platform.lower()
        if platform not in PLATFORM_TO_REGION:
            raise ValueError(f"unsupported platform: {self.platform}")
        object.__setattr__(self, "platform", platform)
        if self.region is None:
            object.__setattr__(self, "region", PLATFORM_TO_REGION[platform])
        if self.rank_refresh_seconds < 60:
            raise ValueError("rank_refresh_seconds must be at least 60")
        if self.cycle_interval_seconds < 60:
            raise ValueError("cycle_interval_seconds must be at least 60")
        if self.cycle_work_budget_seconds < 1:
            raise ValueError("cycle_work_budget_seconds must be positive")
        if self.cycle_work_budget_seconds >= self.cycle_interval_seconds:
            raise ValueError(
                "cycle_work_budget_seconds must be shorter than cycle_interval_seconds"
            )
        if self.players_per_cycle < 1 or self.matches_per_cycle < 1:
            raise ValueError("cycle limits must be positive")
        if self.short_limit < 1 or self.long_limit < 1:
            raise ValueError("rate limits must be positive")
        if not 0.0 <= self.max_missing_puuid_fraction <= 1.0:
            raise ValueError("max_missing_puuid_fraction must be between 0 and 1")
        if self.max_storage_bytes is not None and self.max_storage_bytes < 0:
            raise ValueError("max_storage_bytes must be non-negative")
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must be non-negative")
        if self.max_complete_matches is not None and self.max_complete_matches < 1:
            raise ValueError("max_complete_matches must be positive")
        if (
            self.exact_api_patch is not None
            and self.min_api_patch is not None
            and self.exact_api_patch < self.min_api_patch
        ):
            raise ValueError("exact_api_patch cannot be before min_api_patch")

    @property
    def database_path(self) -> Path:
        return self.state_dir / "collector.sqlite3"

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def auth_required_path(self) -> Path:
        return self.state_dir / "AUTH_REQUIRED.txt"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "collector.lock"
