"""Resumable Riot API acquisition for the current-season dataset.

The package intentionally keeps collection separate from feature extraction so
raw Match-V5 responses can be reprocessed when the scientific pipeline changes.
"""

from .config import CollectorConfig
from .key_provider import EnvFileKeyProvider, KeySnapshot, KeyUnavailable

__all__ = [
    "CollectorConfig",
    "EnvFileKeyProvider",
    "KeySnapshot",
    "KeyUnavailable",
]
