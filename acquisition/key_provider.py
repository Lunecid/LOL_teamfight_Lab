from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


class KeyUnavailable(RuntimeError):
    """Raised when no usable Riot API key is available."""


@dataclass(frozen=True)
class KeySnapshot:
    value: str
    fingerprint: str
    modified_ns: int

    def __repr__(self) -> str:
        return (
            "KeySnapshot(value=<redacted>, "
            f"fingerprint={self.fingerprint!r}, modified_ns={self.modified_ns})"
        )


def _parse_env_file(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0:1] == value[-1:] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class EnvFileKeyProvider:
    """Read a Riot key from an explicit user-owned env file.

    The file is checked on every snapshot request. Only a one-way fingerprint
    may be logged; the key and its prefix/suffix must never be logged.
    """

    def __init__(self, path: Path, variable: str = "RIOT_API_KEY") -> None:
        self.path = Path(path).expanduser().resolve()
        self.variable = variable

    def snapshot(self) -> KeySnapshot:
        try:
            stat = self.path.stat()
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise KeyUnavailable(f"Riot key file is missing: {self.path}") from exc
        except OSError as exc:
            raise KeyUnavailable(f"Riot key file cannot be read: {self.path}") from exc

        value = _parse_env_file(text).get(self.variable, "").strip()
        if not value:
            raise KeyUnavailable(f"{self.variable} is missing from {self.path}")
        if any(ch.isspace() for ch in value) or len(value) < 20:
            raise KeyUnavailable(f"{self.variable} in {self.path} is malformed")
        return KeySnapshot(
            value=value,
            fingerprint=_fingerprint(value),
            modified_ns=stat.st_mtime_ns,
        )

    async def wait_for_change(
        self,
        previous_fingerprint: Optional[str],
        poll_seconds: float,
        stop_event: Optional[asyncio.Event] = None,
    ) -> KeySnapshot:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise asyncio.CancelledError
            try:
                current = self.snapshot()
            except KeyUnavailable:
                current = None
            if current is not None and current.fingerprint != previous_fingerprint:
                return current
            try:
                if stop_event is None:
                    await asyncio.sleep(poll_seconds)
                else:
                    await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass


def secure_file_permissions(path: Path) -> None:
    """Apply owner-only permissions where the operating system supports them."""

    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows ACLs are not fully represented by chmod. The installer docs
        # instruct the user to keep the file under their private profile.
        pass
