from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .client import AuthenticationRejected, RiotAPIError, RiotClient
from .collector import CollectionAgent, CycleSummary
from .config import CollectorConfig
from .key_provider import EnvFileKeyProvider, KeyUnavailable
from .state import CollectionState


LOGGER = logging.getLogger(__name__)


class AlreadyRunning(RuntimeError):
    pass


class SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise AlreadyRunning("another collector agent already holds the lock") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class CollectorSupervisor:
    def __init__(
        self,
        config: CollectorConfig,
        key_provider: EnvFileKeyProvider,
        client: RiotClient,
        state: CollectionState,
        agent: CollectionAgent,
    ) -> None:
        self.config = config
        self.key_provider = key_provider
        self.client = client
        self.state = state
        self.agent = agent
        self.stop_event = asyncio.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def _status(
        self,
        state_name: str,
        *,
        detail: Optional[str] = None,
        cycle: Optional[CycleSummary] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "state": state_name,
            "updated_at": int(time.time()),
            "platform": self.config.platform,
            "region": self.config.region,
            "queue_id": self.config.queue_id,
            "map_id": self.config.map_id,
            "counts": self.state.counts(),
            "patch_counts": self.state.patch_counts(),
            "raw_bytes": self.state.raw_bytes(),
            "api_stats": self.client.stats_snapshot(),
        }
        if detail:
            payload["detail"] = detail[:500]
        if cycle is not None:
            payload["last_cycle"] = cycle.to_dict()
        _atomic_json(self.config.status_path, payload)

    def _write_auth_required(self, status_code: Optional[int]) -> None:
        self.config.auth_required_path.parent.mkdir(parents=True, exist_ok=True)
        status_text = f"HTTP {status_code}" if status_code is not None else "missing key"
        text = (
            "Riot API collection is paused because authentication is unavailable "
            f"({status_text}).\n\n"
            "Development keys cannot be regenerated through an official API. "
            "Reset the key in the Riot Developer Portal and atomically replace "
            f"RIOT_API_KEY in:\n{self.config.key_file}\n\n"
            "The running agent will validate the changed file and resume from "
            "its SQLite checkpoint automatically. Do not paste the key into logs.\n"
        )
        self.config.auth_required_path.write_text(text, encoding="utf-8")

    def _clear_auth_required(self) -> None:
        try:
            self.config.auth_required_path.unlink()
        except FileNotFoundError:
            pass

    async def _run_refresh_command(self, fingerprint: Optional[str]) -> Optional[bool]:
        command = self.config.key_refresh_command
        if not command:
            return None
        env = dict(os.environ)
        env["RIOT_API_KEY_FILE"] = str(self.config.key_file)
        LOGGER.warning("running configured external key provisioner")
        process = await asyncio.create_subprocess_shell(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            LOGGER.error("external key provisioner timed out")
            return False
        else:
            if process.returncode != 0:
                LOGGER.error("external key provisioner exited with code %s", process.returncode)
                return False
        return True

    async def _wait_for_auth(
        self,
        previous_fingerprint: Optional[str],
        status_code: Optional[int],
    ) -> None:
        self._write_auth_required(status_code)
        self._status("WAITING_FOR_KEY", detail="authentication is paused")
        provisioner_backoff = self.config.auth_poll_seconds
        while not self.stop_event.is_set():
            provisioner_result = await self._run_refresh_command(previous_fingerprint)
            try:
                changed = self.key_provider.snapshot()
            except KeyUnavailable:
                changed = None
            if changed is None or changed.fingerprint == previous_fingerprint:
                if provisioner_result is not None:
                    wait_seconds = provisioner_backoff
                    provisioner_backoff = min(provisioner_backoff * 2.0, 300.0)
                else:
                    wait_seconds = self.config.auth_poll_seconds
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self.client.probe()
            except AuthenticationRejected as exc:
                previous_fingerprint = changed.fingerprint
                provisioner_backoff = self.config.auth_poll_seconds
                self._write_auth_required(exc.status_code)
                self._status("WAITING_FOR_KEY", detail="replacement key was rejected")
                continue
            except (KeyUnavailable, RiotAPIError) as exc:
                self._status("WAITING_FOR_KEY", detail=str(exc))
                try:
                    await asyncio.wait_for(
                        self.stop_event.wait(),
                        timeout=self.config.error_retry_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            self._clear_auth_required()
            self.state.record_event("authentication_resumed", {})
            self._status("RUNNING", detail="replacement key validated")
            return

    async def _sleep_until_next_cycle(self, cycle_started: float) -> None:
        elapsed = time.monotonic() - cycle_started
        delay = max(1.0, self.config.cycle_interval_seconds - elapsed)
        self._status("SLEEPING", detail=f"next cycle in {int(delay)} seconds")
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    async def run_forever(self) -> None:
        with SingleInstanceLock(self.config.lock_path):
            self._status("STARTING")
            while not self.stop_event.is_set():
                cycle_started = time.monotonic()
                try:
                    await self.client.probe()
                    self._clear_auth_required()
                    self._status("RUNNING")
                    summary = await self.agent.run_cycle()
                    if summary.quota_reached:
                        self.state.record_event(
                            "quota_complete",
                            {
                                "platform": self.config.platform,
                                "complete": self.agent.complete_count(),
                                "target": self.config.max_complete_matches,
                            },
                        )
                        self._status("COMPLETE", cycle=summary)
                        return
                    self._status("SLEEPING", cycle=summary)
                    await self._sleep_until_next_cycle(cycle_started)
                except AuthenticationRejected as exc:
                    self.state.record_event(
                        "authentication_paused",
                        {"status_code": exc.status_code},
                    )
                    await self._wait_for_auth(exc.key_fingerprint, exc.status_code)
                except KeyUnavailable:
                    self.state.record_event("authentication_paused", {"reason": "missing_key"})
                    await self._wait_for_auth(None, None)
                except RiotAPIError as exc:
                    LOGGER.exception("collection cycle failed after API retries")
                    self._status("BACKOFF", detail=str(exc))
                    try:
                        await asyncio.wait_for(
                            self.stop_event.wait(),
                            timeout=self.config.error_retry_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                except Exception as exc:
                    LOGGER.exception("unexpected collection-agent failure")
                    self._status("ERROR", detail=f"{type(exc).__name__}: {exc}")
                    try:
                        await asyncio.wait_for(
                            self.stop_event.wait(),
                            timeout=self.config.error_retry_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
            self._status("STOPPED")
