from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx

from .config import CollectorConfig
from .key_provider import EnvFileKeyProvider, KeyUnavailable


class RiotAPIError(RuntimeError):
    pass


class AuthenticationRejected(RiotAPIError):
    def __init__(self, status_code: int, key_fingerprint: Optional[str]) -> None:
        self.status_code = int(status_code)
        self.key_fingerprint = key_fingerprint
        super().__init__(f"Riot API authentication rejected (HTTP {status_code})")


class ResourceNotFound(RiotAPIError):
    pass


class ResponseValidationError(RiotAPIError):
    pass


class AsyncMultiWindowLimiter:
    """One application-wide limiter shared by every endpoint and worker."""

    def __init__(self, rules: Sequence[Tuple[int, float]]) -> None:
        self._buckets: List[Tuple[int, float, Deque[float]]] = [
            (int(limit), float(window), deque()) for limit, window in rules
        ]
        self._lock = asyncio.Lock()
        self._blocked_until = 0.0

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                wait_for = max(0.0, self._blocked_until - now)
                for limit, window, calls in self._buckets:
                    while calls and calls[0] <= now - window:
                        calls.popleft()
                    if len(calls) >= limit:
                        wait_for = max(wait_for, calls[0] + window - now)
                if wait_for <= 0:
                    for _, _, calls in self._buckets:
                        calls.append(now)
                    return
            await asyncio.sleep(min(max(wait_for, 0.01), 1.0))

    async def block_for(self, seconds: float) -> None:
        async with self._lock:
            self._blocked_until = max(
                self._blocked_until,
                time.monotonic() + max(0.0, float(seconds)),
            )


class RiotClient:
    def __init__(
        self,
        config: CollectorConfig,
        key_provider: EnvFileKeyProvider,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep=asyncio.sleep,
    ) -> None:
        self.config = config
        self.key_provider = key_provider
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            timeout=config.request_timeout_seconds,
            transport=transport,
            http2=False,
            follow_redirects=False,
        )
        self._limiter = AsyncMultiWindowLimiter(
            [
                (config.short_limit, config.short_window_seconds),
                (config.long_limit, config.long_window_seconds),
            ]
        )
        self._stats: Dict[str, int] = {
            "requests": 0,
            "success": 0,
            "not_found": 0,
            "auth": 0,
            "rate_limited": 0,
            "server_errors": 0,
            "network_errors": 0,
            "retries": 0,
        }

    async def close(self) -> None:
        await self._http.aclose()

    def stats_snapshot(self) -> Dict[str, int]:
        return dict(self._stats)

    @staticmethod
    def _retry_after(response: httpx.Response, fallback: float) -> float:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return fallback
        try:
            return max(0.0, float(raw))
        except ValueError:
            return fallback

    async def _request_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        allow_not_found: bool = False,
    ) -> Any:
        backoff = 0.5
        last_response: Optional[httpx.Response] = None
        for attempt in range(self.config.max_retries + 1):
            key_fingerprint: Optional[str] = None
            try:
                key = self.key_provider.snapshot()
                key_fingerprint = key.fingerprint
            except KeyUnavailable:
                raise

            await self._limiter.acquire()
            try:
                response = await self._http.get(
                    url,
                    params=params,
                    headers={"X-Riot-Token": key.value, "Accept": "application/json"},
                )
                last_response = response
                self._stats["requests"] += 1
            except httpx.TransportError:
                self._stats["network_errors"] += 1
                if attempt >= self.config.max_retries:
                    raise RiotAPIError("Riot API network retries exhausted")
                self._stats["retries"] += 1
                await self._sleep(backoff + random.uniform(0.01, 0.05))
                backoff = min(backoff * 2.0, 30.0)
                continue

            status = response.status_code
            if 200 <= status < 300:
                self._stats["success"] += 1
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ResponseValidationError("Riot API returned invalid JSON") from exc

            if status in (401, 403):
                self._stats["auth"] += 1
                raise AuthenticationRejected(status, key_fingerprint)

            if status == 404 and allow_not_found:
                self._stats["not_found"] += 1
                return None

            if status == 429:
                self._stats["rate_limited"] += 1
                wait_seconds = self._retry_after(response, backoff)
                await self._limiter.block_for(wait_seconds)
                if attempt >= self.config.max_retries:
                    raise RiotAPIError("Riot API rate-limit retries exhausted")
                self._stats["retries"] += 1
                await self._sleep(wait_seconds)
                backoff = min(max(backoff * 1.5, wait_seconds), 30.0)
                continue

            if status >= 500:
                self._stats["server_errors"] += 1
                if attempt >= self.config.max_retries:
                    raise RiotAPIError(f"Riot API server error persisted (HTTP {status})")
                self._stats["retries"] += 1
                await self._sleep(backoff + random.uniform(0.01, 0.05))
                backoff = min(backoff * 2.0, 30.0)
                continue

            raise RiotAPIError(f"Riot API request failed (HTTP {status})")

        status = last_response.status_code if last_response is not None else "unknown"
        raise RiotAPIError(f"Riot API retries exhausted (last HTTP {status})")

    async def probe(self) -> Dict[str, Any]:
        url = f"https://{self.config.platform}.api.riotgames.com/lol/status/v4/platform-data"
        result = await self._request_json(url)
        if not isinstance(result, dict):
            raise ResponseValidationError("health probe returned a non-object")
        return result

    async def high_tier_entries(self, tier: str) -> List[Dict[str, Any]]:
        normalized = tier.upper()
        resource = {
            "CHALLENGER": "challengerleagues",
            "GRANDMASTER": "grandmasterleagues",
            "MASTER": "masterleagues",
        }.get(normalized)
        if resource is None:
            raise ValueError(f"unsupported high tier: {tier}")
        url = (
            f"https://{self.config.platform}.api.riotgames.com/lol/league/v4/"
            f"{resource}/by-queue/{quote(self.config.queue, safe='')}"
        )
        result = await self._request_json(url)
        if not isinstance(result, dict) or not isinstance(result.get("entries"), list):
            raise ResponseValidationError("high-tier response is missing entries[]")
        entries = result["entries"]
        if any(not isinstance(entry, dict) for entry in entries):
            raise ResponseValidationError("high-tier entries contain a non-object")
        return entries

    async def match_ids_by_puuid(
        self,
        puuid: str,
        *,
        start_time: int,
        end_time: int,
        start: int = 0,
        count: int = 100,
    ) -> List[str]:
        encoded = quote(puuid, safe="")
        url = (
            f"https://{self.config.region}.api.riotgames.com/lol/match/v5/"
            f"matches/by-puuid/{encoded}/ids"
        )
        result = await self._request_json(
            url,
            params={
                "startTime": int(start_time),
                "endTime": int(end_time),
                "queue": self.config.queue_id,
                "start": int(start),
                "count": min(max(int(count), 1), 100),
            },
        )
        if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
            raise ResponseValidationError("match-ID response must be a string array")
        return result

    async def match_detail(self, match_id: str) -> Optional[Dict[str, Any]]:
        encoded = quote(match_id, safe="")
        url = (
            f"https://{self.config.region}.api.riotgames.com/lol/match/v5/"
            f"matches/{encoded}"
        )
        result = await self._request_json(url, allow_not_found=True)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise ResponseValidationError("match detail response must be an object")
        return result

    async def match_timeline(self, match_id: str) -> Optional[Dict[str, Any]]:
        encoded = quote(match_id, safe="")
        url = (
            f"https://{self.config.region}.api.riotgames.com/lol/match/v5/"
            f"matches/{encoded}/timeline"
        )
        result = await self._request_json(url, allow_not_found=True)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise ResponseValidationError("match timeline response must be an object")
        return result
