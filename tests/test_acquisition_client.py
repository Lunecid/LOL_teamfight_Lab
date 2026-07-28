import asyncio

import httpx
import pytest

from acquisition.client import (
    AuthenticationRejected,
    ResponseValidationError,
    RiotAPIError,
    RiotClient,
)
from acquisition.config import CollectorConfig
from acquisition.key_provider import EnvFileKeyProvider


def make_config(tmp_path, **overrides):
    values = dict(
        key_file=tmp_path / "riot.env",
        output_root=tmp_path / "raw",
        state_dir=tmp_path / "state",
        short_limit=100,
        long_limit=100,
        short_window_seconds=0.01,
        long_window_seconds=0.01,
        max_retries=2,
    )
    values.update(overrides)
    return CollectorConfig(**values)


def write_key(path, value="RGAPI-test-value-long-enough-and-redacted"):
    path.write_text(f"RIOT_API_KEY={value}\n", encoding="utf-8")
    return value


def test_probe_uses_header_without_exposing_key(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        secret = write_key(config.key_file)
        seen = {}

        def handler(request):
            seen["token"] = request.headers.get("X-Riot-Token")
            return httpx.Response(200, json={"id": "KR"})

        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(handler),
        )
        try:
            assert await client.probe() == {"id": "KR"}
            assert seen["token"] == secret
            assert secret not in repr(client.stats_snapshot())
        finally:
            await client.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("status", [401, 403])
def test_auth_rejection_is_typed_and_redacted(tmp_path, status):
    async def exercise():
        config = make_config(tmp_path)
        secret = write_key(config.key_file)

        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={})),
        )
        try:
            with pytest.raises(AuthenticationRejected) as caught:
                await client.probe()
            assert caught.value.status_code == status
            assert secret not in str(caught.value)
            assert caught.value.key_fingerprint is not None
        finally:
            await client.close()

    asyncio.run(exercise())


def test_429_honors_retry_after_and_retries(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        write_key(config.key_file)
        calls = {"n": 0}
        sleeps = []

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, json={})
            return httpx.Response(200, json={"id": "KR"})

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
        )
        try:
            await client.probe()
            assert calls["n"] == 2
            assert sleeps == [0.0]
            assert client.stats_snapshot()["rate_limited"] == 1
        finally:
            await client.close()

    asyncio.run(exercise())


def test_match_id_request_carries_queue_and_time_window(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        write_key(config.key_file)
        seen = {}

        def handler(request):
            seen.update(dict(request.url.params))
            return httpx.Response(200, json=["KR_123"])

        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(handler),
        )
        try:
            ids = await client.match_ids_by_puuid(
                "puuid-value", start_time=100, end_time=200, start=0, count=100
            )
            assert ids == ["KR_123"]
            assert seen["queue"] == "420"
            assert seen["startTime"] == "100"
            assert seen["endTime"] == "200"
        finally:
            await client.close()

    asyncio.run(exercise())


def test_malformed_success_body_is_not_treated_as_empty_match_ids(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        write_key(config.key_file)
        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"unexpected": "shape"})
            ),
        )
        try:
            with pytest.raises(ResponseValidationError):
                await client.match_ids_by_puuid(
                    "puuid", start_time=100, end_time=200
                )
        finally:
            await client.close()

    asyncio.run(exercise())


def test_malformed_success_body_is_not_treated_as_missing_detail(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        write_key(config.key_file)
        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=["not", "an", "object"])
            ),
        )
        try:
            with pytest.raises(ResponseValidationError):
                await client.match_detail("KR_123")
        finally:
            await client.close()

    asyncio.run(exercise())


def test_remote_protocol_error_uses_transport_retry_path(tmp_path):
    async def exercise():
        config = make_config(tmp_path, max_retries=1)
        write_key(config.key_file)
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            raise httpx.RemoteProtocolError("peer disconnected", request=request)

        async def no_wait(seconds):
            return None

        client = RiotClient(
            config,
            EnvFileKeyProvider(config.key_file),
            transport=httpx.MockTransport(handler),
            sleep=no_wait,
        )
        try:
            with pytest.raises(RiotAPIError):
                await client.probe()
            assert calls["n"] == 2
            assert client.stats_snapshot()["network_errors"] == 2
        finally:
            await client.close()

    asyncio.run(exercise())
