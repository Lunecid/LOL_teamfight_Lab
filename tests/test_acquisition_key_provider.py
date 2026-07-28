import asyncio

import pytest

from acquisition.key_provider import EnvFileKeyProvider, KeyUnavailable


def test_key_provider_reads_explicit_file_and_redacts_repr(tmp_path):
    secret = "RGAPI-test-value-that-must-never-be-logged"
    path = tmp_path / "riot.env"
    path.write_text(f"RIOT_API_KEY={secret}\n", encoding="utf-8")

    snapshot = EnvFileKeyProvider(path).snapshot()

    assert snapshot.value == secret
    assert len(snapshot.fingerprint) == 12
    assert secret not in repr(snapshot)
    assert "RGAPI" not in repr(snapshot)


def test_key_provider_rejects_missing_or_malformed_file(tmp_path):
    provider = EnvFileKeyProvider(tmp_path / "missing.env")
    with pytest.raises(KeyUnavailable):
        provider.snapshot()

    path = tmp_path / "bad.env"
    path.write_text("RIOT_API_KEY=short\n", encoding="utf-8")
    with pytest.raises(KeyUnavailable):
        EnvFileKeyProvider(path).snapshot()


def test_wait_for_change_returns_only_a_different_key(tmp_path):
    async def exercise():
        first = "RGAPI-first-value-long-enough-for-validation"
        second = "RGAPI-second-value-long-enough-for-validation"
        path = tmp_path / "riot.env"
        path.write_text(f"RIOT_API_KEY={first}\n", encoding="utf-8")
        provider = EnvFileKeyProvider(path)
        old = provider.snapshot()

        waiter = asyncio.create_task(provider.wait_for_change(old.fingerprint, 0.01))
        await asyncio.sleep(0.02)
        assert not waiter.done()
        path.write_text(f"RIOT_API_KEY={second}\n", encoding="utf-8")

        changed = await asyncio.wait_for(waiter, timeout=1.0)
        assert changed.value == second
        assert changed.fingerprint != old.fingerprint

    asyncio.run(exercise())
