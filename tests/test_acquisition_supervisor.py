import asyncio

from acquisition.client import AuthenticationRejected
from acquisition.config import CollectorConfig
from acquisition.key_provider import EnvFileKeyProvider
from acquisition.state import CollectionState
from acquisition.supervisor import CollectorSupervisor
from acquisition.supervisor import AlreadyRunning, SingleInstanceLock


class ProbeClient:
    def __init__(self, provider, rejected_value):
        self.provider = provider
        self.rejected_value = rejected_value
        self.calls = 0

    async def probe(self):
        self.calls += 1
        snapshot = self.provider.snapshot()
        if snapshot.value == self.rejected_value:
            raise AuthenticationRejected(403, snapshot.fingerprint)
        return {"id": "KR"}

    def stats_snapshot(self):
        return {"requests": self.calls}


class UnusedAgent:
    async def run_cycle(self):
        raise AssertionError("not used")


def test_auth_pause_resumes_after_key_file_changes_without_logging_secret(tmp_path):
    async def exercise():
        first = "RGAPI-first-rejected-value-long-enough"
        second = "RGAPI-second-accepted-value-long-enough"
        key_file = tmp_path / "riot.env"
        key_file.write_text(f"RIOT_API_KEY={first}\n", encoding="utf-8")
        config = CollectorConfig(
            key_file=key_file,
            output_root=tmp_path / "raw",
            state_dir=tmp_path / "state",
            auth_poll_seconds=0.01,
            error_retry_seconds=0.01,
        )
        provider = EnvFileKeyProvider(key_file)
        old = provider.snapshot()
        client = ProbeClient(provider, first)
        state = CollectionState(config.database_path)
        supervisor = CollectorSupervisor(config, provider, client, state, UnusedAgent())

        async def replace_key():
            await asyncio.sleep(0.03)
            key_file.write_text(f"RIOT_API_KEY={second}\n", encoding="utf-8")

        try:
            replacer = asyncio.create_task(replace_key())
            await asyncio.wait_for(
                supervisor._wait_for_auth(old.fingerprint, 403), timeout=1.0
            )
            await replacer
            assert not config.auth_required_path.exists()
            status = config.status_path.read_text(encoding="utf-8")
            assert '"state": "RUNNING"' in status
            assert first not in status and second not in status
            events = [
                row["event_type"]
                for row in state.conn.execute("SELECT event_type FROM collection_events")
            ]
            assert "authentication_resumed" in events
        finally:
            state.close()

    asyncio.run(exercise())


def test_single_instance_lock_rejects_manual_second_collector(tmp_path):
    lock_path = tmp_path / "collector.lock"
    with SingleInstanceLock(lock_path):
        try:
            with SingleInstanceLock(lock_path):
                raise AssertionError("second lock unexpectedly succeeded")
        except AlreadyRunning:
            pass


def test_failed_external_provisioner_is_retried_until_key_changes(tmp_path):
    async def exercise():
        first = "RGAPI-first-rejected-value-long-enough"
        second = "RGAPI-second-accepted-value-long-enough"
        key_file = tmp_path / "riot.env"
        key_file.write_text(f"RIOT_API_KEY={first}\n", encoding="utf-8")
        config = CollectorConfig(
            key_file=key_file,
            output_root=tmp_path / "raw",
            state_dir=tmp_path / "state",
            auth_poll_seconds=0.01,
            error_retry_seconds=0.01,
            key_refresh_command="configured-by-test",
        )
        provider = EnvFileKeyProvider(key_file)
        client = ProbeClient(provider, first)
        state = CollectionState(config.database_path)
        supervisor = CollectorSupervisor(config, provider, client, state, UnusedAgent())
        calls = {"n": 0}

        async def fake_provisioner(fingerprint):
            calls["n"] += 1
            if calls["n"] == 1:
                return False
            key_file.write_text(f"RIOT_API_KEY={second}\n", encoding="utf-8")
            return True

        supervisor._run_refresh_command = fake_provisioner
        try:
            await asyncio.wait_for(
                supervisor._wait_for_auth(provider.snapshot().fingerprint, 403),
                timeout=1.0,
            )
            assert calls["n"] >= 2
            assert not config.auth_required_path.exists()
        finally:
            state.close()

    asyncio.run(exercise())


def test_zero_exit_without_key_change_is_retried(tmp_path):
    async def exercise():
        first = "RGAPI-first-rejected-value-long-enough"
        second = "RGAPI-second-accepted-value-long-enough"
        key_file = tmp_path / "riot.env"
        key_file.write_text(f"RIOT_API_KEY={first}\n", encoding="utf-8")
        config = CollectorConfig(
            key_file=key_file,
            output_root=tmp_path / "raw",
            state_dir=tmp_path / "state",
            auth_poll_seconds=0.01,
            error_retry_seconds=0.01,
            key_refresh_command="configured-by-test",
        )
        provider = EnvFileKeyProvider(key_file)
        client = ProbeClient(provider, first)
        state = CollectionState(config.database_path)
        supervisor = CollectorSupervisor(config, provider, client, state, UnusedAgent())
        calls = {"n": 0}

        async def zero_exit_provisioner(fingerprint):
            calls["n"] += 1
            if calls["n"] >= 2:
                key_file.write_text(f"RIOT_API_KEY={second}\n", encoding="utf-8")
            return True

        supervisor._run_refresh_command = zero_exit_provisioner
        try:
            await asyncio.wait_for(
                supervisor._wait_for_auth(provider.snapshot().fingerprint, 403),
                timeout=1.0,
            )
            assert calls["n"] >= 2
        finally:
            state.close()

    asyncio.run(exercise())
