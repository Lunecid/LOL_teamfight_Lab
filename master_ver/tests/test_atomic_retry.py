import errno
import json

import pytest

from studies.execution_progress_20261003 import progress as ep


class _WindowsReplaceError(OSError):
    def __init__(self, winerror: int) -> None:
        super().__init__(13, f"temporary replace failure ({winerror})")
        self.winerror = winerror


@pytest.mark.parametrize("winerror", [5, 32])
def test_atomic_write_retries_transient_replace_once_written(tmp_path, monkeypatch, winerror):
    target = tmp_path / "progress.json"
    target.write_text('{"version": 1}\n', encoding="utf-8")
    original_replace = ep.os.replace
    failures = [_WindowsReplaceError(winerror), _WindowsReplaceError(winerror)]
    attempts = []
    sleeps = []

    def transient_replace(source, destination):
        attempts.append((source, destination))
        if failures:
            raise failures.pop(0)
        return original_replace(source, destination)

    monkeypatch.setattr(ep.os, "replace", transient_replace)
    monkeypatch.setattr(ep.time, "sleep", sleeps.append)

    ep._atomic_write_json(target, {"version": 2})

    assert len(attempts) == 3
    assert len({source for source, _ in attempts}) == 1
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}
    assert sleeps == [ep._ATOMIC_REPLACE_RETRY_DELAY_SECONDS] * 2


def test_atomic_write_exhaustion_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    original = '{"version": 1}\n'
    target.write_text(original, encoding="utf-8")
    attempts = []
    sleeps = []

    def permanently_locked(source, destination):
        attempts.append((source, destination))
        raise _WindowsReplaceError(5)

    monkeypatch.setattr(ep.os, "replace", permanently_locked)
    monkeypatch.setattr(ep.time, "sleep", sleeps.append)

    with pytest.raises(OSError) as raised:
        ep._atomic_write_json(target, {"version": 2})

    assert raised.value.winerror == 5
    assert len(attempts) == ep._ATOMIC_REPLACE_RETRY_ATTEMPTS
    assert len(sleeps) == len(attempts) - 1
    assert sum(sleeps) <= 3.0
    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_does_not_retry_other_replace_errors(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    original = '{"version": 1}\n'
    target.write_text(original, encoding="utf-8")
    attempts = []
    sleeps = []
    error = OSError(errno.EINVAL, "invalid replace")

    def permanent_failure(source, destination):
        attempts.append((source, destination))
        raise error

    monkeypatch.setattr(ep.os, "replace", permanent_failure)
    monkeypatch.setattr(ep.time, "sleep", sleeps.append)

    with pytest.raises(OSError) as raised:
        ep._atomic_write_json(target, {"version": 2})

    assert raised.value is error
    assert len(attempts) == 1
    assert sleeps == []
    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".*.tmp")) == []
