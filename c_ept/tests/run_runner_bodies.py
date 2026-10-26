"""Loader and stage bodies for the runner tests (``c_ept/tests/test_run_runners.py``, S4b-prep K6a): what a child process of the supervisor imports by name.

Not a test module (no ``test_`` prefix, nothing here is collected). ``toy_inputs`` is an inputs loader (``module:function``, the ``loader`` argument of the
stage bodies) that opens the toy stores of the manifest and takes the static channels from ``ept_toys.toy_static`` instead of a terrain folder.
``train_stage`` runs the real ``runners.train_stage`` after it has replaced the E6 check of this child with one that finds no model server (the machine of a
test run is not asked about its processes, and a GPU campaign may be running on it). The children see ``CUDA_VISIBLE_DEVICES=''``.

해석: 감독자 시험의 자식 프로세스가 이름으로 불러 쓰는 장난감 입력 적재기와 단계 본문이다. 학습 본문은 진짜 본문을 부르되 이 자식의 E6 확인만 모형 서버가 없다고 답하는
것으로 바꾼다.
"""
from __future__ import annotations

from pathlib import Path

from c_ept.run import guards, runners
from c_ept.run import identity as run_identity

E6_RECORD = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 5, 'cmdline_unreadable': 0, 'rule': 'test rule'}


def toy_inputs(manifest_path) -> runners.Inputs:
    from c_ept.tests.ept_toys import toy_static
    manifest = runners.read_manifest(manifest_path)
    s1, s2, labels = runners.open_stores(manifest)
    values, path = runners.read_scales(manifest)
    return runners.Inputs(s1=s1, s2=s2, labels=labels, static=toy_static(), static_source=None, scales=values, scales_path=path)


def quiet_e6(*, processes=None, probe=None) -> dict:
    return dict(E6_RECORD)


def _recording_verify(run_dir, *, change_at=None):
    """A replacement of ``identity.verify`` for this child (the identity of a test run is made up): it appends the commit of the identity it is given to
    ``<run_dir>/verified_commit.txt`` and, with ``change_at`` = n, raises ``IdentityChanged`` at its n-th call."""
    calls = []

    def verify(saved, *, repo=None) -> None:
        calls.append(1)
        with (Path(run_dir) / 'verified_commit.txt').open('a', encoding='utf-8') as stream:
            stream.write(saved.commit + '\n')
        if change_at is not None and len(calls) == change_at:
            raise run_identity.IdentityChanged('imported code changed since the identity was bound', ['c_ept/run/runners.py'])
    return verify


def train_stage(run_dir, *arguments) -> None:
    guards.e6 = quiet_e6                                              # this child only
    run_identity.verify = _recording_verify(run_dir)
    runners.train_stage(run_dir, *arguments)


def changed_identity_train_stage(run_dir, *arguments) -> None:
    """The training stage in a child whose identity check finds a changed file at the second epoch."""
    guards.e6 = quiet_e6
    run_identity.verify = _recording_verify(run_dir, change_at=2)
    runners.train_stage(run_dir, *arguments)


def budget_train_stage(run_dir, *arguments) -> None:
    """The training stage in a child whose training clock jumps 90,000 s (more than the 24 h cap) at the second E6 check, that is while the second epoch trains:
    no time is spent, the cap is reached at the end of that epoch."""
    import time
    from c_ept.ept import train as ept_train
    bump, calls = [0.0], []
    setattr(ept_train, '_clock', lambda: time.perf_counter() + bump[0])        # the clock of the time bound of the trainer; this child only

    def e6(*, processes=None, probe=None) -> dict:
        calls.append(1)
        if len(calls) == 2:
            bump[0] += 90_000.0
        return dict(E6_RECORD)
    guards.e6 = e6
    run_identity.verify = _recording_verify(run_dir)
    runners.train_stage(run_dir, *arguments)
