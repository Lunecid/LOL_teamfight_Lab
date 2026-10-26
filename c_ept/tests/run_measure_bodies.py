"""Stage bodies for the measurement-runner tests (``c_ept/tests/test_run_measure.py``, S4b-prep K6c): what a child process of the supervisor imports by name.

Not a test module (no ``test_`` prefix, nothing here is collected). The bodies run the real stage bodies of ``c_ept.run.measure`` after they have replaced two checks of this child
that a test run cannot pass: the E6 check (the machine of a test run is not asked about its processes, and a GPU campaign may be running on it) and the identity check (the identity of a
test run is made up). ``perturbed_call_stage`` is the body of a child whose training nudges one parameter by one ulp at a chosen step of a chosen call: the injected non-determinism that
the measurement must find (the environment variable ``K6C_PERTURB`` is ``<call>:<step>``). The children see ``CUDA_VISIBLE_DEVICES=''``.

해석: 감독자 시험의 자식 프로세스가 이름으로 불러 쓰는 측정 단계 본문이다. 진짜 본문을 부르되 이 자식의 E6 확인과 정체성 확인만 시험용으로 바꾼다. 비결정성 주입 본문은 정해진 호출의
정해진 걸음에서 매개변수 하나를 1 ulp 흔든다.
"""
from __future__ import annotations

import os

import torch

from c_ept.ept import training_step
from c_ept.run import guards, measure
from c_ept.run import identity as run_identity

E6_RECORD = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 5, 'cmdline_unreadable': 0, 'rule': 'test rule'}
PERTURB_VARIABLE = 'K6C_PERTURB'


def quiet_e6(*, processes=None, probe=None) -> dict:
    return dict(E6_RECORD)


def quiet_verify(saved, *, repo=None) -> None:
    return None


def quiet_child() -> None:
    guards.e6 = quiet_e6                                              # this child only
    run_identity.verify = quiet_verify


def stats_stage(run_dir, *arguments) -> None:
    measure.stats_stage(run_dir, *arguments)


def call_stage(run_dir, *arguments) -> None:
    quiet_child()
    measure.call_stage(run_dir, *arguments)


def one_ulp_up(parameter) -> None:
    """The first element of ``parameter`` moves to the next float32 number: the smallest change that a race in a kernel could make."""
    with torch.no_grad():
        flat = parameter.view(-1)
        flat[0] = torch.nextafter(flat[0], torch.tensor(float('inf'), dtype=flat.dtype))


def perturbed_call_stage(run_dir, *arguments) -> None:
    """The call stage of a child that nudges the first parameter of the model by one ulp after the optimiser step ``step`` of the call ``call``, when ``K6C_PERTURB`` is ``call:step``
    (the number of the call is the last stage argument); every other call is the real stage body."""
    quiet_child()
    call, _, step = os.environ.get(PERTURB_VARIABLE, '0:0').partition(':')
    if int(call) == int(arguments[-1]):
        real, seen = training_step.optimize_batch, []

        def nudging(fa, *args, **kwargs):
            total = real(fa, *args, **kwargs)
            seen.append(1)
            if len(seen) == int(step):
                one_ulp_up(next(fa.parameters()))
            return total
        training_step.optimize_batch = nudging                         # the recorder of the measurement wraps whatever is there when it starts
    measure.call_stage(run_dir, *arguments)
