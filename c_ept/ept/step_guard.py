"""The checks of a training step, split by where they can be made (EPT-T012, S4b speed-ups SPD, author decision D32, item 3).

A step of the EPT used to refuse bad input in the middle of the forward pass: every head and loss function read the range or the finiteness of what it was given back from the device
(``int(t.min())``, ``bool(torch.isfinite(t).all())``), about three hundred reads in a step, and each read waits for the whole device queue. Two kinds of check were mixed:

    data checks      are functions of the batch alone (a category in its range, a count finite and not negative, a view number inside the batch, an exposure not negative): they can be made once,
                     on the CPU, where the batch is made (``batch_prep``), and need not be made again on the device;
    model checks     are functions of what the model computed (a mean that is not negative or finite, a log law that adds up to 1): they can only be made after the forward, but they can be made
                     for the whole step at once.

``StepGuards`` is the second kind collected: while a step runs *trusted* (its batch has passed the data checks on the CPU), the loss functions do not read a model check back; they ``defer`` it, the
number of failing cells as a 0-d tensor on the device, with the message that they would have raised. The step reads all of them together with the finiteness of the loss in ONE transfer, in
``check``, before ``backward``: the first failing guard in the order of the forward raises the ValueError that the check raised before, a loss that is not finite raises the FloatingPointError, and
nothing has been changed (no backward, no optimizer step).

방법. ``trust(guards)`` is a context (``contextvars``: the scope belongs to the thread that entered it, a producer thread or a test in another thread sees none) in which ``current()`` is the
``StepGuards`` and ``trusted()`` is True. The check functions of ``heads`` and ``loss`` ask ``trusted()``: outside the context they are the checks of every earlier run, inside it they skip the data
checks and defer the model checks. ``untrusted()`` is the opposite context, for the dry run of the data checks (``batch_prep``), which must never skip what it is there to make.

식 (식 번호는 ``(GRD-n)``)

    (GRD-1)  step is refused ⇔ ∃ k: g_k > 0   or   ¬finite(L)        g_k = the number of failing cells (rows) of the k-th model check, recorded in the order of the forward
    (GRD-2)  the error is that of the first k with g_k > 0 (ValueError, the message of the check with its count), else FloatingPointError for ¬finite(L)   — as the first failing check was before
    (GRD-3)  one read: v = (g_1, …, g_K, 1[¬finite(L)]) is stacked on the device and moved to the host once

      기호   | 뜻                                                          | 단위 | 범위     | 코드 변수
      g_k    | k번째 모형 점검에서 실패한 칸(행)의 수                       | 개   | ≥ 0      | StepGuards._items[k][0]
      L      | 걸음의 손실 (벌점 포함, 역전파하는 값)                       | —    | 실수     | objective
      v      | 한 번에 읽는 벡터                                            | —    | 정수     | check의 flags

해석
----
걸음이 막히는 조건과 막히는 오류는 이전과 같다: 같은 점검, 같은 문구(실패한 칸 수 포함), 같은 순서, 역전파와 옵티마이저 걸음 앞. 달라지는 것은 읽는 횟수 하나뿐이다(모형 점검마다 한 번에서 걸음마다 한 번).
모형이 잘못된 값을 냈을 때 순전파가 끝까지 도는 것(이전에는 첫 실패에서 멈췄다)은 값에 영향이 없다: 읽기 전에 걸음이 끝나지 않고, 실패한 걸음의 가중은 바뀌지 않는다. 데이터 점검을 건너뛰는 것은
그 점검이 CPU에서 같은 함수로 이미 끝났을 때만이다(``batch_prep.Preparation``이 믿을 수 있는 배치만 만든다). 두 군데에서 동시에 잘못된 경우(자료와 모형)에는 어느 오류가 먼저 나오는지가
이전과 다를 수 있다(둘 다 같은 걸음에서 닫힌다).

근거
----
저자 결정 D32, 성능 감사 perf-audit.md 항목 3 (자료 점검 CPU에서 한 번, 모형 점검 장치 깃발 하나).
"""
# Ref: the performance audit of the S4b hot path (perf-audit.md, item 3): the data checks once per batch on the CPU, the model-output guards as one device flag read once per step before backward.
from __future__ import annotations

import contextlib
import contextvars

import torch

_CURRENT: contextvars.ContextVar = contextvars.ContextVar('c_ept_step_guards', default=None)


class StepGuards:
    """The model checks of one step, deferred: ``defer(count, message)`` in the order of the forward, ``check(finite, step)`` once before backward (module text)."""

    __slots__ = ('_items',)

    def __init__(self):
        self._items: list = []

    def __len__(self) -> int:
        return len(self._items)

    def defer(self, count: torch.Tensor, message: str) -> None:
        """Record a check that would have raised ``ValueError(message.format(n))`` for n = ``count`` > 0: ``count`` a 0-d integer tensor, the number of failing cells (or rows, or 0/1 for a check without a
        count); ``message`` its text, with ``{}`` where the count goes (a text without it is used as it is)."""
        self._items.append((count.detach().reshape(()).to(torch.int64), message))

    def check(self, finite: torch.Tensor, step: int) -> None:
        """Read every deferred count and ``finite`` (0-d bool: the loss is finite) in one transfer and raise for the first failure: the ValueError of the first deferred check with a count above 0,
        else FloatingPointError(f'the training loss is not finite at step {step}'). Returns None when nothing failed."""
        flags = torch.stack([count for count, _ in self._items] + [(~finite.detach()).reshape(()).to(torch.int64)]).tolist()
        for (_, message), n in zip(self._items, flags):
            if n:
                raise ValueError(message.format(n))
        if flags[-1]:
            raise FloatingPointError(f'the training loss is not finite at step {step}')


def current():
    """The ``StepGuards`` of the trusted step that is running in this thread, or None."""
    return _CURRENT.get()


def trusted() -> bool:
    """True inside ``trust(...)``: the batch of this step passed the data checks on the CPU (the check functions of ``heads`` and ``loss`` then skip them and defer the model checks)."""
    return _CURRENT.get() is not None


@contextlib.contextmanager
def trust(guards: StepGuards):
    """Run a step trusted: ``current()`` is ``guards`` inside the block (and the previous state again after it, also after an exception)."""
    if not isinstance(guards, StepGuards):
        raise TypeError('trust needs the StepGuards of the step')
    token = _CURRENT.set(guards)
    try:
        yield guards
    finally:
        _CURRENT.reset(token)


@contextlib.contextmanager
def untrusted():
    """Run a block with every check on, whatever the caller's scope says (the dry run of the data checks on the CPU)."""
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)
