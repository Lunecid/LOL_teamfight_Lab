"""Ordered prefetch of the batches of a training run, for the run kernel (EPT-T012, S4b speed-ups SPD, author decision D32, item 2).

The kernel's training runner used to assemble every FIT and STOP batch inline: the main thread, which drives the device, stopped between two steps to read the memory-mapped store, standardise the
views and cut the rows of 13 blocks (about 5 ms per batch on a page-cached toy store, more on a cold one). ``Prefetch`` makes the next batches in one producer thread while the device works on the
current one. It is the ordered prefetch that T009 had (``comparison/infrastructure.py``), written again here because the run kernel may not import the comparison stack (plan Q8); it changes no
number.

방법. ``Prefetch(source, depth)`` wraps a ``data.Batches`` (any object with ``epoch(epoch, start=0)``, ``len`` and the attributes that ``ept.train`` reads: all of them are forwarded). ``epoch(e)`` returns
an iterator whose items are those of ``source.epoch(e)`` in the same order, made by ONE producer thread that is at most ``depth`` batches ahead (queued or in the making). An optional ``transform`` (the CPU
work of ``batch_prep``) is applied to each batch in that thread. Nothing else is shared with the main thread but a queue and two events.

식 (식 번호는 ``(PRE-n)``)

    (PRE-1)  out_j = f(b_j),  b_j = source.epoch(e)[j]  for j = 0, 1, ...   (the same items, the same order; f the transform or the identity)
    (PRE-2)  A_j = made(j) − taken(j) ≤ D   at every moment  (made = batches whose production has begun, taken = batches the consumer holds)
    (PRE-3)  a failure at batch j raises in the consumer at its j-th next(), after the items 0 .. j−1

      기호   | 뜻                                                                 | 단위 | 범위     | 코드 변수
      b_j    | 원천(Batches)이 내는 j번째 배치                                     | —    | 텐서의 사전 | next(source iterator)
      f      | 생산자 스레드에서 쓰는 변환(없으면 항등)                            | —    | —        | transform
      D      | 앞서 만들어 두는 배치 수 (실행 상수, 기록한다)                       | 개   | 1 .. 8   | depth, DEPTH
      A_j    | 만들었으나 소비자가 아직 가져가지 않은 배치 수 (만드는 중인 것 포함)  | 개   | 0 .. D   | the semaphore of the slots

해석
----
(PRE-1): 배치의 내용과 순서는 시드·에폭·배치 번호만의 함수이고(``Batches``는 전역 난수를 쓰지 않는다: 동전도 ``SeedSequence([seed, epoch, batch, 0x5157])``에서 나온다), 생산자는 한 스레드뿐이라서 큐의 순서가
원천의 순서이다. 그래서 학습의 난수 흐름(드롭아웃 마스크)과 비트는 그대로이다. (PRE-2): 생산자는 슬롯을 먼저 받고 나서 배치를 만들므로(소비자가 가져갈 때 슬롯을 돌려준다) 메모리는 D개 배치를 넘지 않는다.
(PRE-3): 생산자의 예외는 큐에 순서대로 들어가므로 그 배치 앞의 배치는 모두 소비자에게 간 뒤에 같은 예외가 소비자 스레드에서 올라온다(원인 추적은 그대로 남는다).

종료. ``close()``는 취소 표시를 세우고 생산자를 합친다(join). 생산자는 배치를 하나 만드는 중이 아니면 0.05 s 안에 끝나고, 만드는 중이면 그 배치가 끝난 뒤 끝난다(스레드는 강제로 죽일 수 없다).
합치지 못하면 ``PrefetchError``이며 대체 생산자를 시작하지 않는다. 소비자의 ``next()``는 0.05 s 간격으로 깨어나므로 KeyboardInterrupt가 막히지 않는다. 방치된 감싸개(와 그 반복자)는 수거될 때 생산자를
취소한다(``weakref.finalize``; 생산자 스레드는 반복자에 대한 참조를 갖지 않는다). 살아 있는 감싸개 안에서는 ``Prefetch._current``가 돌고 있는 반복자를 붙들고 반복자의 ``_owner``가 감싸개를 가리키므로 반복자만
버려서는 수거되지 않는다: 그 반복자는 ``close()``(러너의 ``close_all``)가 닫을 때까지 남는다 (검토 m4). 러너는 학습이 어떻게 끝나든(성공, 예외, KeyboardInterrupt) ``close_all``로 닫는다.

근거
----
저자 결정 D32 (수치를 바꾸지 않는 속도 개선 1–4), 성능 감사 perf-audit.md 항목 2 (T009는 FIT를 prefetch 2로 감쌌고 S4b 커널은 아니다), 계획 Q8 (커널은 comparison을 가져오지 않는다).
"""
# Ref: c_ept/comparison/infrastructure.py ``OrderedPrefetch`` (T009: one CPU producer, at most two reads outstanding, original order, a slot acquired before the read): the design is the same, written again
#      inside the kernel with a polling consumer, a producer that cannot be forgotten (finalizer) and the CPU transform. The frozen stack is not imported (plan Q8).
from __future__ import annotations

import queue
import threading
import weakref
from collections.abc import Callable

DEPTH = 2                                        # the run constant: batches made ahead of the one in use (queued or in the making); T009 used the same value (settings['prefetch'] = 2)
MAX_DEPTH = 8
JOIN_SECONDS = 30.0                              # how long close() waits for a producer that is making a batch
POLL_SECONDS = 0.05                              # the period at which a waiting thread looks at the stop flag (a Ctrl+C or a cancel is seen within it)
THREAD_NAME = 'ept-run-prefetch'
_VALUE, _ERROR, _END = 'value', 'error', 'end'


class PrefetchError(RuntimeError):
    """The wrapper cannot do what it was asked to: a second epoch while one is running, a producer that cannot be stopped, a producer that ended without an end marker."""


def describe(depth: int = DEPTH) -> dict:
    """The record of the run constant that goes into the ledger header and the training record (``runners.prefetch_record`` adds what the producers do): ``{'depth', 'producers', 'ordered'}``."""
    return {'depth': depth, 'producers': 1, 'ordered': True}


class _Shared:
    """What the producer and the consumer share (nothing else: the producer thread holds no reference to the iterator, so an abandoned wrapper and its iterator can be collected, and the finalizer of the iterator cancels the producer; within a live wrapper ``Prefetch._current`` keeps the running iterator until ``close()``)."""

    def __init__(self, depth: int):
        self.slots = threading.Semaphore(depth)
        self.items = queue.Queue()                           # unbounded: the slots bound the values, the one terminal item needs no slot
        self.cancel = threading.Event()
        self.done = threading.Event()


def _produce(shared: _Shared, iterator, transform) -> None:
    """The body of the producer thread: take a slot, make the next batch (``next`` then ``transform``), queue it; at the end or at an error queue the terminal item once."""
    terminal = None
    try:
        while not shared.cancel.is_set():
            if not shared.slots.acquire(timeout=POLL_SECONDS):
                continue
            if shared.cancel.is_set():
                break
            try:
                try:
                    item = next(iterator)
                except StopIteration:
                    terminal = (_END, None)
                    break
                if transform is not None:
                    try:
                        item = transform(item)
                    except StopIteration as exc:               # a transform must not end the epoch (it would look like the end of the source)
                        raise PrefetchError('the transform of the batches raised StopIteration') from exc
            except BaseException as exc:                       # also KeyboardInterrupt/SystemExit of the source: delivered to the consumer, never lost
                terminal = (_ERROR, exc)
                break
            shared.items.put((_VALUE, item))
    finally:
        try:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
        except BaseException as exc:                           # a source that cannot close is an error too, unless one was reported already
            if terminal is None or terminal[0] == _END:
                terminal = (_ERROR, exc)
        if terminal is not None and not shared.cancel.is_set():
            shared.items.put(terminal)
        shared.done.set()


class _Iterator:
    """The iterator of one epoch: one producer thread, items in the order of the source."""

    def __init__(self, owner: 'Prefetch', iterator):
        self._owner = owner
        self._shared = _Shared(owner.depth)
        self._finished = False
        self._joined = False
        self._thread = threading.Thread(target=_produce, args=(self._shared, iterator, owner.transform), name=THREAD_NAME, daemon=True)
        self._finalizer = weakref.finalize(self, self._shared.cancel.set)
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        shared = self._shared
        while True:
            try:
                kind, value = shared.items.get(timeout=POLL_SECONDS)
                break
            except queue.Empty:
                if shared.done.is_set() and shared.items.empty():          # the producer ended and left nothing: it died without a word
                    self._finish()
                    raise PrefetchError('the producer thread ended without an end marker') from None
        if kind == _VALUE:
            shared.slots.release()
            return value
        self._finish()
        if kind == _ERROR:
            raise value
        raise StopIteration

    def _finish(self) -> None:
        """The epoch is over (end, error or close): stop the producer and free the wrapper for the next epoch."""
        self._finished = True
        self._shared.cancel.set()
        waited = 0.0
        while self._thread.is_alive() and waited < self._owner.join_seconds:       # joined in short slices: a Ctrl+C is seen between them (a long join is not interruptible on Windows)
            self._thread.join(timeout=POLL_SECONDS)
            waited += POLL_SECONDS
        if not self._thread.is_alive():
            self._joined = True
            self._owner._release(self)

    def close(self, *, quiet: bool = False) -> bool:
        """Stop the producer and wait for it; True when it has ended. A producer that is still making a batch after ``join_seconds`` cannot be forced: ``PrefetchError`` (or False with ``quiet``), and the wrapper
        stays busy, so that no second producer is ever started beside it."""
        if not self._joined:
            self._finish()
        if not self._joined:
            if quiet:
                return False
            raise PrefetchError(f'the producer thread did not stop within {self._owner.join_seconds:g} s (a batch is still being made); no replacement is started')
        return True


class Prefetch:
    """``Prefetch(source, depth=DEPTH, *, transform=None, join_seconds=JOIN_SECONDS)``: the batches of ``source.epoch(e)`` made ``depth`` ahead by one producer thread (module text).

    ``epoch(epoch, start=0)`` is the iterator of one epoch (the call of the source is made here, in the calling thread, so a refused ``epoch`` or ``start`` fails here as it does without the wrapper; the
    batches are made by the producer). ``len`` and every attribute of the source (``rows``, ``stats``, ``batch_size``, ``seed``, ``training``, ``swap_mode``, ...) are those of the source. The wrapper is not
    iterable, as ``Batches`` is not (no default epoch). Only one epoch can run at a time (``PrefetchError``). ``close()`` (or ``with``) stops a running epoch. ``transform`` is called with each batch in the
    producer thread and its result is the item. ``source`` is the wrapped object (a test can read the plain batches from it).
    """

    def __init__(self, source, depth: int = DEPTH, *, transform: Callable | None = None, join_seconds: float = JOIN_SECONDS):
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= MAX_DEPTH:
            raise ValueError(f'the prefetch depth must be a whole number in 1 .. {MAX_DEPTH}, not {depth!r}')
        if transform is not None and not callable(transform):
            raise TypeError('transform must be None or a function of one batch')
        if isinstance(join_seconds, bool) or not isinstance(join_seconds, (int, float)) or not join_seconds > 0:
            raise ValueError('join_seconds must be a positive number of seconds')
        self.source = source
        self.depth, self.transform, self.join_seconds = depth, transform, float(join_seconds)
        self._busy = threading.Lock()
        self._current = None

    def __getattr__(self, name):                                 # only for names that the wrapper does not have
        if name.startswith('_') or name == 'source':
            raise AttributeError(name)
        return getattr(self.source, name)

    def __len__(self) -> int:
        return len(self.source)

    def epoch(self, epoch, start=0) -> _Iterator:
        if not self._busy.acquire(blocking=False):
            raise PrefetchError('only one epoch can be prefetched at a time: the iterator of the last call is still open (close it first)')
        try:
            iterator = iter(self.source.epoch(epoch, start=start))
            self._current = _Iterator(self, iterator)
            return self._current
        except BaseException:
            self._busy.release()
            raise

    def _release(self, iterator: _Iterator) -> None:
        if self._current is iterator:
            self._current = None
            self._busy.release()

    def close(self, *, quiet: bool = False) -> bool:
        """Close the epoch that is running, if any; True when no producer is left."""
        current = self._current
        return True if current is None else current.close(quiet=quiet)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(quiet=exc_type is not None)                  # never hide the error that is on its way out
        return False


def close_all(*wrappers, quiet: bool = False) -> list:
    """Close every wrapper of ``wrappers`` that has ``close``; the list of those whose producer could not be stopped (empty when all ended). With ``quiet`` nothing is raised, so that the call can stand in a
    ``finally`` without hiding the exception that is leaving; otherwise the first failure is raised after every wrapper has been tried."""
    stuck, first = [], None
    for wrapper in wrappers:
        close = getattr(wrapper, 'close', None)
        if close is None:
            continue
        try:
            if close(quiet=True) is False:
                stuck.append(wrapper)
        except BaseException as exc:                             # a failing close must not keep the other wrappers open
            stuck.append(wrapper)
            first = first or exc
    if stuck and not quiet:
        raise first if first is not None else PrefetchError('a producer thread could not be stopped')
    return stuck
