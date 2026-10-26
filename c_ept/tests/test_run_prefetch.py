"""Tests of the ordered prefetch of the run kernel (``c_ept.run.prefetch``; EPT-T012 S4b speed-ups SPD, author decision D32, item 2).

What is pinned, by behaviour:

* the batches of ``Prefetch(batches).epoch(e)`` are those of ``batches.epoch(e)``, in the same order and bit for bit, for the mode 'random_view' too (the coins are those of the batch), also from a
  ``start`` on and with a transform; the global random streams (torch, numpy, python) are not touched;
* the queue is bounded: at most ``depth`` batches are made ahead of the one the consumer has taken;
* a failure of the producer (the source or the transform) is raised in the consumer at the batch where it happened, after the batches before it; the same exception object, then the end;
* the producer stops on ``close``, on leaving ``with`` by an exception (also a KeyboardInterrupt), on an abandoned iterator that is collected, and a waiting consumer can be interrupted;
* a producer that cannot be stopped is reported and never replaced; only one epoch runs at a time; the wrapper forwards what ``ept.train`` reads (``rows``, ``stats``, ``len``, ...).

The sources are plain generators except where the test needs the real ``Batches`` of the toy EPT world of ``ept_toys``. No real store, no GPU and no cohort module is read.
"""
# Ref: c_ept/comparison/infrastructure.py OrderedPrefetch (the T009 design that this module writes again inside the kernel, plan Q8) and the audit perf-audit.md item 2.
from __future__ import annotations

import gc
import random
import threading
import time
import _thread

import numpy as np
import pytest
import torch

from c_ept.ept import data
from c_ept.run import prefetch
from c_ept.tests.ept_toys import build_ept_world


def producers() -> list:
    return [t for t in threading.enumerate() if t.name == prefetch.THREAD_NAME and t.is_alive()]


def settle(predicate, seconds=5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture(autouse=True)
def no_producer_is_left_behind():
    yield
    assert settle(lambda: not producers()), 'a prefetch producer thread is still alive after the test'


class Counting:
    """A source that counts the batches it has made, so that a test can see how far ahead the producer is; ``fail_at`` raises there."""

    def __init__(self, n, fail_at=None, error=None):
        self.n, self.fail_at, self.error = n, fail_at, error or RuntimeError('boom')
        self.made, self.closed, self.calls = 0, 0, []
        self.rows = 'the rows'
        self.training = True
        self._secret = 'a private name of the source'

    def __len__(self):
        return self.n

    def epoch(self, epoch, start=0):
        self.calls.append((epoch, start))

        def generator():
            try:
                for j in range(start, self.n):
                    if j == self.fail_at:
                        raise self.error
                    self.made += 1
                    yield {'epoch': epoch, 'index': j}
            finally:
                self.closed += 1
        return generator()


# ---- the order and the contents --------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp('prefetch_world')
    w = build_ept_world(folder, n_events=7, n_edges=5, consistent_first_kill=True)
    yield w
    del w
    gc.collect()


@pytest.fixture(scope='module')
def batches(world):
    stats = data.fit_stats(world.s1, world.s2)
    rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT'))
    return data.Batches(rows, 7, 7, stats=stats, training=True, swap_mode='random_view')


def same_batch(a, b, where=''):
    assert list(a) == list(b), where
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), (where, key)


def test_the_batches_are_those_of_the_source_in_the_same_order_and_bit_for_bit(batches):
    """The coins of 'random_view' are in the batch (``swap_view``), so equal batches mean equal coins; three epochs, so another order each time."""
    wrapper = prefetch.Prefetch(batches)
    for epoch in range(3):
        plain = list(batches.epoch(epoch))
        ahead = list(wrapper.epoch(epoch))
        assert len(plain) == len(ahead) == len(batches) == len(wrapper) > 2
        for j, (a, b) in enumerate(zip(plain, ahead)):
            assert 'swap_view' in b
            same_batch(a, b, f'epoch {epoch} batch {j}')
    from_two = list(wrapper.epoch(1, start=2))
    assert len(from_two) == len(batches) - 2 and all(torch.equal(a['view'], b['view']) for a, b in zip(list(batches.epoch(1, start=2)), from_two))


def test_prefetching_draws_no_random_number_from_the_global_streams(batches):
    """No change of the random-number consumption: the global streams of torch, numpy and python are where they were after a whole epoch (and the dropout of a training that uses them is not moved)."""
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    wrapper = prefetch.Prefetch(batches, transform=lambda b: b)
    assert len(list(wrapper.epoch(0))) == len(batches)
    after = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    assert torch.equal(before[0], after[0]) and before[2] == after[2]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]


def test_the_transform_runs_in_the_producer_thread_and_its_result_is_the_item():
    seen = []
    source = Counting(4)
    wrapper = prefetch.Prefetch(source, transform=lambda b: (seen.append(threading.current_thread().name), {**b, 'done': True})[1])
    out = list(wrapper.epoch(0))
    assert [b['index'] for b in out] == [0, 1, 2, 3] and all(b['done'] for b in out)
    assert seen == [prefetch.THREAD_NAME] * 4 and threading.current_thread().name not in seen


def test_the_source_is_called_in_the_calling_thread_so_a_refused_epoch_fails_at_once():
    class Strict:
        def epoch(self, epoch, start=0):
            if epoch < 0:
                raise ValueError('epoch must be non-negative')
            return iter([])

        def __len__(self):
            return 0

    wrapper = prefetch.Prefetch(Strict())
    with pytest.raises(ValueError, match='non-negative'):
        wrapper.epoch(-1)
    assert list(wrapper.epoch(0)) == []                                    # and the wrapper is not left busy by the refusal


# ---- the bound ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('depth', [1, 2, 3])
def test_at_most_depth_batches_are_made_ahead_of_the_consumer(depth):
    source = Counting(20)
    wrapper = prefetch.Prefetch(source, depth)
    iterator = wrapper.epoch(0)
    assert settle(lambda: source.made == depth)
    time.sleep(0.2)
    assert source.made == depth                                              # the producer waits for a free slot: nothing is made beyond the bound
    for taken in range(1, 6):
        next(iterator)
        assert settle(lambda: source.made == taken + depth)
        time.sleep(0.1)
        assert source.made == taken + depth                                  # one more slot, one more batch (never more than ``depth`` ahead)
    wrapper.close()


@pytest.mark.parametrize('bad', [0, 9, -1, 1.0, True, None, '2'])
def test_the_depth_is_a_small_whole_number(bad):
    with pytest.raises(ValueError, match='depth'):
        prefetch.Prefetch(Counting(1), bad)
    assert prefetch.DEPTH == 2 and prefetch.Prefetch(Counting(1)).depth == 2 and prefetch.describe() == {'depth': 2, 'producers': 1, 'ordered': True}


# ---- failures ------------------------------------------------------------------------------------------------------------------------------------
def test_a_failure_of_the_source_is_raised_at_the_batch_where_it_happened_after_the_batches_before_it():
    error = RuntimeError('boom at 3')
    source = Counting(8, fail_at=3, error=error)
    iterator = prefetch.Prefetch(source, 2).epoch(0)
    assert [next(iterator)['index'] for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError) as raised:
        next(iterator)
    assert raised.value is error                                              # the very exception, with its own traceback
    with pytest.raises(StopIteration):
        next(iterator)
    assert settle(lambda: not producers()) and source.closed == 1


def test_a_failure_of_the_transform_is_raised_at_its_batch_and_a_transform_cannot_end_the_epoch():
    def transform(batch):
        if batch['index'] == 2:
            raise KeyError('bad batch')
        return batch

    iterator = prefetch.Prefetch(Counting(6), transform=transform).epoch(0)
    assert [next(iterator)['index'] for _ in range(2)] == [0, 1]
    with pytest.raises(KeyError, match='bad batch'):
        next(iterator)

    def stops(batch):
        raise StopIteration

    with pytest.raises(prefetch.PrefetchError, match='StopIteration'):
        list(prefetch.Prefetch(Counting(3), transform=stops).epoch(0))          # not a silent end of the epoch


def test_a_keyboard_interrupt_of_the_source_is_delivered_too():
    with pytest.raises(KeyboardInterrupt):
        list(prefetch.Prefetch(Counting(4, fail_at=1, error=KeyboardInterrupt())).epoch(0))


# ---- the end of the producer -----------------------------------------------------------------------------------------------------------------------------
def test_close_stops_the_producer_closes_the_source_and_frees_the_wrapper():
    source = Counting(50)
    wrapper = prefetch.Prefetch(source)
    iterator = wrapper.epoch(0)
    next(iterator)
    assert wrapper.close() is True
    assert not producers() and source.closed == 1 and source.made < 50
    with pytest.raises(StopIteration):
        next(iterator)
    assert [b['index'] for b in wrapper.epoch(0, start=48)] == [48, 49]       # free for the next epoch
    assert wrapper.close() is True                                            # closing a wrapper with nothing running is harmless


def test_the_producer_closes_the_iterator_of_the_source_itself_and_does_not_leave_it_to_the_garbage_collector():
    """An iterator that is no generator (a plain object with ``close``) is closed by the producer at the end of the epoch, at an error and when the epoch is closed early."""
    closes = []

    class Plain:
        def __init__(self, n, fail_at=None):
            self.j, self.n, self.fail_at = 0, n, fail_at

        def __iter__(self):
            return self

        def __next__(self):
            if self.j == self.fail_at:
                raise RuntimeError('boom')
            if self.j >= self.n:
                raise StopIteration
            self.j += 1
            return {'index': self.j - 1}

        def close(self):
            closes.append(self.j)

    class Source:
        def __init__(self, **kw):
            self.kw = kw

        def epoch(self, epoch, start=0):
            return Plain(**self.kw)

        def __len__(self):
            return 5

    assert [b['index'] for b in prefetch.Prefetch(Source(n=3)).epoch(0)] == [0, 1, 2] and closes == [3]            # the end
    with pytest.raises(RuntimeError):
        list(prefetch.Prefetch(Source(n=5, fail_at=2)).epoch(0))
    assert closes == [3, 2]                                                                                       # an error
    wrapper = prefetch.Prefetch(Source(n=10_000))
    next(wrapper.epoch(0))
    wrapper.close()
    assert len(closes) == 3 and closes[2] >= 1                                                                    # an early close


def test_a_source_that_cannot_close_is_an_error_of_the_epoch():
    class Bad:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise OSError('cannot close')

    class Source:
        def epoch(self, epoch, start=0):
            return Bad()

        def __len__(self):
            return 0

    with pytest.raises(OSError, match='cannot close'):
        list(prefetch.Prefetch(Source()).epoch(0))


def test_leaving_with_by_an_exception_stops_the_producer_and_keeps_the_exception():
    source = Counting(50)
    with pytest.raises(KeyboardInterrupt):
        with prefetch.Prefetch(source) as wrapper:
            iterator = wrapper.epoch(0)
            next(iterator)
            raise KeyboardInterrupt
    assert settle(lambda: not producers()) and source.closed == 1


def test_a_consumer_that_waits_for_a_slow_producer_can_be_interrupted():
    release = threading.Event()

    class Slow(Counting):
        def epoch(self, epoch, start=0):
            def generator():
                release.wait(10)
                yield {'index': 0}
            return generator()

    wrapper = prefetch.Prefetch(Slow(1), join_seconds=0.3)
    iterator = wrapper.epoch(0)
    timer = threading.Timer(0.3, _thread.interrupt_main)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            next(iterator)
        assert time.monotonic() - started < 3.0                                   # seen within a poll, not after the 10 s of the producer
    finally:
        timer.cancel()
        release.set()
        wrapper.close(quiet=True)
    assert settle(lambda: not producers())


def test_an_abandoned_iterator_that_is_collected_cancels_its_producer():
    source = Counting(10_000)
    wrapper = prefetch.Prefetch(source)
    iterator = wrapper.epoch(0)
    next(iterator)
    wrapper._current = None                                                       # what a dropped wrapper amounts to: nobody holds the iterator any more
    del iterator
    gc.collect()
    assert settle(lambda: not producers()), 'the producer of a collected iterator went on'
    assert source.closed == 1


def test_a_producer_that_cannot_stop_is_reported_and_never_replaced():
    release = threading.Event()

    class Blocked(Counting):
        def epoch(self, epoch, start=0):
            def generator():
                release.wait(20)
                yield {'index': 0}
            return generator()

    wrapper = prefetch.Prefetch(Blocked(1), join_seconds=0.2)
    wrapper.epoch(0)
    assert wrapper.close(quiet=True) is False                                      # still making its batch
    with pytest.raises(prefetch.PrefetchError, match='did not stop'):
        wrapper.close()
    with pytest.raises(prefetch.PrefetchError, match='one epoch'):
        wrapper.epoch(1)                                                          # no second producer beside the stuck one
    assert prefetch.close_all(wrapper, quiet=True) == [wrapper]
    with pytest.raises(prefetch.PrefetchError):
        prefetch.close_all(wrapper)
    release.set()
    assert wrapper.close() is True and settle(lambda: not producers())
    assert list(wrapper.epoch(0)) is not None                                      # and the wrapper works again


def test_only_one_epoch_runs_at_a_time():
    wrapper = prefetch.Prefetch(Counting(5))
    first = wrapper.epoch(0)
    with pytest.raises(prefetch.PrefetchError, match='one epoch'):
        wrapper.epoch(1)
    assert [b['index'] for b in first] == [0, 1, 2, 3, 4]                            # the end of the epoch frees the wrapper
    assert [b['index'] for b in wrapper.epoch(1)] == [0, 1, 2, 3, 4]


def test_close_all_closes_every_wrapper_and_skips_what_has_no_close():
    a, b = prefetch.Prefetch(Counting(100)), prefetch.Prefetch(Counting(100))
    a.epoch(0), b.epoch(0)
    assert prefetch.close_all(a, object(), b) == []
    assert not producers()


# ---- what the trainer reads ------------------------------------------------------------------------------------------------------------------------
def test_the_wrapper_forwards_the_attributes_and_the_length_of_the_source_and_is_not_iterable(batches):
    wrapper = prefetch.Prefetch(batches)
    for name in ('rows', 'stats', 'batch_size', 'seed', 'training', 'swap_mode'):
        assert getattr(wrapper, name) is getattr(batches, name), name
    assert len(wrapper) == len(batches)
    with pytest.raises(TypeError):
        iter(wrapper)                                                              # as ``Batches``: there is no default epoch
    with pytest.raises(AttributeError):
        wrapper.no_such_attribute
    source = Counting(1)
    assert source._secret and prefetch.Prefetch(source).rows == 'the rows'
    with pytest.raises(AttributeError):
        prefetch.Prefetch(source)._secret                                          # a private name of the source is the source's, not the wrapper's
