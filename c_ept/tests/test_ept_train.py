"""Tests of the EPT training loop ``c_ept/ept/train.py`` (stage S4a Task 5, fix round 1). The contract is ``.superpowers/sdd/2026-10-22-c-ept-s4a/task-5-contract.md`` (v4).

What is pinned, by behaviour:

* the STOP number is a pool over every batch: an evaluation row is the mean of its scored entries, each match weighs 1/n_m inside its size stratum (``StopMeter``), and the
  number does not depend on how the rows are cut into batches;
* the output biases start at FIT marginals, layer by layer, with the shared scalar of the mark head left alone;
* a run is bit-identical for one seed and after a hard interrupt: a child process ends itself with ``os._exit`` at four points and a fresh process resumes;
* a checkpoint is complete only with its ``head.json`` (written last); a torn write is ignored, a damaged one is refused, nothing is deleted;
* the call that a hard interrupt leaves behind is adopted and charged from its last heartbeat; the memory (sampled private bytes) and the 24 h bound are enforced with
  injected samples and clock, never by using the resources.

The model is tiny (d = 16) and the world is the toy EPT world of ``ept_toys`` on the CPU. No real store, no GPU and no cohort module is read.
"""
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md Task 5 and decisions Q8, Q10; docs/superpowers/plans/2026-10-21-c-ept-s3.md P9 (equal row weights in
#      training, 1/n_m in evaluation); the handoff 2026-10-23 section 5.2.
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept.ept import data, encoder, heads, loss, step_guard, train
from c_ept.tests.ept_toys import build_ept_world, toy_static

SEED = 3
EPOCHS = 4
LONG_PATIENCE = 10                                    # larger than any horizon here: no early stop
FIT_BATCHES = 3
SCALES = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}      # unequal on purpose: the fixed scales must be the ones used
GIB = 2 ** 30
CUT_EXIT = 17                                         # the exit code of a child that ended itself with os._exit
REPO = Path(__file__).resolve().parents[2]


# ---- fixtures -------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def restore_torch_state():
    """``train`` changes process-wide settings (threads, deterministic algorithms, the global random streams); give them back after each test."""
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp('train_world')
    w = build_ept_world(folder, n_events=7, n_edges=5)
    yield w
    del w
    gc.collect()                                      # nothing is deleted here: pytest keeps and rotates its own temporary folders


@pytest.fixture(scope='module')
def stats(world):
    return data.fit_stats(world.s1, world.s2)


@pytest.fixture(scope='module')
def rows(world):
    """``EngagementRows`` of the roles that the tests use (read only: a test that changes rows builds its own)."""
    return {role: data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role)) for role in ('FIT', 'STOP', 'DEV')}


class SpyBatches(data.Batches):
    """A real ``Batches`` that writes ``('epoch', name, epoch, start)`` when an epoch starts and ``('batch', name)`` for each batch; ``hook(name, epoch, i)`` is called
    inside ``next()`` just before batch ``i`` of the epoch is yielded."""

    def __init__(self, rows, batch_size, seed, *, stats, training, name, log, hook=None, swap_mode='average'):
        super().__init__(rows, batch_size, seed, stats=stats, training=training, **({} if swap_mode == 'average' else {'swap_mode': swap_mode}))
        self.name, self.log, self.hook = name, log, hook

    def epoch(self, epoch, start=0):
        self.log.append(('epoch', self.name, epoch, start))
        for i, batch in enumerate(super().epoch(epoch, start), start):
            if self.hook is not None:
                self.hook(self.name, epoch, i)
            self.log.append(('batch', self.name))
            yield batch


def make_data(rows, stats, *, log=None, stop_rows=None, static=None, scales=None, fit_batches=FIT_BATCHES, static_source=None, stop_size=None, hook=None, swap_mode='average'):
    """A ``TrainData``: FIT in ``fit_batches`` shuffled batches (made for ``swap_mode``, S4b preparation M7), STOP in one batch (or ``stop_size`` engagements per batch)."""
    log = [] if log is None else log
    fit_rows, stop_rows = rows['FIT'], rows['STOP'] if stop_rows is None else stop_rows
    size = max(1, math.ceil(len(fit_rows.units) / fit_batches))
    fit = SpyBatches(fit_rows, size, 7, stats=stats, training=True, name='fit', log=log, hook=hook, swap_mode=swap_mode)
    stop = SpyBatches(stop_rows, stop_size or max(1, len(stop_rows.units)), 7, stats=stats, training=False, name='stop', log=log, hook=hook)
    return train.TrainData(fit=fit, stop=stop, scales=dict(SCALES if scales is None else scales), static=toy_static() if static is None else static,
                           static_source=static_source)


def make_cfg(**over):
    fields = dict(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), lr=3e-3, weight_decay=0.01,
                  warmup_steps=2, patience=LONG_PATIENCE)
    fields.update(over)
    return train.TrainConfig(**fields)


def run(folder, data_, cfg=None, *, epochs=EPOCHS, seed=SEED, check=None):
    return train.train(make_cfg() if cfg is None else cfg, data_, Path(folder), device='cpu', epochs=epochs,
                       check=(lambda: None) if check is None else check, seed=seed)


@pytest.fixture(scope='module')
def base(tmp_path_factory, rows, stats):
    """One uninterrupted run of EPOCHS epochs (the reference of the identity and resume tests) and its folder."""
    folder = tmp_path_factory.mktemp('train_base') / 'run'
    return run(folder, make_data(rows, stats)), folder


def assert_same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def assert_same_state(a, b, where='state'):
    """Tensors bit for bit, containers element by element, everything else by ``==``."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert list(a) == list(b), where
        for key in a:
            assert_same_state(a[key], b[key], f'{where}.{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_state(x, y, f'{where}[{i}]')
    else:
        assert a == b, where


def assert_same_run(a, b):
    """The returned (best) weights bit for bit, and the recorded curves equal."""
    assert_same_weights(a.model, b.model)
    assert a.fit_batch_mean_losses == b.fit_batch_mean_losses and a.stop_losses == b.stop_losses and a.lrs == b.lrs and a.best_epoch == b.best_epoch


def checkpoints(folder) -> list:
    """``[(head.json content, folder)]`` of every complete checkpoint under ``folder``, by epochs done."""
    found = [(json.loads(p.read_text(encoding='utf-8')), p.parent) for p in Path(folder).rglob('head.json')]
    return sorted(found, key=lambda t: t[0]['epoch'])


def last_pt(folder, epoch: int) -> dict:
    return torch.load(Path(folder) / f'epoch_{epoch:04d}' / 'last.pt', weights_only=True)


def last_state(folder, epoch: int) -> dict:
    return last_pt(folder, epoch)['model']


def file_hashes(folder) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def call_marker(folder, n=1) -> dict:
    return json.loads((Path(folder) / 'calls' / f'call_{n:04d}.json').read_text(encoding='utf-8'))


def beat_lines(path) -> list:
    """The complete lines of a heartbeat file, parsed (a torn last line has no newline and is left out)."""
    return [json.loads(line) for line in Path(path).read_bytes().decode('utf-8').split('\n')[:-1]]


class Cut(Exception):
    """A managed interruption: an exception, which ``train`` finalizes. The hard interrupts are the subprocess tests below."""


def cut_on(call: int, then=None):
    """An E6 guard that raises ``Cut`` (or runs ``then``) on its ``call``-th call (1-based): call 1 is made before the preparation, call k + 1 at the start of epoch k."""
    state = {'n': 0}

    def check():
        state['n'] += 1
        if state['n'] == call:
            if then is None:
                raise Cut
            then()
    return check


def scripted(monkeypatch, curve):
    """Replace the STOP evaluation by a list of losses, one for each call: the selection logic is then tested apart from the numbers."""
    calls = []

    def evaluate(model, stop, scales, device='cpu', guard=None):
        calls.append(stop)
        return float(curve[len(calls) - 1])

    monkeypatch.setattr(train, 'evaluate_stop', evaluate)
    return calls


def no_work(log) -> bool:
    return not [e for e in log if e[0] in ('epoch', 'batch')]


def fit_batches_logged(log) -> int:
    return len([e for e in log if e[:2] == ('batch', 'fit')])


def stop_batches_logged(log) -> int:
    return len([e for e in log if e[:2] == ('batch', 'stop')])


# =====================================================================================================================================
# the STOP number: a pool over all batches, an evaluation row = the mean of its scored entries, the weight 1 / n_m of a match
# =====================================================================================================================================
def ids(*a):
    return np.array(a, dtype=np.int64)


def test_stop_pools_the_rows_of_a_match_across_batches_and_weights_each_match_by_one():
    """Hand case (block M2Zone): match 1 has rows 0, 0, 0 in the first batch and 0 in the second, match 2 has the row 8 in the second batch. The weight of a row is
    1 / n_m with n_m over ALL batches, so the value is the mean of the match means (0 + 8) / 2 = 4. The row pool would give 8 / 5 = 1.6 and the mean of the batch
    means (0 + (0 + 8) / 2) / 2 = 2."""
    meter = train.StopMeter()
    meter.add('M2Zone/masked', np.zeros(3), np.ones(3, bool), ids(1, 1, 1))
    meter.add('M2Zone/masked', np.array([0.0, 8.0]), np.ones(2, bool), ids(1, 2))
    assert meter.value({'M2Zone/masked': 2.0}) == pytest.approx(4.0 / 2.0, rel=1e-12)


def test_stop_unequal_batches_and_uneven_match_counts():
    """Match 1 has 4 scored rows (mean 2), match 2 has 1 (value 5), match 3 has 2 (mean 3): 1/n_m weights give (2 + 5 + 3) / 3 and not the row mean 20 / 7."""
    meter = train.StopMeter()
    meter.add('M2Mark/masked', np.array([1.0, 3.0]), np.ones(2, bool), ids(1, 1))
    meter.add('M2Mark/masked', np.array([5.0, 2.0, 4.0]), np.ones(3, bool), ids(2, 1, 3))
    meter.add('M2Mark/masked', np.array([2.0, 2.0]), np.ones(2, bool), ids(1, 3))
    assert meter.value({'M2Mark/masked': 1.0}) == pytest.approx((2.0 + 5.0 + 3.0) / 3.0, rel=1e-12)


def test_rows_without_a_score_change_neither_numerator_nor_denominator():
    meter = train.StopMeter()
    meter.add('M2Zone/masked', np.array([0.4, 1e9, 1e9]), np.array([True, False, False]), ids(1, 1, 2))
    meter.add('M2Zone/masked', np.array([1e9]), np.array([False]), ids(3))
    assert meter.value({'M2Zone/masked': 2.0}) == pytest.approx(0.2, rel=1e-12)                 # only match 1 has a scored row; matches 2 and 3 do not count


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_stop_value_does_not_depend_on_the_cut_into_batches(seed):
    rng = np.random.default_rng(seed)
    n = 60
    nll, scored, match = rng.random(n) * 5, rng.random(n) < 0.8, rng.integers(0, 5, n)
    size = rng.integers(0, 4, n)
    whole = train.StopMeter()
    whole.add('M3Kills/masked', nll, scored, match, size=size)
    whole.add('M2Zone/masked', nll, scored, match)
    order = rng.permutation(n)
    cut = train.StopMeter()
    for part in np.array_split(order, 7):
        cut.add('M3Kills/masked', nll[part], scored[part], match[part], size=size[part])
        cut.add('M2Zone/masked', nll[part], scored[part], match[part])
    scales = {'M3Kills/masked': 1.5, 'M2Zone/masked': 0.5}
    assert cut.value(scales) == pytest.approx(whole.value(scales), rel=1e-12)


def test_the_experts_average_the_observed_size_strata_and_the_gate_does_not():
    """M3Kills: size 0 has rows 1 and 3 of match 1 (mean 2) and the row 10 of match 2; size 1 has the row 4 of match 1. Stratum 0 is (2 + 10) / 2 = 6, stratum 1
    is 4, sizes 2 and 3 do not occur: (6 + 4) / 2 = 5. The size gate has one stratum: (match 1: (1 + 3 + 4) / 3, match 2: 10) -> (8/3 + 10) / 2."""
    nll, match, size = np.array([1.0, 3.0, 10.0, 4.0]), ids(1, 1, 2, 1), ids(0, 0, 0, 1)
    experts, gate = train.StopMeter(), train.StopMeter()
    experts.add('M3Kills/masked', nll, np.ones(4, bool), match, size=size)
    gate.add('M3Size/masked', nll, np.ones(4, bool), match)
    assert experts.value({'M3Kills/masked': 1.0}) == pytest.approx(5.0, rel=1e-12)
    assert gate.value({'M3Size/masked': 1.0}) == pytest.approx((8.0 / 3.0 + 10.0) / 2.0, rel=1e-12)


def test_the_size_strata_are_pooled_across_batches_too():
    meter = train.StopMeter()
    meter.add('M3Obj/open', np.array([2.0]), np.ones(1, bool), ids(1), size=ids(2))
    meter.add('M3Obj/open', np.array([4.0, 9.0]), np.ones(2, bool), ids(1, 7), size=ids(2, 3))
    assert meter.value({'M3Obj/open': 1.0}) == pytest.approx((3.0 + 9.0) / 2.0, rel=1e-12)


def test_a_block_without_a_scored_row_is_left_out_of_the_sum():
    meter = train.StopMeter()
    meter.add('M2Zone/masked', np.array([5.0]), np.array([False]), ids(1))
    meter.add('M2Mark/masked', np.array([1.0, 2.0]), np.ones(2, bool), ids(1, 2))
    assert meter.value({'M2Zone/masked': 1.0, 'M2Mark/masked': 2.0}) == pytest.approx(0.75, rel=1e-12)
    with pytest.raises(ValueError, match='nothing to average'):
        train.StopMeter().value(SCALES)                                                             # nothing at all


def test_the_sum_over_blocks_follows_the_scales_and_the_order_of_the_blocks():
    meter = train.StopMeter()
    for i, key in enumerate(loss.BLOCKS):
        size = ids(0, 1) if loss.split_key(key)[0] in loss.SIZE_NORMALISED else None
        meter.add(key, np.array([1.0 + i, 3.0 + i]), np.ones(2, bool), ids(1, 1), size=size)
    expect = 0.0
    for i, key in enumerate(loss.BLOCKS):
        expect += (2.0 + i) / SCALES[key]                         # the experts: the strata have the means 1 + i and 3 + i, and their mean is 2 + i
    assert meter.value(SCALES) == pytest.approx(expect, rel=1e-12)


@pytest.mark.parametrize('call', [
    dict(block='M9/masked'), dict(block='M2Zone/open'), dict(block='M3Kills/masked', size=None), dict(block='M2Zone/masked', size=ids(0)),
    dict(block='M3Kills/masked', size=ids(4)), dict(block='M2Zone/masked', nll=np.zeros(2)), dict(block='M2Zone/masked', scored=np.ones(1)),
    dict(block='M2Zone/masked', match=ids(1, 2))])
def test_the_meter_refuses_what_it_cannot_pool(call):
    args = dict(block='M2Zone/masked', nll=np.zeros(1), scored=np.ones(1, bool), match=ids(1), size=None)
    args.update(call)
    with pytest.raises((ValueError, TypeError)):
        train.StopMeter().add(args['block'], args['nll'], args['scored'], args['match'], size=args['size'])


def test_the_stop_rule_normalises_the_matches_inside_each_size_stratum_not_inside_the_whole_head():
    """R-S4a-16, the synthetic discriminator. Match A: one size-0 row of loss 0 and nine size-1 rows of loss 100. Match B: nine size-0 rows of loss 10 and one size-1
    row of loss 0. Inside a stratum the matches weigh the same: stratum 0 is (0 + 10) / 2 = 5, stratum 1 is (100 + 0) / 2 = 50, the strata are averaged: 27.5. A weight of
    1/10 for every row of a match over the whole head would give 9 and 90 and 49.5."""
    nll = np.array([0.0] + [100.0] * 9 + [10.0] * 9 + [0.0])
    match = np.array([1] * 10 + [2] * 10, dtype=np.int64)
    size = np.array([0] + [1] * 9 + [0] * 9 + [1], dtype=np.int64)
    meter = train.StopMeter()
    meter.add('M3Kills/masked', nll, np.ones(20, bool), match, size=size)
    assert meter.value({'M3Kills/masked': 1.0}) == pytest.approx(27.5, rel=1e-12)
    assert meter.value({'M3Kills/masked': 2.0}) == pytest.approx(13.75, rel=1e-12)
    for per_call in (1, 3, 20):                                                                 # the same number for any cut into batches
        cut = train.StopMeter()
        for i in range(0, 20, per_call):
            part = slice(i, i + per_call)
            cut.add('M3Kills/masked', nll[part], np.ones(len(nll[part]), bool), match[part], size=size[part])
        assert cut.value({'M3Kills/masked': 1.0}) == pytest.approx(27.5, rel=1e-12)


def test_a_gold_evaluation_row_is_the_mean_of_its_team_entries_before_the_match_weight():
    """R-S4a-16 (modified). One evaluation row (match 1, row 0) with the scored team losses 2 and 8 is worth (2 + 8) / 2 = 5, in the unit of its B0 scale (one team entry).
    Match 1 has a second row 3, match 2 has the rows (1 + 1) / 2 and 5 with unequal counts: match 1 -> (5 + 3) / 2 = 4, match 2 -> (1 + 5) / 2 = 3, value (4 + 3) / 2."""
    nll = np.array([2.0, 8.0, 3.0, 1.0, 1.0, 5.0])
    match = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    row = np.array([0, 0, 1, 0, 0, 1], dtype=np.int64)
    meter = train.StopMeter()
    meter.add('M3Gold/masked', nll, np.ones(6, bool), match, row=row, size=np.zeros(6, dtype=np.int64))
    assert meter.value({'M3Gold/masked': 1.0}) == pytest.approx(3.5, rel=1e-12)
    lone = train.StopMeter()
    lone.add('M3Gold/masked', nll[:2], np.ones(2, bool), match[:2], row=row[:2], size=np.zeros(2, dtype=np.int64))
    assert lone.value({'M3Gold/masked': 1.0}) == pytest.approx(5.0, rel=1e-12)


def test_constant_entry_cost_equal_to_its_scale_makes_each_stop_block_one():
    """Every scored entry costs exactly its B0 scale, and some evaluation rows hold two entries (the team entries of M3Gold): each block, M3Gold included, adds 1, so the
    STOP number is the number of blocks in this constant-score fixture. This is not a universal B0-prediction identity; summing entries would give M3Gold more than 1 here."""
    meter = train.StopMeter()
    for key in loss.BLOCKS:
        size = ids(0, 0, 1, 1, 2) if loss.split_key(key)[0] in loss.SIZE_NORMALISED else None
        meter.add(key, np.full(5, SCALES[key]), np.ones(5, bool), ids(1, 1, 2, 2, 3), row=ids(0, 0, 0, 1, 0), size=size)
    assert meter.value(SCALES) == pytest.approx(len(loss.BLOCKS), rel=1e-12)


def test_the_stop_value_is_the_head_scores_mean_of_the_row_values_that_are_entry_means():
    """``HeadScores.per_row`` sums the entries of an evaluation row (S3); the STOP pool divides that sum by the scored entries, so on rows of one entry it is
    ``HeadScores.mean()`` and on the rows below it is ``HeadScores.mean()`` of the entry means."""
    from c_ept.data import head_rows
    nll = np.array([2.0, 8.0, 3.0, 1.0, 1.0, 5.0, 7.0])
    match = np.array([1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    row = np.array([0, 0, 1, 2, 2, 3, 4], dtype=np.int64)
    sums, per_match = head_rows.HeadScores(nll=nll, match=match, row=row).per_row()
    assert sums.tolist() == [10.0, 3.0, 2.0, 5.0, 7.0] and per_match.tolist() == [1, 1, 2, 2, 2]
    entry_means = head_rows.HeadScores(nll=sums / np.array([2, 1, 2, 1, 1]), match=per_match, row=np.arange(5))
    gold = train.StopMeter()
    gold.add('M3Gold/open', nll, np.ones(7, bool), match, row=row, size=np.zeros(7, dtype=np.int64))
    assert gold.value({'M3Gold/open': 1.0}) == pytest.approx(entry_means.mean(), rel=1e-12)
    single = train.StopMeter()
    single.add('M2Zone/masked', nll, np.ones(7, bool), match, row=np.arange(7))
    assert single.value({'M2Zone/masked': 1.0}) == pytest.approx(head_rows.HeadScores(nll=nll, match=match, row=np.arange(7)).mean(), rel=1e-12)


def test_entries_of_one_evaluation_row_in_different_calls_are_one_row():
    whole = train.StopMeter()
    whole.add('M3Gold/masked', np.array([2.0, 8.0, 3.0]), np.ones(3, bool), ids(1, 1, 1), row=ids(0, 0, 1), size=ids(0, 0, 0))
    split = train.StopMeter()
    split.add('M3Gold/masked', np.array([2.0]), np.ones(1, bool), ids(1), row=ids(0), size=ids(0))
    split.add('M3Gold/masked', np.array([3.0, 8.0]), np.ones(2, bool), ids(1, 1), row=ids(1, 0), size=ids(0, 0))
    assert split.value({'M3Gold/masked': 1.0}) == pytest.approx(whole.value({'M3Gold/masked': 1.0}), rel=1e-12) == pytest.approx(4.0, rel=1e-12)


def test_an_evaluation_row_without_a_scored_entry_is_not_a_row_and_an_unscored_entry_adds_nothing():
    """Row 0 has both entries unscored (junk 1e9): it is not counted. Row 1 holds the scored entries 3 and 4 and an unscored 1e9, which is left out of the sum and
    of the count: (3 + 4) / 2."""
    nll = np.array([1e9, 1e9, 3.0, 4.0, 1e9])
    scored = np.array([False, False, True, True, False])
    meter = train.StopMeter()
    meter.add('M3Gold/masked', nll, scored, ids(1, 1, 1, 1, 1), row=ids(0, 0, 1, 1, 1), size=ids(0, 0, 0, 0, 0))
    assert meter.value({'M3Gold/masked': 1.0}) == pytest.approx(3.5, rel=1e-12)


def test_two_sizes_inside_one_evaluation_row_are_refused():
    meter = train.StopMeter()
    meter.add('M3Gold/masked', np.array([1.0]), np.ones(1, bool), ids(1), row=ids(0), size=ids(0))
    with pytest.raises(ValueError, match='size'):
        meter.add('M3Gold/masked', np.array([1.0]), np.ones(1, bool), ids(1), row=ids(0), size=ids(1))
    with pytest.raises(ValueError, match='size'):
        train.StopMeter().add('M3Gold/masked', np.array([1.0, 1.0]), np.ones(2, bool), ids(1, 1), row=ids(0, 0), size=ids(0, 1))


def test_an_evaluation_row_that_is_missing_from_a_block_is_simply_absent():
    """A block that has no row of a match contributes nothing for that match (no zero is invented); another block still counts it."""
    meter = train.StopMeter()
    meter.add('M2Zone/masked', np.array([4.0]), np.ones(1, bool), ids(1), row=ids(0))
    meter.add('M2Mark/masked', np.array([2.0, 6.0]), np.ones(2, bool), ids(1, 2), row=ids(0, 0))
    assert meter.value({'M2Zone/masked': 1.0, 'M2Mark/masked': 1.0}) == pytest.approx(4.0 + 4.0, rel=1e-12)


# =====================================================================================================================================
# the stop number of a model on real batches, and the blocks of a batch
# =====================================================================================================================================
def block_outputs(fa, enc, batch) -> dict:
    """Independent of ``train``: the 13 outputs of a batch as ``{block: (head, target, out)}``, one head call for each block."""
    outs = {}
    for head, kind in data.ROW_FIELDS:
        b = data.block(batch, head, kind)
        q, m, z = b.get('q'), b.get('m'), b.get('z')
        if head in ('M2Rate', 'M2Mark', 'M2Zone'):
            m2 = fa.m2(enc, b['e'] if head == 'M2Rate' else 1.0, pos=b['pos'])
            out = {'M2Rate': m2.rate, 'M2Mark': m2.mark_logp, 'M2Zone': m2.zone_logp}[head]
        elif head in ('M3Size', 'M3Kills', 'M3Gold'):
            m3 = fa.m3(enc, q, m, z, pos=b['pos'])
            out = {'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate, 'M3Gold': m3.gold_logp}[head]
        elif head == 'M3Obj':
            out = fa.m3_obj(enc, q, m, z, b['s'], b['K'], pos=b['pos'])
        else:
            out = fa.m4(enc, q, m, z, b['K'], b['O'], b['e'], pos=b['pos'])
        outs[loss.block_key(head, kind)] = (head, b, out)
    return outs


def reference_stop(model, batches, scales, *, average=True) -> float:
    """The STOP number by the definition (R-S4a-16), from one pass over the entries: the scored entries of an evaluation row ``(match, row)`` of a block are averaged,
    each evaluation row is weighted 1 / n (n: the evaluation rows of the match in that block and size stratum), the weighted mean is taken in each stratum, the strata
    are averaged, divided by the scale and the blocks added. ``average=False`` evaluates the recorded frame only (S4b preparation M7, ``eval_average=False``)."""
    model.eval()
    fa = heads.FrameAveraged(model) if average else heads.FrameAveraged(model, average=False)
    cols = {}
    with torch.no_grad():
        for batch in batches.epoch(0):
            enc = fa.encode(batch)
            for key, (head, b, out) in block_outputs(fa, enc, batch).items():
                if len(b['y']) == 0:
                    continue
                nll = loss.head_nll(head, out, b).numpy()
                ok = loss.scored_rows(head, b).numpy()
                size = b['s'].numpy() if head in loss.SIZE_NORMALISED else np.zeros(len(nll), dtype=np.int64)
                cols.setdefault(key, []).append((nll[ok], b['match'].numpy()[ok], b['row'].numpy()[ok], size[ok]))
    total = 0.0
    for key in loss.BLOCKS:
        if key not in cols:
            continue
        nll, match, row, size = (np.concatenate([c[i] for c in cols[key]]) for i in range(4))
        if not len(nll):
            continue
        groups = {}
        for v, u, r, s in zip(nll.tolist(), match.tolist(), row.tolist(), size.tolist()):
            g = groups.setdefault((u, r), [0.0, s, 0])
            assert g[1] == s
            g[0] += v
            g[2] += 1
        strata = []
        for s in sorted({g[1] for g in groups.values()}):
            per = {}
            for (u, r), (v, size_g, count) in groups.items():
                if size_g == s:
                    per.setdefault(u, []).append(v / count)
            strata.append(float(np.mean([sum(vs) / len(vs) for vs in per.values()])))
        total += float(np.mean(strata)) / scales[key]
    return total


@pytest.fixture(scope='module')
def tiny_model(stats):
    torch.manual_seed(5)
    cfg = encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2)
    return heads.EPT(cfg, toy_static())


@pytest.mark.parametrize('role', ['FIT', 'DEV'])
@pytest.mark.parametrize('size', [1, 5, None])
def test_evaluate_stop_equals_the_definition_whatever_the_batches(tiny_model, rows, stats, role, size):
    """On real rows of 4 (FIT) or 2 (DEV) matches with uneven counts: the pooled number with batches of 1, 5 or all engagements equals the reference that
    computes the weights 1 / n_m from all rows at once."""
    units = len(rows[role].units)
    batches = data.Batches(rows[role], size or units, 0, stats=stats, training=False)
    want = reference_stop(tiny_model, data.Batches(rows[role], units, 0, stats=stats, training=False), SCALES)
    got = train.evaluate_stop(tiny_model, batches, SCALES)
    assert got == pytest.approx(want, rel=2e-5)


class Fixed:
    """One batch as the ``Batches`` that ``evaluate_stop`` reads (it only calls ``epoch``)."""

    def __init__(self, batch):
        self.batch = batch

    def epoch(self, epoch, start=0):
        yield self.batch


def test_the_stop_number_is_not_the_mean_of_the_numbers_of_the_batches(tiny_model, rows, stats):
    """Guard on the toy rows: the pooled number differs from the mean of the numbers of its own batches, so the equality tests above can tell the two apart."""
    batches = data.Batches(rows['FIT'], 5, 0, stats=stats, training=False)
    pooled = train.evaluate_stop(tiny_model, batches, SCALES)
    per_batch = [train.evaluate_stop(tiny_model, Fixed(b), SCALES) for b in batches.epoch(0)]
    assert len(per_batch) > 1 and abs(pooled - sum(per_batch) / len(per_batch)) > 1e-3 * pooled


def test_evaluation_is_deterministic_draws_no_random_numbers_and_leaves_no_gradient(tiny_model, rows, stats):
    """Dropout is on in the module but off in ``model.eval()``: a second call gives the same bits, the global random stream is not advanced and no parameter
    receives a gradient (the frame-average invariance of the heads holds in evaluation mode only)."""
    tiny_model.train()
    batches = data.Batches(rows['STOP'], 100, 0, stats=stats, training=False)
    before = torch.get_rng_state().clone()
    a = train.evaluate_stop(tiny_model, batches, SCALES)
    b = train.evaluate_stop(tiny_model, batches, SCALES)
    assert a == b and torch.equal(before, torch.get_rng_state())
    assert all(p.grad is None for p in tiny_model.parameters())


def test_m2_is_called_once_for_the_marks_and_zones_on_the_union_of_their_views_and_scores_as_before(tiny_model, rows, stats, monkeypatch):
    """R-S4a-20. M2Rate uses its rate-only evaluation call; M2Mark and M2Zone share one full call on the union of the views they read, and the scores equal
    those of one call for each block."""
    calls, real = [], heads.FrameAveraged.m2
    monkeypatch.setattr(heads.FrameAveraged, 'm2', lambda self, enc, exposure, pos=None: (calls.append(None if pos is None else pos.clone()), real(self, enc, exposure, pos))[1])
    batch = next(iter(data.Batches(rows['FIT'], 100, 0, stats=stats, training=False).epoch(0)))
    fa = heads.FrameAveraged(tiny_model.eval()).eval()
    with torch.no_grad():
        enc = fa.encode(batch)
        scored = train.score_blocks(fa, enc, batch)
        mark, zone = (data.block(batch, h, 'masked')['pos'] for h in ('M2Mark', 'M2Zone'))
        assert len(calls) == 1 and torch.equal(calls[0], torch.unique(torch.cat([mark, zone])))
        _, rate_target, rate_score = scored['M2Rate/masked']
        expected_rate = real(fa, enc, rate_target['e'], pos=rate_target['pos']).rate
        torch.testing.assert_close(rate_score, loss.head_nll('M2Rate', expected_rate, rate_target), rtol=1e-5, atol=1e-6)
        for head, field in (('M2Mark', 'mark_logp'), ('M2Zone', 'zone_logp')):
            _, target, got = scored[f'{head}/masked']
            want = loss.head_nll(head, getattr(real(fa, enc, 1.0, pos=target['pos']), field), target)
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


# =====================================================================================================================================
# the learning-rate schedule
# =====================================================================================================================================
def test_the_schedule_rises_over_the_warmup_then_follows_a_cosine_to_the_horizon():
    lrs = [train.lr_at(i, 10, 2, 1.0) for i in range(10)]
    assert lrs[:2] == [0.5, 1.0]
    assert lrs[2] == pytest.approx(1.0)                                   # the cosine starts at the peak
    assert lrs[6] == pytest.approx(0.5 * (1 + math.cos(math.pi * 4 / 8)), rel=1e-12)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:])) and 0 < lrs[-1] < 0.05
    assert lrs[-1] == pytest.approx(0.5 * (1 + math.cos(math.pi * 7 / 8)), rel=1e-12)          # the last step used is T - 1 = 9: the schedule does not reach 0


def test_a_longer_horizon_decays_more_slowly_and_the_value_depends_on_the_step_only():
    assert train.lr_at(5, 20, 2, 1.0) > train.lr_at(5, 10, 2, 1.0)
    assert train.lr_at(5, 20, 2, 1.0) == train.lr_at(5, 20, 2, 1.0)
    assert train.lr_at(0, 10, 0, 0.3) == pytest.approx(0.3)               # no warm-up: the first step is at the peak
    assert train.lr_at(0, 1, 5, 0.3) == pytest.approx(0.06)               # a horizon shorter than the warm-up never divides by zero


# =====================================================================================================================================
# the output biases
# =====================================================================================================================================
def empty_fields(head: str, kind: str) -> dict:
    width = {'M2Rate': 18, 'M3Kills': 2, 'M3Obj': 18, 'M4Conv': 18}.get(head)
    fields = {'match': np.zeros(0, np.int64), 'y': np.zeros((0, width) if width else (0,), np.int64)}
    if head == 'M2Rate':
        fields['e'] = np.zeros((0, 18), np.float32)
    if head == 'M4Conv':
        fields['e'] = np.zeros(0, np.float32)
    if head in ('M3Kills', 'M3Gold', 'M3Obj'):
        fields['s'] = np.zeros(0, np.int64)
    return fields


def stub_rows(role='FIT', **given):
    """A stand-in for ``EngagementRows`` with the 13 blocks (empty unless given as ``Head_kind=fields``)."""
    blocks = {}
    for (head, kind) in data.ROW_FIELDS:
        fields = empty_fields(head, kind)
        fields.update({k: np.asarray(v) for k, v in given.get(f'{head}_{kind}', {}).items()})
        blocks[(head, kind)] = SimpleNamespace(fields=fields)
    return SimpleNamespace(role=role, blocks=blocks)


def test_a_category_bias_is_the_log_of_the_smoothed_fit_frequency_and_a_missing_category_stays_finite():
    """M2Zone y = 0, 0, 2, -1, 9 (the -1 is not a category): counts 2, 0, 1, ..., 1 over N = 4, alpha = 0.5, 10 zones: p_c = (n_c + 0.5) / 9."""
    got, info = train.fit_biases(stub_rows(M2Zone_masked=dict(y=[0, 0, 2, -1, 9], match=[1] * 5)))
    n = np.zeros(10)
    n[[0, 2, 9]] = [2, 1, 1]
    np.testing.assert_allclose(got['zone_net.out.bias'], np.log((n + train.CATEGORY_PSEUDOCOUNT) / (4 + 10 * train.CATEGORY_PSEUDOCOUNT)), rtol=1e-6)
    assert np.isfinite(got['zone_net.out.bias']).all() and got['zone_net.out.bias'].shape == (10,)
    assert not info['zone_net.out.bias']['fallback']


def test_a_rate_bias_is_the_log_of_counts_over_observed_seconds():
    """M2Rate: two frame rows. Row 0 observes bins 0-2 for 10 s each (1 count), row 1 observes bins 0-1 for 5 s each (2 counts): 3 / 40 per second."""
    y, e = np.zeros((2, 18), np.int64), np.zeros((2, 18), np.float32)
    e[0, :3], e[1, :2] = 10.0, 5.0
    y[0, 0], y[1, 1] = 1, 2
    got, _ = train.fit_biases(stub_rows(M2Rate_masked=dict(y=y, e=e, match=[1, 1])))
    assert got['rate_net.out.bias'].shape == (1,)
    assert got['rate_net.out.bias'][0] == pytest.approx(math.log(3 / 40), rel=1e-6)


def test_a_rate_without_a_count_is_clamped_and_a_rate_without_exposure_is_the_documented_fallback():
    e = np.zeros((1, 18), np.float32)
    e[0, :4] = 10.0
    got, info = train.fit_biases(stub_rows(M2Rate_masked=dict(y=np.zeros((1, 18), np.int64), e=e, match=[1])))
    assert got['rate_net.out.bias'][0] == pytest.approx(math.log(train.RATE_FLOOR), rel=1e-6) and not info['rate_net.out.bias']['fallback']
    got, info = train.fit_biases(stub_rows())
    assert got['rate_net.out.bias'][0] == 0.0 and info['rate_net.out.bias']['fallback']


def test_the_shared_outputs_pool_the_masked_and_the_open_rows():
    """M3Size: masked 0, 1, 1 and open 1, 3 -> counts 1, 3, 0, 1 of N = 5; the answer is the same if all five rows sit in the open block."""
    a, _ = train.fit_biases(stub_rows(M3Size_masked=dict(y=[0, 1, 1], match=[1, 1, 2]), M3Size_open=dict(y=[1, 3], match=[1, 2])))
    b, _ = train.fit_biases(stub_rows(M3Size_open=dict(y=[0, 1, 1, 1, 3], match=[1, 1, 2, 1, 2])))
    want = np.log((np.array([1, 3, 0, 1]) + 0.5) / (5 + 4 * 0.5))
    np.testing.assert_allclose(a['size_out.bias'], want, rtol=1e-6)
    np.testing.assert_allclose(a['size_out.bias'], b['size_out.bias'], rtol=1e-12)


def test_the_gold_bias_is_one_pooled_law_over_all_slots_and_both_views():
    y_masked, y_open = [0, 30, 30, -1], [5]
    got, _ = train.fit_biases(stub_rows(M3Gold_masked=dict(y=y_masked, s=[0, 1, 2, 3], layer=[1] * 4, team=[0] * 4, match=[1] * 4),
                                        M3Gold_open=dict(y=y_open, s=[0], layer=[1], team=[1], match=[1])))
    n = np.zeros(31)
    n[[0, 30, 5]] = [1, 2, 1]
    assert got['gold_out.bias'].shape == (31,)
    np.testing.assert_allclose(got['gold_out.bias'], np.log((n + 0.5) / (4 + 31 * 0.5)), rtol=1e-6)


def test_the_kill_bias_has_one_log_mean_for_each_size_and_team_with_a_documented_fallback_for_a_size_that_never_occurs():
    """Index 2 s + t. Size 0: rows (1, 2) and (3, 0) -> means (2, 1). Size 2: (2, 2) masked and (1, 0) open -> (1.5, 1). Size 1 never occurs: the mean over all
    four rows of the team, (7/4, 4/4). Size 3 as well."""
    masked = dict(s=[0, 0, 2], y=[[1, 2], [3, 0], [2, 2]], match=[1, 1, 2])
    opened = dict(s=[2], y=[[1, 0]], match=[1])
    got, info = train.fit_biases(stub_rows(M3Kills_masked=masked, M3Kills_open=opened))
    want = np.log([2.0, 1.0, 1.75, 1.0, 1.5, 1.0, 1.75, 1.0])
    np.testing.assert_allclose(got['kill_out.bias'], want, rtol=1e-6)
    assert info['kill_out.bias']['fallback']
    zero, info = train.fit_biases(stub_rows())
    np.testing.assert_array_equal(zero['kill_out.bias'], np.zeros(8))
    assert info['kill_out.bias']['fallback']


def test_the_objective_and_conversion_biases_use_row_means_and_conversion_seconds():
    """M3Obj (exposure 1): mean count per row over both views. M4Conv: counts over the summed windows of the rows with a window, the rows with window 0 do not count."""
    y_obj = np.zeros((3, 18), np.int64)
    y_obj[:, 0], y_obj[0, 3] = [1, 0, 2], 3
    y_masked, y_open = np.zeros((3, 18), np.int64), np.zeros((1, 18), np.int64)
    y_masked[0, 1], y_masked[2, 1], y_open[0, 1] = 1, 4, 2
    got, _ = train.fit_biases(stub_rows(M3Obj_masked=dict(y=y_obj[:2], s=[0, 0], match=[1, 1]), M3Obj_open=dict(y=y_obj[2:], s=[0], match=[1]),
                                        M4Conv_masked=dict(y=y_masked, e=[10, 0, 5], match=[1, 1, 1]), M4Conv_open=dict(y=y_open, e=[5], match=[1])))
    obj = np.log(np.maximum(y_obj.sum(0) / 3, train.RATE_FLOOR))
    np.testing.assert_allclose(got['obj_net.out.bias'], obj, rtol=1e-6)
    conv = np.log(np.maximum((y_masked.sum(0) + y_open.sum(0)) / 20.0, train.RATE_FLOOR))                  # windows 10 + 5 + 5; the 0 does not count
    np.testing.assert_allclose(got['conv_net.out.bias'], conv, rtol=1e-6)


def test_every_bias_is_finite_with_no_observation_at_all_and_the_mark_scalar_is_not_in_the_list():
    got, info = train.fit_biases(stub_rows())
    assert all(np.isfinite(v).all() for v in got.values())
    assert 'mark_net.out.bias' not in got and info['mark_net.out.bias']['unchanged']
    assert 'shared scalar' in info['mark_net.out.bias']['reason']


def test_the_biases_have_the_shapes_of_the_output_layers_of_the_heads(rows):
    got, _ = train.fit_biases(rows['FIT'])
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, kappas=1, field_channels=2), toy_static())
    params = dict(model.heads.named_parameters())
    for name, value in got.items():
        assert tuple(params[name].shape) == value.shape, name
    assert tuple(params['mark_net.out.bias'].shape) == (1,)                                                  # why the mark bias cannot be a category vector


def test_the_biases_read_fit_rows_only():
    with pytest.raises(ValueError, match='FIT'):
        train.fit_biases(stub_rows(role='STOP'))
    with pytest.raises(ValueError, match='FIT'):
        train.fit_biases(stub_rows(role='DEV'))


def test_the_start_biases_name_the_axes_that_share_an_output(rows):
    _, info = train.fit_biases(rows['FIT'])
    assert 'scalar' in info['rate_net.out.bias']['shared_axes'] and 'all 18 bins' in info['rate_net.out.bias']['shared_axes']
    assert '48' in info['gold_out.bias']['shared_axes'] and 'one law' in info['gold_out.bias']['shared_axes']
    assert 'shared by all sizes' in info['obj_net.out.bias']['shared_axes'] and '18' in info['obj_net.out.bias']['shared_axes']
    assert 'size x team' in info['kill_out.bias']['shared_axes'] and 'masked and open' in info['size_out.bias']['shared_axes']
    assert 'cancels' in info['mark_net.out.bias']['shared_axes']


def test_a_run_starts_at_the_fit_biases_and_never_reads_the_stop_targets_for_them(tmp_path, rows, stats, world):
    """With lr = 0 the weights after the run are the starting values: the output biases equal ``fit_biases`` of the FIT rows (independent oracles for the
    rate and the size gate on the toy rows), the mark scalar is still 0, and changing the STOP counts changes no bias."""
    changed = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'STOP'))
    for key in (('M2Rate', 'masked'), ('M4Conv', 'masked')):
        fields = changed.blocks[key].fields
        seen = fields['e'] > 0
        fields['y'] = fields['y'] + (seen if seen.ndim == fields['y'].ndim else seen[:, None]).astype(fields['y'].dtype)
    a = run(tmp_path / 'a', make_data(rows, stats), make_cfg(lr=0.0), epochs=1)
    b = run(tmp_path / 'b', make_data(rows, stats, stop_rows=changed), make_cfg(lr=0.0), epochs=1)
    assert a.stop_losses != b.stop_losses
    assert_same_weights(a.model, b.model)
    h = a.model.heads
    fit = rows['FIT'].blocks
    y, e = np.asarray(fit[('M2Rate', 'masked')].fields['y'], np.float64), np.asarray(fit[('M2Rate', 'masked')].fields['e'], np.float64)
    assert h.rate_net.out.bias.item() == pytest.approx(math.log(y[e > 0].sum() / e.sum()), abs=1e-5)
    size = np.concatenate([fit[('M3Size', 'masked')].fields['y'], fit[('M3Size', 'open')].fields['y']])
    want = np.log((np.bincount(size, minlength=4) + 0.5) / (len(size) + 2.0))
    np.testing.assert_allclose(h.size_out.bias.detach().numpy(), want, atol=1e-5)
    assert tuple(h.mark_net.out.bias.shape) == (1,) and h.mark_net.out.bias.item() == 0.0
    assert a.head['bias_init']['mark_net.out.bias']['unchanged']


# =====================================================================================================================================
# determinism, the settings, the best and the last state, the stopping rule
# =====================================================================================================================================
def test_two_runs_with_one_seed_are_bit_identical(tmp_path, rows, stats, base):
    """Dropout is on (0.1), so the identity needs the same random stream as well as the same data order: the returned weights, the curves and the last checkpoint
    (model, Adam moments and random streams) are the same bits."""
    first, first_folder = base
    again = run(tmp_path / 'again', make_data(rows, stats))
    assert first.epochs_run == again.epochs_run == EPOCHS
    assert_same_run(first, again)
    a, b = last_pt(first_folder, EPOCHS), last_pt(tmp_path / 'again', EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)


def test_another_seed_gives_other_weights(tmp_path, rows, stats, base):
    first, _ = base
    other = run(tmp_path / 'other', make_data(rows, stats), seed=SEED + 1)
    assert any(not torch.equal(a, b) for a, b in zip(first.model.state_dict().values(), other.model.state_dict().values()))


def test_the_run_fixes_the_threads_and_the_algorithms(tmp_path, rows, stats):
    seen = []
    run(tmp_path / 'det', make_data(rows, stats, hook=lambda *a: seen.append((torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()))), epochs=1)
    assert seen and set(seen) == {(8, True)}
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)
    train.deterministic_cpu()
    assert torch.get_num_threads() == 8 and torch.are_deterministic_algorithms_enabled()


def test_deterministic_cpu_seeds_the_three_streams_when_it_is_given_a_seed():
    """The brief: threads, deterministic algorithms and the seeds. Without a seed the streams are left alone."""
    train.deterministic_cpu(threads=3, seed=11)
    first = (torch.rand(2), np.random.rand(2), random.random())
    assert torch.get_num_threads() == 3 and torch.are_deterministic_algorithms_enabled()
    train.deterministic_cpu(threads=3, seed=11)
    second = (torch.rand(2), np.random.rand(2), random.random())
    assert torch.equal(first[0], second[0]) and np.array_equal(first[1], second[1]) and first[2] == second[2]
    state = torch.get_rng_state().clone()
    train.deterministic_cpu(threads=3)
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match='threads'):
        train.deterministic_cpu(threads=0)


def test_a_cuda_run_sets_the_cublas_workspace_and_the_seeds_before_any_model_work(tmp_path, rows, stats, monkeypatch):
    """Without CUBLAS_WORKSPACE_CONFIG the deterministic mode fails at the first cuBLAS call, so the setting is made before the device is touched. The GPU path itself is
    never run here: the model constructor is replaced by a stop."""
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'placeholder')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')

    class Stop(Exception):
        pass

    def stop(self, *a, **k):
        raise Stop

    monkeypatch.setattr(heads.EPT, '__init__', stop)
    torch.use_deterministic_algorithms(False)
    with pytest.raises(Stop):
        train.train(make_cfg(), make_data(rows, stats), tmp_path / 'gpu', device='cuda', epochs=1, check=lambda: None, seed=SEED)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and torch.are_deterministic_algorithms_enabled()


def test_a_managed_interruption_is_finalized_and_resumes_with_the_same_bits_whatever_the_epoch(tmp_path, rows, stats, base):
    """An exception ends the call as ``interrupted`` (under the caps), and a later call resumes bit-identically: before any work (call 1) and at the last epoch (call 4)."""
    full, _ = base
    for k in (1, 4):
        folder = tmp_path / f'cut{k}'
        with pytest.raises(Cut):
            run(folder, make_data(rows, stats), check=cut_on(k))
        if k > 1:
            assert call_marker(folder)['outcome'] == 'interrupted' and call_marker(folder)['status'] == 'ok'
        assert_same_run(full, run(folder, make_data(rows, stats)))


def test_a_finished_run_is_returned_without_training_and_without_adding_accounting(tmp_path, rows, stats, base):
    """A call on a finished run asks E6 once, trains nothing, writes nothing (no call marker, no new seconds) and returns the recorded accounting."""
    full, full_folder = base
    folder = tmp_path / 'finished'
    shutil.copytree(full_folder, folder)
    before, log, asked = file_hashes(folder), [], []
    again = run(folder, make_data(rows, stats, log=log), check=lambda: asked.append(1))
    assert not [e for e in log if e[:2] == ('epoch', 'fit')] and asked == [1]
    assert_same_run(full, again)
    assert file_hashes(folder) == before
    assert (again.seconds, again.peak_memory_bytes, again.peak_rss_bytes, again.resource) == (full.seconds, full.peak_memory_bytes, full.peak_rss_bytes, full.resource)


def test_the_first_check_runs_before_any_preparation(tmp_path, rows, stats, monkeypatch):
    built = []
    real = heads.EPT.__init__
    monkeypatch.setattr(heads.EPT, '__init__', lambda self, *a, **k: (built.append(1), real(self, *a, **k))[1])
    folder = tmp_path / 'first'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), check=cut_on(1))
    assert not built and not folder.exists()


def test_the_best_stop_state_is_returned_and_the_last_state_is_what_resumes(tmp_path, rows, stats, monkeypatch):
    """Scripted STOP losses 5, 3, 4, 6 with patience 10: the best epoch is 1 (0-based), ``TrainResult.model`` holds the weights after epoch 1 while the last
    checkpoint holds those after epoch 3 and its optimizer."""
    scripted(monkeypatch, [5.0, 3.0, 4.0, 6.0])
    folder = tmp_path / 'best'
    result = run(folder, make_data(rows, stats))
    assert result.best_epoch == 1 and result.epochs_run == 4 and not result.stopped_early and result.stop_losses == [5.0, 3.0, 4.0, 6.0]
    after_epoch_1, after_epoch_3 = last_state(folder, 2), last_state(folder, 4)
    state = result.model.state_dict()
    assert all(torch.equal(state[k], after_epoch_1[k]) for k in state)
    assert any(not torch.equal(state[k], after_epoch_3[k]) for k in state)


def test_best_pt_is_written_only_in_an_epoch_that_improves_and_head_json_names_the_folder_of_the_best(tmp_path, rows, stats, monkeypatch):
    """M7. Scripted STOP losses 5, 3, 4, 6: best.pt is written after epochs 0 and 1 (folders epoch_0001, epoch_0002) and not after 2 and 3; the head.json of every
    checkpoint names the folder that holds the current best with its hash, a resume of the finished run reads it from there, and nothing is written again."""
    scripted(monkeypatch, [5.0, 3.0, 4.0, 6.0])
    folder = tmp_path / 'bestfiles'
    result = run(folder, make_data(rows, stats))
    assert [(p / 'best.pt').exists() for _, p in checkpoints(folder)] == [True, True, False, False]
    assert [m['best']['folder'] for m, _ in checkpoints(folder)] == ['epoch_0001', 'epoch_0002', 'epoch_0002', 'epoch_0002']
    newest = checkpoints(folder)[-1][0]
    raw = (folder / 'epoch_0002' / 'best.pt').read_bytes()
    assert newest['best'] == {'folder': 'epoch_0002', 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)} and sorted(newest['files']) == ['last.pt']
    assert_same_weights(result.model, run(tmp_path / 'bestfiles', make_data(rows, stats)).model)


def test_early_stopping_reads_stop_only(tmp_path, rows, stats, monkeypatch):
    """``evaluate_stop`` is called once for each epoch with ``data.stop`` and nothing else, and on a scripted STOP curve the stop epoch follows (TRN-3) while the FIT
    numbers keep moving: best epoch 1 (value 3), two epochs without a strictly lower value end the run after epoch 3."""
    curve = [5.0, 3.0, 4.0, 4.5, 0.5, 0.1]
    seen = scripted(monkeypatch, curve)
    data_ = make_data(rows, stats)
    result = run(tmp_path / 'stoponly', data_, make_cfg(patience=2), epochs=6)
    assert len(seen) == result.epochs_run == 4 and all(s is data_.stop for s in seen) and data_.stop.rows.role == 'STOP'
    assert (result.best_epoch, result.stopped_early, result.stop_losses) == (1, True, curve[:4])
    assert len(set(result.fit_batch_mean_losses)) == 4                                    # the FIT numbers are not the scripted ones


def test_a_different_stop_changes_the_choice_and_the_length_but_never_the_fit_prefix(tmp_path, rows, stats, monkeypatch):
    """Two runs on the same FIT rows and seed with different (scripted) STOP curves: the first improves every epoch (run of 4 epochs, best = last), the second
    gets worse from epoch 1 (patience 2 stops after epoch 2, best = 0). The returned models differ; the last states of the epochs that both ran are
    bit-identical (the early stopping reads STOP, the gradients never do)."""
    scripted(monkeypatch, [4.0, 3.0, 2.0, 1.0])
    a = run(tmp_path / 'a', make_data(rows, stats), make_cfg(patience=2))
    scripted(monkeypatch, [1.0, 2.0, 3.0, 4.0])
    b = run(tmp_path / 'b', make_data(rows, stats), make_cfg(patience=2))
    assert (a.best_epoch, a.epochs_run, a.stopped_early) == (3, 4, False) and (b.best_epoch, b.epochs_run, b.stopped_early) == (0, 3, True)
    assert any(not torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values()))
    for epoch in (1, 2, 3):
        sa, sb = last_state(tmp_path / 'a', epoch), last_state(tmp_path / 'b', epoch)
        assert all(torch.equal(sa[k], sb[k]) for k in sa), epoch
    assert a.fit_batch_mean_losses[:3] == b.fit_batch_mean_losses and a.lrs[:len(b.lrs)] == b.lrs


def test_patience_two_stops_after_two_epochs_without_a_strictly_lower_stop_loss(tmp_path, rows, stats, monkeypatch):
    """An equal loss is not an improvement. Curve 2, 2, 2, ...: best epoch 0, the run ends after epoch 2 (two epochs without improvement)."""
    scripted(monkeypatch, [2.0] * 10)
    result = run(tmp_path / 'flat', make_data(rows, stats), make_cfg(patience=2), epochs=10)
    assert result.stopped_early and result.epochs_run == 3 and result.best_epoch == 0 and len(result.stop_losses) == len(result.fit_batch_mean_losses) == 3
    scripted(monkeypatch, [3.0, 2.0, 2.5, 1.0, 1.5, 1.2, 9.0, 9.0])
    result = run(tmp_path / 'saw', make_data(rows, stats), make_cfg(patience=2), epochs=8)
    assert result.epochs_run == 6 and result.best_epoch == 3 and result.stopped_early                      # the counter resets at epoch 3, the stop comes at 5


def test_the_default_patience_is_two():
    assert train.TrainConfig(encoder=encoder.EncoderConfig(d=16, layers=1, heads=2), lr=1e-3, weight_decay=0.0, warmup_steps=1).patience == 2
    assert train.PATIENCE == 2


def test_the_recorded_stop_loss_is_the_pooled_stop_number_of_the_returned_weights(tmp_path, rows, stats):
    data_ = make_data(rows, stats)
    result = run(tmp_path / 'one', data_, epochs=1)
    assert result.stop_losses[0] == pytest.approx(reference_stop(result.model, data_.stop, SCALES), rel=2e-5)


def test_a_non_finite_loss_stops_the_run_before_it_is_recorded(tmp_path, rows, stats, monkeypatch):
    monkeypatch.setattr(train, 'evaluate_stop', lambda *a, **k: float('nan'))
    with pytest.raises(FloatingPointError, match='STOP'):
        run(tmp_path / 'nan', make_data(rows, stats), epochs=2)


# ---- the loop, the batches and the phases ----------------------------------------------------------------------------------------------
def test_the_loop_uses_frame_averaged_encode_epoch_batches_and_selective_m2(tmp_path, rows, stats, monkeypatch):
    """Every batch is encoded once through ``FrameAveraged.encode``, batches come from ``Batches.epoch(epoch)`` for the epochs in order, ``m2`` is called with
    ``pos`` of the rows of the batch, training batches run with gradients in training mode and STOP batches without gradients in evaluation mode."""
    log = []
    real_encode, real_m2 = heads.FrameAveraged.encode, heads.FrameAveraged.m2

    def encode(self, batch):
        log.append(('encode', torch.is_grad_enabled(), self.model.training))
        return real_encode(self, batch)

    def m2(self, enc, exposure, pos=None, *args, **kwargs):
        log.append(('m2', None if pos is None else len(pos), torch.is_grad_enabled(), self.model.training))
        return real_m2(self, enc, exposure, pos, *args, **kwargs)

    monkeypatch.setattr(heads.FrameAveraged, 'encode', encode)
    monkeypatch.setattr(heads.FrameAveraged, 'm2', m2)
    data_ = make_data(rows, stats, log=log)
    result = run(tmp_path / 'phases', data_, epochs=2)
    assert [e[2] for e in log if e[:2] == ('epoch', 'fit')] == [0, 1] and [e[2] for e in log if e[:2] == ('epoch', 'stop')] == [0, 0]            # STOP is read in its fixed order (epoch 0) every time
    batch_at = [i for i, e in enumerate(log) if e[0] == 'batch'] + [len(log)]
    assert len(batch_at) - 1 == result.epochs_run * (len(data_.fit) + len(data_.stop))
    for start, end in zip(batch_at, batch_at[1:]):
        name = log[start][1]
        inside = log[start + 1:end]
        encodes, m2s = [e for e in inside if e[0] == 'encode'], [e for e in inside if e[0] == 'm2']
        assert len(encodes) == 1 and 1 <= len(m2s) <= 2 and all(e[1] is not None for e in m2s)
        for e in encodes + m2s:
            assert e[-2:] == ((True, True) if name == 'fit' else (False, False)), (name, e)


def test_e6_check_runs_every_epoch(tmp_path, rows, stats):
    """``check()`` takes no argument and is called once for each epoch before any work of that epoch (the first call comes before the preparation), also when the run
    stops early; a run that is already finished asks once."""
    log = []
    result = run(tmp_path / 'e6', make_data(rows, stats, log=log), epochs=3, check=lambda: log.append(('check',)))
    assert result.epochs_run == 3
    order = [e for e in log if e == ('check',) or e[:2] == ('epoch', 'fit')]
    assert order == [('check',), ('epoch', 'fit', 0, 0), ('check',), ('epoch', 'fit', 1, 0), ('check',), ('epoch', 'fit', 2, 0)]


def test_e6_check_runs_every_epoch_also_when_the_run_stops_early(tmp_path, rows, stats, monkeypatch):
    scripted(monkeypatch, [2.0] * 10)
    calls = []
    result = run(tmp_path / 'e6early', make_data(rows, stats), make_cfg(patience=2), epochs=10, check=lambda: calls.append(1))
    assert result.stopped_early and len(calls) == result.epochs_run == 3


def test_a_refusing_check_stops_before_any_work(tmp_path, rows, stats):
    log = []
    folder = tmp_path / 'refused'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats, log=log), check=cut_on(1))
    assert no_work(log) and not checkpoints(folder)


def test_records_time_memory_and_checkpoint_size(base):
    """The call is timed (seconds, each epoch, and its parts: the time inside ``next()`` of the FIT batches, the optimizer steps and the STOP evaluation), the memory
    is the sampled peak with its scope, and ``checkpoint_bytes`` is the on-disk size of the last.pt of the newest checkpoint and of the best.pt that it names."""
    result, folder = base
    n = result.epochs_run
    assert result.seconds > 0 and len(result.epoch_seconds) == n and all(s > 0 for s in result.epoch_seconds)
    for name in ('load_seconds', 'step_seconds', 'stop_seconds'):
        assert len(getattr(result, name)) == n and all(v > 0 for v in getattr(result, name)), name
    assert len(result.steps) == n and len(set(result.steps)) == 1 and sum(result.steps) == len(result.lrs)
    assert all(a + b + c <= e + 1e-9 for a, b, c, e in zip(result.load_seconds, result.step_seconds, result.stop_seconds, result.epoch_seconds))
    assert result.memory_metric in ('private_bytes_windows', 'rss_bytes_fallback') and 0 < result.peak_memory_bytes < 40 * GIB
    assert result.peak_upper_bytes is None or result.peak_upper_bytes >= result.peak_memory_bytes
    assert isinstance(result.peak_rss_bytes, int) and result.peak_rss_bytes > 0
    meta = checkpoints(folder)[-1][0]
    mem = meta['memory']
    assert meta['timing'] == {'load_seconds': result.load_seconds, 'step_seconds': result.step_seconds, 'steps': result.steps, 'stop_seconds': result.stop_seconds, **result.timing_components['epochs']}
    assert mem['metric'] == result.memory_metric and mem['peak_rss_bytes'] <= result.peak_rss_bytes and mem['peak_bytes'] <= result.peak_memory_bytes
    assert mem['sampled'] is True and mem['interval_s'] == train.MEMORY_INTERVAL_S and mem['samples'] >= 2 and 'children' in mem['scope']
    assert meta['caps'] == {'memory_bytes': 40 * GIB, 'wall_seconds': 24 * 3600} and meta['seconds'] > 0
    on_disk = (Path(result.checkpoint_dir) / 'last.pt').stat().st_size + (folder / meta['best']['folder'] / 'best.pt').stat().st_size
    assert result.checkpoint_bytes == meta['checkpoint_bytes'] == on_disk > 0


def test_the_load_time_is_the_time_inside_next_of_the_batches_and_the_stop_time_is_the_evaluation(tmp_path, rows, stats, monkeypatch):
    """A fake clock moves 5 s inside every ``next()`` of the FIT iterator and 7 s inside that of STOP: the load time of an epoch is the first (once for each batch,
    the step time is not charged with it), and the STOP time of the epoch holds the second."""
    clock = Clock(monkeypatch)

    def hook(name, epoch, i):
        clock.bump += 5.0 if name == 'fit' else 7.0

    data_ = make_data(rows, stats, hook=hook)
    result = run(tmp_path / 'timing', data_, epochs=2)
    n = len(data_.fit)
    assert result.load_seconds == pytest.approx([5.0 * n] * 2, abs=1.0) and all(s < 5.0 for s in result.step_seconds)
    assert result.stop_seconds == pytest.approx([7.0 * len(data_.stop)] * 2, abs=1.0)


# =====================================================================================================================================
# the checkpoints
# =====================================================================================================================================
def test_a_checkpoint_has_its_own_folder_with_last_pt_and_a_head_json_that_lists_the_hashes(base):
    result, folder = base
    assert [meta['epoch'] for meta, _ in checkpoints(folder)] == [1, 2, 3, 4]
    for meta, path in checkpoints(folder):
        assert 'last.pt' in meta['files'] and set(meta['files']) <= {'last.pt', 'best.pt'} and 'head.json' not in meta['files']
        assert {p.name for p in path.iterdir()} == set(meta['files']) | {'head.json', 'epoch_metrics.json', 'epoch_metrics.csv'}
        for name, entry in meta['files'].items():
            raw = (path / name).read_bytes()
            assert entry['sha256'] == hashlib.sha256(raw).hexdigest() and entry['bytes'] == len(raw)
    meta, path = checkpoints(folder)[-1]
    assert not list(Path(folder).rglob('*.tmp'))                                  # every temporary name was renamed
    assert meta['done'] and meta['schema'] == train.checkpoints.METRICS_SCHEMA and result.head == meta and Path(result.checkpoint_dir) == path


def test_files_are_flushed_to_disk_before_a_checkpoint_counts(tmp_path, rows, stats, monkeypatch):
    real = os.fsync
    calls = []
    monkeypatch.setattr(os, 'fsync', lambda fd: (calls.append(fd), real(fd))[1])
    run(tmp_path / 'sync', make_data(rows, stats), epochs=1)
    assert len(calls) >= 4                                                                          # last.pt, best.pt, head.json and the call markers


def test_a_failed_write_of_the_last_file_leaves_the_earlier_checkpoint_valid(tmp_path, rows, stats, base, monkeypatch):
    """Make the write of the ``head.json`` of the second checkpoint fail. The error is not swallowed, the first checkpoint is byte for byte what it was, the
    second folder has no ``head.json`` and a later call finishes with the bits of the uninterrupted run."""
    full, _ = base
    folder = tmp_path / 'fail'
    real = train._write_atomic

    def write(path, payload):
        if Path(path).name == 'head.json' and Path(path).parent.name == 'epoch_0002':
            raise OSError('disk full (test)')
        return real(path, payload)

    monkeypatch.setattr(train, '_write_atomic', write)
    with pytest.raises(OSError, match='disk full'):
        run(folder, make_data(rows, stats))
    assert [m['epoch'] for m, _ in checkpoints(folder)] == [1] and (folder / 'epoch_0002' / 'last.pt').is_file()
    first = file_hashes(folder / 'epoch_0001')
    monkeypatch.setattr(train, '_write_atomic', real)
    assert_same_run(full, run(folder, make_data(rows, stats)))
    assert file_hashes(folder / 'epoch_0001') == first and (folder / 'epoch_0002' / 'head.json').is_file()


def test_a_failed_write_of_a_state_file_leaves_no_complete_checkpoint_behind(tmp_path, rows, stats, monkeypatch):
    monkeypatch.setattr(torch, 'save', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full (test)')))
    folder = tmp_path / 'nosave'
    with pytest.raises(OSError, match='disk full'):
        run(folder, make_data(rows, stats), epochs=1)
    assert not checkpoints(folder)


@pytest.fixture(scope='module')
def cut_folder(tmp_path_factory, rows, stats):
    """A run cut by an exception at the start of epoch 2: two checkpoints and one finalized (``interrupted``) call marker."""
    folder = tmp_path_factory.mktemp('cut_records') / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), check=cut_on(3))
    return folder


@pytest.mark.parametrize('damage', ['flip_last', 'truncate_best', 'delete_last', 'bad_json', 'wrong_size'])
def test_a_damaged_checkpoint_is_refused_and_nothing_is_deleted(tmp_path, cut_folder, rows, stats, damage):
    """A ``head.json`` is there but a file does not match it: that is damage, not a torn write, so the run stops with a message instead of silently going back to an older
    checkpoint. ``best.pt`` is the file of the folder that the newest head.json names. Nothing is deleted."""
    folder = tmp_path / 'damaged'
    shutil.copytree(cut_folder, folder)
    newest = folder / 'epoch_0002'
    best = folder / json.loads((newest / 'head.json').read_text(encoding='utf-8'))['best']['folder'] / 'best.pt'
    if damage == 'flip_last':
        raw = bytearray((newest / 'last.pt').read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        (newest / 'last.pt').write_bytes(bytes(raw))
    elif damage == 'truncate_best':
        best.write_bytes(best.read_bytes()[:-8])
    elif damage == 'delete_last':
        (newest / 'last.pt').rename(newest / 'last.pt.moved')
    elif damage == 'bad_json':
        (newest / 'head.json').write_text('{"epoch": 2,', encoding='utf-8')
    else:
        meta = json.loads((newest / 'head.json').read_text(encoding='utf-8'))
        meta['files']['last.pt']['bytes'] += 1
        (newest / 'head.json').write_text(json.dumps(meta), encoding='utf-8')
    before = set(file_hashes(folder))
    with pytest.raises(RuntimeError, match='checkpoint epoch_000'):
        run(folder, make_data(rows, stats))
    assert before <= set(file_hashes(folder))


@pytest.mark.parametrize('change', ['epochs', 'seed', 'static', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'encoder', 'scales', 'stop_rows',
                                    'fit_batch_size'])
def test_a_resume_with_another_horizon_seed_terrain_setting_or_data_is_refused_before_any_work(cut_folder, rows, stats, world, change):
    """The schedule, the random stream, the model and the data are fixed by these: resuming with another value would silently be another run. The refusal writes nothing."""
    kwargs, cfg, log = {'epochs': EPOCHS, 'seed': SEED}, make_cfg(), []
    data_ = make_data(rows, stats, log=log)
    if change == 'epochs':
        kwargs['epochs'] = EPOCHS + 1
    elif change == 'seed':
        kwargs['seed'] = SEED + 1
    elif change == 'static':
        data_ = make_data(rows, stats, static=toy_static(False), log=log)
    elif change in ('lr', 'weight_decay', 'warmup_steps', 'patience'):
        cfg = make_cfg(**{change: {'lr': 1e-3, 'weight_decay': 0.02, 'warmup_steps': 3, 'patience': 5}[change]})
    elif change == 'encoder':
        cfg = make_cfg(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.2, kappas=1, field_channels=2))
    elif change == 'scales':
        data_ = make_data(rows, stats, scales={**SCALES, loss.BLOCKS[0]: 9.0}, log=log)
    elif change == 'stop_rows':
        other = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'STOP'))
        other.blocks[('M3Size', 'masked')].fields['y'] = (other.blocks[('M3Size', 'masked')].fields['y'] + 1) % 4
        data_ = make_data(rows, stats, stop_rows=other, log=log)
    else:
        data_ = make_data(rows, stats, fit_batches=2, log=log)
    before = file_hashes(cut_folder)
    with pytest.raises(ValueError, match='differ'):
        run(cut_folder, data_, cfg, **kwargs)
    assert no_work(log) and file_hashes(cut_folder) == before


def test_the_static_channels_are_recorded_with_their_provenance_and_used(tmp_path, rows, stats, base):
    """The static terrain channels are not in the ``state_dict`` (non-persistent buffers), so ``head.json`` records the function, the shape, the channel names, a
    hash of the float32 array and what the caller says about its source; a run on another terrain gives another encoder."""
    _, folder = base
    static = toy_static()
    source = {'terrain_manifest_sha256': 'ab' * 32}
    got = run(tmp_path / 'src', make_data(rows, stats, static_source=source), epochs=1)
    rec = got.head['static']
    assert rec['sha256'] == hashlib.sha256(np.ascontiguousarray(static, dtype=np.float32).tobytes()).hexdigest()
    assert rec['channels'] == list(encoder.STATIC_CHANNELS) and rec['shape'] == list(static.shape)
    assert rec['function'] == 'c_ept.ept.encoder.static_channels' and rec['source'] == source
    assert checkpoints(folder)[-1][0]['static']['source'] is None
    walls, open_grid = (run(tmp_path / name, make_data(rows, stats, static=toy_static(w)), make_cfg(lr=0.0), epochs=1) for name, w in (('walls', True), ('open', False)))
    batch = next(iter(make_data(rows, stats).stop.epoch(0)))
    views = {k: batch[k] for k in data.VIEW_KEYS}
    with torch.no_grad():
        assert not torch.equal(walls.model.eval().encode(views).h_pool, open_grid.model.eval().encode(views).h_pool)


@pytest.mark.parametrize('static', [np.zeros((75, 75, 11), np.float32), np.zeros((75, 75), np.float32), np.zeros((74, 75, 12), np.float32)], ids=['channels', 'ndim', 'grid'])
def test_the_static_channels_are_checked_for_their_shape_and_channel_count_before_they_are_recorded(tmp_path, rows, stats, static):
    """M8. The record names ``encoder.static_channels``: an array that is not that function's grid x the twelve ``STATIC_CHANNELS`` is refused before any work."""
    log = []
    with pytest.raises(ValueError, match='static'):
        run(tmp_path / 'static', make_data(rows, stats, static=static, log=log))
    assert no_work(log) and not (tmp_path / 'static').exists()


# =====================================================================================================================================
# hard interrupts: a child process ends itself with os._exit, a fresh process resumes
# =====================================================================================================================================
def _child_main(spec: dict) -> None:
    """The child of the hard-interrupt tests (a fresh interpreter). It trains on the toy world of ``spec['world']`` into ``spec['out']`` and ends itself, without any
    handler, finaliser or flush, with ``os._exit(CUT_EXIT)`` at ``spec['cut']`` of epoch ``spec['epoch']`` (default 2): ``epoch_start`` (the E6 check of that epoch),
    ``between_batches`` (after the first FIT batch of the epoch), ``after_last_pt`` (head.json of the checkpoint with that many epochs done is about to be written) or
    ``after_head`` (it has just been written). Without a cut it runs to the end and saves the result to ``spec['result']``. ``spec['beat']`` sets the heartbeat period."""
    from c_ept.data.store_core import Store
    folder = Path(spec['world'])
    s1, labels, s2 = Store(folder / 'base' / 's1'), Store(folder / 'base' / 'labels'), Store(folder / 's2')
    stats_ = data.fit_stats(s1, s2)
    rows_ = {role: data.EngagementRows(data.ept_head_rows(s1, s2, labels, role)) for role in ('FIT', 'STOP')}
    cut, at = spec.get('cut'), spec.get('epoch', 2)
    if 'beat' in spec:
        train.BEAT_INTERVAL_S = spec['beat']

    def die():
        os._exit(CUT_EXIT)

    check, hook = None, None
    if cut == 'epoch_start':
        check = cut_on(at + 1, die)
    elif cut == 'between_batches':
        hook = lambda name, epoch, i: die() if (name, epoch, i) == ('fit', at, 1) else None             # noqa: E731
    elif cut in ('after_last_pt', 'after_head'):
        real = train._write_atomic

        def write(path, payload):
            hit = Path(path).name == 'head.json' and Path(path).parent.name == f'epoch_{at:04d}'
            if hit and cut == 'after_last_pt':
                die()
            real(path, payload)
            if hit and cut == 'after_head':
                die()

        train._write_atomic = write
    result = run(spec['out'], make_data(rows_, stats_, hook=hook), check=check)
    if spec.get('result'):
        torch.save({'state': result.model.state_dict(), 'stop_losses': result.stop_losses, 'lrs': result.lrs}, spec['result'])


def run_child(spec: dict, folder, name: str):
    """Run ``_child_main`` in a fresh interpreter (the spec goes through a JSON file, so there is no quoting) and return the completed process."""
    path = Path(folder) / f'{name}.spec.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    return subprocess.run([sys.executable, '-B', '-m', 'c_ept.tests.test_ept_train', str(path)], cwd=REPO, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'},
                          capture_output=True, text=True, timeout=300)


CUTS = {'epoch_start': {'cut': 'epoch_start'}, 'between_batches': {'cut': 'between_batches'}, 'after_last_pt': {'cut': 'after_last_pt'},
        'after_head': {'cut': 'after_head'}, 'after_the_last_head': {'cut': 'after_head', 'epoch': EPOCHS}}


@pytest.fixture(scope='module')
def hard_cuts(tmp_path_factory, world):
    """For each cut point, a child process that is cut there (heartbeat period 0.01 s), a copy of its folder as the cut left it (``<name>_cut``) and a second, fresh
    process that resumes the original folder to the end and saves its result (``<name>.result.pt``). The five cuts run at the same time."""
    root, base_spec = tmp_path_factory.mktemp('hard_cuts'), {'world': str(world.s2.root.parent)}

    def cut_then_resume(name):
        out = root / name
        cut = run_child({**base_spec, 'out': str(out), 'beat': 0.01, **CUTS[name]}, root, f'{name}.cut')
        assert cut.returncode == CUT_EXIT, f'{name}: the child did not reach its cut point: {cut.stderr[-1500:]}'
        shutil.copytree(out, root / f'{name}_cut')
        done = run_child({**base_spec, 'out': str(out), 'result': str(root / f'{name}.result.pt')}, root, f'{name}.resume')
        assert done.returncode == 0, f'{name}: the fresh process did not resume: {done.stderr[-1500:]}'

    with ThreadPoolExecutor(len(CUTS)) as pool:
        list(pool.map(cut_then_resume, CUTS))
    return root


@pytest.mark.parametrize('name', list(CUTS))
def test_cut_and_resume_gives_bit_identical_weights(hard_cuts, base, name):
    """A real hard interrupt (``os._exit`` of a child at an epoch start, between FIT batches, after last.pt and before head.json, after head.json, and after the head.json
    of the last epoch) and a fresh process that resumes: its weights, stop_losses and lrs are those of the uninterrupted run, and so is the last state (model, Adam
    moments, random streams) of every epoch, also of the epoch whose torn folder was written again."""
    full, full_folder = base
    out = hard_cuts / name
    saved = torch.load(hard_cuts / f'{name}.result.pt', weights_only=True)
    assert_same_state(saved['state'], full.model.state_dict())
    assert saved['stop_losses'] == full.stop_losses and saved['lrs'] == full.lrs
    for epoch in range(1, EPOCHS + 1):
        a, b = last_pt(full_folder, epoch), last_pt(out, epoch)
        for part in ('model', 'optimizer', 'rng'):
            assert_same_state(a[part], b[part], f'epoch {epoch} {part}')


def test_a_hard_interrupt_leaves_an_orphan_call_that_the_next_process_adopts_and_charges_from_its_last_heartbeat(hard_cuts):
    """The cut left a start without an end and heartbeat lines. The next process wrote the missing end as a new file: ``hard_interrupted``, charged the seconds of the
    pending start + the last heartbeat + one period, and the counters of the next call start there."""
    cut, done = hard_cuts / 'epoch_start_cut' / 'calls', hard_cuts / 'epoch_start'
    assert (cut / 'call_0001.pending.json').is_file() and not (cut / 'call_0001.json').exists()
    pending = json.loads((cut / 'call_0001.pending.json').read_text(encoding='utf-8'))
    beats = beat_lines(cut / 'call_0001.beat')
    assert beats and [b['seconds'] for b in beats] == sorted(b['seconds'] for b in beats) and all(set(b) == {'seconds', 'peak_bytes'} for b in beats)
    first, second = call_marker(done, 1), call_marker(done, 2)
    start = pending['start_accounting']
    assert (first['outcome'], first['status'], second['outcome']) == ('hard_interrupted', 'ok', 'completed') and first['evidence']['basis'] == 'beat'
    assert first['seconds'] == pytest.approx(start['seconds'] + beats[-1]['seconds'] + train.BEAT_INTERVAL_S, rel=1e-12)
    assert first['peak_bytes'] >= max(beats[-1]['peak_bytes'], start['peak_bytes']) > 0
    assert (first['epoch_done_start'], first['epoch_done_end'], second['epoch_done_start']) == (0, 2, 2)
    assert second['seconds'] >= first['seconds'] and second['peak_bytes'] >= first['peak_bytes']


def test_a_hard_cut_after_the_last_head_is_finalized_by_the_next_process_which_trains_nothing(hard_cuts):
    done = hard_cuts / 'after_the_last_head'
    first = call_marker(done, 1)
    assert first['outcome'] == 'hard_interrupted' and first['epoch_done_end'] == EPOCHS
    assert not (done / 'calls' / 'call_0002.pending.json').exists()                          # the finished run opens no new call


def test_a_hard_cut_without_a_heartbeat_is_charged_one_period_from_the_pending_start(tmp_path, hard_cuts, rows, stats, base):
    folder = tmp_path / 'nobeat'
    shutil.copytree(hard_cuts / 'epoch_start_cut', folder, ignore=shutil.ignore_patterns('*.beat'))
    assert_same_run(base[0], run(folder, make_data(rows, stats)))
    start = json.loads((folder / 'calls' / 'call_0001.pending.json').read_text(encoding='utf-8'))['start_accounting']
    first = call_marker(folder, 1)
    assert first['evidence']['basis'] == 'pending_start' and first['seconds'] == pytest.approx(start['seconds'] + train.BEAT_INTERVAL_S, rel=1e-12)
    assert first['peak_bytes'] >= start['peak_bytes'] > 0


def test_the_last_complete_heartbeat_is_read_and_a_torn_tail_is_ignored(tmp_path):
    path = tmp_path / 'call_0001.beat'
    assert train._read_beats(path) == (0, None)                                               # no file: no heartbeat
    path.write_bytes(b'{"seconds": 1.5, "peak_bytes": 10}\n{"seconds": 61.5, "peak_bytes": 20}\n{"seconds": 12')                  # a cut in the middle of the third line
    assert train._read_beats(path) == (2, {'seconds': 61.5, 'peak_bytes': 20})
    path.write_bytes(b'{"seconds": 1.5, "peak_bytes": 10}\n\x00\x00\x00\x00')                    # zeros where the size grew and the data did not
    assert train._read_beats(path) == (1, {'seconds': 1.5, 'peak_bytes': 10})


@pytest.mark.parametrize('content', [b'{"seconds": 1.0, "peak_bytes": 1}\nnot json\n{"seconds": 2.0, "peak_bytes": 1}\n', b'{"seconds": -1.0, "peak_bytes": 1}\n',
                                     b'{"seconds": 2.0, "peak_bytes": 1}\n{"seconds": 1.0, "peak_bytes": 1}\n', b'{"seconds": true, "peak_bytes": 1}\n',
                                     b'{"seconds": 1.0, "peak_bytes": 1.5}\n', b'{"seconds": 1.0}\n'])
def test_a_complete_heartbeat_line_that_is_not_valid_is_refused(tmp_path, content):
    path = tmp_path / 'call_0001.beat'
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match='heartbeat'):
        train._read_beats(path)


def test_the_heartbeat_appends_and_fsyncs_one_line_per_period(tmp_path, rows, stats, monkeypatch):
    monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 1e6)                                      # only the explicit samples
    monkeypatch.setattr(train, 'BEAT_INTERVAL_S', 0.0)
    synced, real = [], os.fsync
    monkeypatch.setattr(os, 'fsync', lambda fd: (synced.append(fd), real(fd))[1])
    run(tmp_path / 'beats', make_data(rows, stats), epochs=1)
    beats = beat_lines(tmp_path / 'beats' / 'calls' / 'call_0001.beat')
    assert len(beats) >= 2 and all(b['peak_bytes'] > 0 for b in beats) and [b['seconds'] for b in beats] == sorted(b['seconds'] for b in beats)
    assert len(synced) >= len(beats) + 4
    monkeypatch.setattr(train, 'BEAT_INTERVAL_S', 60.0)
    run(tmp_path / 'quiet', make_data(rows, stats), epochs=1)
    assert not (tmp_path / 'quiet' / 'calls' / 'call_0001.beat').exists()                       # a short call writes no heartbeat at the default period


def test_a_charge_over_the_cap_is_a_stored_failure_and_refuses_every_later_call(tmp_path, hard_cuts, rows, stats):
    folder = tmp_path / 'over'
    shutil.copytree(hard_cuts / 'epoch_start_cut', folder)
    with open(folder / 'calls' / 'call_0001.beat', 'ab') as handle:
        handle.write(b'{"seconds": 90000.0, "peak_bytes": 1000000}\n')                            # a later heartbeat of the dead call: it was already over 24 h
    log = []
    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(folder, make_data(rows, stats, log=log))
    first = call_marker(folder, 1)
    assert no_work(log) and list(folder.glob('resource_failure_*.json')) and (first['outcome'], first['status']) == ('hard_interrupted', 'exceeded')
    again = []
    with pytest.raises(train.ResourceBoundError, match='earlier'):
        run(folder, make_data(rows, stats, log=again))
    assert no_work(again)


def test_a_charge_of_exactly_24_hours_is_adopted_and_the_first_moment_that_the_next_call_adds_is_refused(tmp_path, hard_cuts, rows, stats):
    """The time bound is inclusive: 0 + 86,340 + 60 = 24 h exactly is adopted as ``ok``; the new call then starts at 24 h and is over the bound as soon as its own clock moves."""
    folder = tmp_path / 'edge'
    shutil.copytree(hard_cuts / 'epoch_start_cut', folder)
    pending_path = folder / 'calls' / 'call_0001.pending.json'
    pending = json.loads(pending_path.read_text(encoding='utf-8'))
    pending['start_accounting']['seconds'] = 0.0
    pending_path.write_text(json.dumps(pending), encoding='utf-8')
    (folder / 'calls' / 'call_0001.beat').write_text('{"seconds": 86340.0, "peak_bytes": 1000000}\n', encoding='utf-8')
    log = []
    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(folder, make_data(rows, stats, log=log))
    first = call_marker(folder, 1)
    assert first['seconds'] == 86_400.0 and first['status'] == 'ok' and no_work(log)
    assert json.loads(next(folder.glob('resource_failure_*.json')).read_text(encoding='utf-8'))['seconds'] > 86_400.0


def test_the_bounds_are_a_strict_memory_cap_and_an_inclusive_time_cap():
    ok = 'private_bytes_windows'
    assert train._violation(40 * GIB - 1, ok, False, 0.0) is None and '40 GiB' in train._violation(40 * GIB, ok, False, 0.0)
    assert train._violation(GIB, ok, False, 24 * 3600.0) is None and '24 h' in train._violation(GIB, ok, False, 24 * 3600.0 + 1e-6)
    assert 'measure' in train._violation(GIB, 'unavailable', False, 0.0) and 'measure' in train._violation(GIB, ok, True, 0.0)


@pytest.fixture(scope='module')
def crash_loop(tmp_path_factory, world):
    """A child that is cut in epoch 0 (no epoch done); then, on two copies of its folder, a second child that is cut at the same point and one that is cut at the start of
    epoch 1 (after an epoch was done). Each second child resumed the first one's orphan, so a single hard interrupt is not refused."""
    root, spec = tmp_path_factory.mktemp('crash_loop'), {'world': str(world.s2.root.parent)}
    first = run_child({**spec, 'out': str(root / 'a'), 'cut': 'between_batches', 'epoch': 0}, root, 'a')
    assert first.returncode == CUT_EXIT, first.stderr[-1500:]

    def second(name, extra):
        shutil.copytree(root / 'a', root / name)
        got = run_child({**spec, 'out': str(root / name), **extra}, root, name)
        assert got.returncode == CUT_EXIT, f'{name}: {got.stderr[-1500:]}'

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda kv: second(*kv), [('same_point', {'cut': 'between_batches', 'epoch': 0}), ('after_an_epoch', {'cut': 'epoch_start', 'epoch': 1})]))
    return root


def test_two_consecutive_hard_interrupts_without_a_new_epoch_are_refused_for_good(crash_loop, tmp_path, rows, stats):
    folder = tmp_path / 'loop'
    shutil.copytree(crash_loop / 'same_point', folder)
    for _ in range(2):                                              # the refusal does not use itself up: the next call finds the same two interrupts
        log = []
        with pytest.raises(train.ResourceBoundError, match='consecutive hard interrupts'):
            run(folder, make_data(rows, stats, log=log))
        assert no_work(log) and not checkpoints(folder)


def test_a_second_hard_interrupt_after_a_new_epoch_is_resumed(crash_loop, tmp_path, rows, stats, base):
    folder = tmp_path / 'progress'
    shutil.copytree(crash_loop / 'after_an_epoch', folder)
    assert_same_run(base[0], run(folder, make_data(rows, stats)))
    markers = [call_marker(folder, n) for n in (1, 2, 3)]
    assert [m['outcome'] for m in markers] == ['hard_interrupted', 'hard_interrupted', 'completed']
    assert [m['epoch_done_end'] for m in markers[:2]] == [0, 1]                                # the epoch that call 2 finished separates the two interrupts


def test_a_stored_failure_record_refuses_every_later_call_before_any_work(tmp_path, rows, stats):
    folder = tmp_path / 'failed'
    folder.mkdir()
    (folder / 'resource_failure_0000.json').write_text('{}', encoding='utf-8')
    log = []
    with pytest.raises(train.ResourceBoundError, match='earlier'):
        run(folder, make_data(rows, stats, log=log))
    assert no_work(log) and not (folder / 'calls').exists()


@pytest.mark.parametrize('damage', ['other_run', 'gap', 'keys', 'exceeded'])
def test_call_markers_of_another_run_with_a_gap_or_wrong_keys_are_refused_and_an_exceeded_one_ends_the_run(tmp_path, hard_cuts, rows, stats, damage):
    folder = tmp_path / damage
    shutil.copytree(hard_cuts / ('epoch_start_cut' if damage in ('other_run', 'gap') else 'epoch_start'), folder)
    calls = folder / 'calls'
    if damage == 'other_run':
        doc = json.loads((calls / 'call_0001.pending.json').read_text(encoding='utf-8'))
        doc['identity_sha256'] = '0' * 64
        (calls / 'call_0001.pending.json').write_text(json.dumps(doc), encoding='utf-8')
        error, match = RuntimeError, 'another run'
    elif damage == 'gap':
        (calls / 'call_0001.pending.json').rename(calls / 'call_0003.pending.json')
        error, match = RuntimeError, 'numbered'
    else:
        rec = json.loads((calls / 'call_0001.json').read_text(encoding='utf-8'))
        if damage == 'keys':
            del rec['metric']
            error, match = RuntimeError, 'final marker'
        else:
            rec.update(outcome='exceeded', status='exceeded', reason='forced for the test')
            error, match = train.ResourceBoundError, 'earlier'
        (calls / 'call_0001.json').write_text(json.dumps(rec), encoding='utf-8')
    log = []
    with pytest.raises(error, match=match):
        run(folder, make_data(rows, stats, log=log))
    assert no_work(log)


# =====================================================================================================================================
# the resources: the sampled memory, the clock, the guard
# =====================================================================================================================================
def test_the_memory_sample_is_private_bytes_on_windows_and_an_explicit_rss_fallback_elsewhere():
    assert train._metric_name(SimpleNamespace(private=3 * GIB, rss=5 * GIB)) == 'private_bytes_windows'
    assert train._metric_name(SimpleNamespace(rss=5 * GIB)) == 'rss_bytes_fallback'
    used, upper, rss, name = train._memory_sample()
    assert used > 0 and rss > 0 and name == ('private_bytes_windows' if os.name == 'nt' else 'rss_bytes_fallback')
    assert upper is None or upper >= used                                                      # the Windows lifetime upper bound is a separate number


def test_the_memory_sample_is_the_tree_of_this_process_and_its_children_through_the_public_runtime_helper(monkeypatch):
    """``_memory_sample`` calls ``runtime.tree_memory`` on this process and its children: the private bytes and the observed lifetime upper bound (``peak_pagefile``)
    of both are summed, the RSS of both is summed separately."""
    me = SimpleNamespace(memory_info=lambda: SimpleNamespace(private=3, rss=4, peak_pagefile=10))
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(private=5, rss=6, peak_pagefile=20))
    fake = SimpleNamespace(children=lambda recursive=True: [child], memory_info=me.memory_info)
    monkeypatch.setattr(train, '_PROCESS', fake)
    assert train._memory_sample() == (8, 30, 10, 'private_bytes_windows')


def injected_memory(monkeypatch, level):
    """No real resource is used: the memory samples are those of ``level()`` and only explicit samples are taken (the sampling thread never fires)."""
    monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 1e6)
    monkeypatch.setattr(train, '_memory_sample', lambda: (level(), None, level() // 2, 'private_bytes_windows'))


class Clock:
    """The real clock plus an offset that a test moves: no time is actually spent."""

    def __init__(self, monkeypatch, extra=None):
        self.bump, self.extra = 0.0, extra
        monkeypatch.setattr(train, '_clock', lambda: time.perf_counter() + self.bump + (self.extra() if self.extra else 0.0))


def test_a_sample_over_40_gib_stops_the_run_before_its_first_epoch(tmp_path, rows, stats, monkeypatch):
    injected_memory(monkeypatch, lambda: 41 * GIB)
    log = []
    folder = tmp_path / 'big'
    with pytest.raises(train.ResourceBoundError, match='40 GiB'):
        run(folder, make_data(rows, stats, log=log))
    assert no_work(log) and not checkpoints(folder)


def test_a_sample_at_the_cap_is_refused_and_one_byte_below_is_allowed(tmp_path, rows, stats, monkeypatch):
    """The bound is a strict less-than: 40 GiB itself is refused."""
    injected_memory(monkeypatch, lambda: 40 * GIB)
    with pytest.raises(train.ResourceBoundError, match='40 GiB'):
        run(tmp_path / 'edge', make_data(rows, stats), epochs=1)
    injected_memory(monkeypatch, lambda: 40 * GIB - 1)
    assert run(tmp_path / 'below', make_data(rows, stats), epochs=1).peak_memory_bytes == 40 * GIB - 1


def test_a_missing_measurement_is_not_a_measured_zero(tmp_path, rows, stats, monkeypatch):
    """A sample of 0 bytes (no psutil, nothing readable) must not pass as a tiny peak: the run refuses before any work, with a message that says it could not measure."""
    monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 1e6)
    monkeypatch.setattr(train, '_memory_sample', lambda: (0, None, 0, 'unavailable'))
    log = []
    with pytest.raises(train.ResourceBoundError, match='measure'):
        run(tmp_path / 'blind', make_data(rows, stats, log=log))
    assert no_work(log)


def test_the_upper_bound_and_the_sampled_peak_are_recorded_separately(tmp_path, rows, stats, monkeypatch):
    monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 1e6)
    monkeypatch.setattr(train, '_memory_sample', lambda: (GIB, 7 * GIB, 2 * GIB, 'private_bytes_windows'))
    result = run(tmp_path / 'upper', make_data(rows, stats), epochs=1)
    assert (result.peak_memory_bytes, result.peak_upper_bytes, result.peak_rss_bytes) == (GIB, 7 * GIB, 2 * GIB)
    mem = result.head['memory']
    assert mem['peak_bytes'] == GIB and mem['peak_upper_bytes'] == 7 * GIB and 'child' in mem['scope']


def test_the_guard_reads_the_sampler_and_the_clock_and_takes_no_sample_of_its_own(monkeypatch):
    """M3. ``guard()`` between batches costs no process-tree sample; the explicit sample of an epoch boundary or a checkpoint write is ``boundary()``."""
    calls = []
    monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 1e6)
    monkeypatch.setattr(train, '_memory_sample', lambda: (calls.append(1), (GIB, None, GIB, 'private_bytes_windows'))[1])
    res = train._Resources()
    with res.sampler:
        n = len(calls)
        for _ in range(5):
            res.guard()
        assert len(calls) == n
        res.boundary()
        assert len(calls) == n + 1
        res.sampler.peak = 41 * GIB                                                              # what the sampling thread would have seen
        with pytest.raises(train.ResourceBoundError, match='40 GiB'):
            res.guard()


@pytest.mark.parametrize('where', ['fit', 'stop'])
def test_the_sampling_thread_and_the_guard_stop_the_run_between_batches(tmp_path, rows, stats, monkeypatch, where):
    """High memory appears when the second batch of FIT (or of STOP) is fetched. The sampling thread sees it, the guard reads it before that batch is processed, nothing
    is checkpointed, and a later call is refused."""
    state = {'high': False, 'seen': 0}

    def sample():
        state['seen'] += state['high']
        return (41 * GIB if state['high'] else GIB), None, GIB, 'private_bytes_windows'

    def hook(name, epoch, i):
        if (name, i) == (where, 1) and not state['high']:
            state['high'] = True
            deadline = time.time() + 10
            while state['seen'] < 2 and time.time() < deadline:                                  # the first high sample is then part of the peak
                time.sleep(0.001)

    monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 0.001)
    monkeypatch.setattr(train, '_memory_sample', sample)
    log, folder = [], tmp_path / where
    data_ = make_data(rows, stats, log=log, stop_size=2, hook=hook)
    with pytest.raises(train.ResourceBoundError, match='40 GiB'):
        run(folder, data_)
    assert (fit_batches_logged(log), stop_batches_logged(log)) == ((2, 0) if where == 'fit' else (len(data_.fit), 2)) and not checkpoints(folder)
    state['high'] = False
    again = []
    with pytest.raises(train.ResourceBoundError, match='earlier'):
        run(folder, make_data(rows, stats, log=again))
    assert no_work(again)


def test_the_time_is_guarded_between_batches_too(tmp_path, rows, stats, monkeypatch):
    clock, log = Clock(monkeypatch), []

    def hook(name, epoch, i):
        if (name, i) == ('fit', 1):
            clock.bump += 90_000.0

    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(tmp_path / 'midtime', make_data(rows, stats, log=log, hook=hook))
    assert fit_batches_logged(log) == 2 and stop_batches_logged(log) == 0


@pytest.mark.parametrize('epochs', [4, 2])
def test_an_overrun_during_the_checkpoint_write_is_kept_and_refuses_every_later_call(tmp_path, rows, stats, monkeypatch, epochs):
    """The memory is over the cap only once the second ``head.json`` is on disk, that is, caused by the write of epoch 2 (an intermediate epoch for 4 epochs, the last
    one for 2). The explicit sample after the write finds it: the checkpoint is kept, the failure is stored, and a later call raises again before any work, with a low
    memory, also when the run was finished."""
    folder = tmp_path / f'late{epochs}'
    injected_memory(monkeypatch, lambda: 41 * GIB if (folder / 'epoch_0002' / 'head.json').exists() else GIB)
    log = []
    with pytest.raises(train.ResourceBoundError, match='40 GiB'):
        run(folder, make_data(rows, stats, log=log), epochs=epochs)
    assert [m['epoch'] for m, _ in checkpoints(folder)] == [1, 2] and {e[2] for e in log if e[:2] == ('epoch', 'fit')} == {0, 1}
    assert json.loads(next(folder.glob('resource_failure_*.json')).read_text(encoding='utf-8'))['peak_bytes'] >= 41 * GIB
    assert checkpoints(folder)[-1][0]['memory']['peak_bytes'] < 41 * GIB                       # the sample that broke the bound is after the write of that head.json
    injected_memory(monkeypatch, lambda: GIB)
    again = []
    with pytest.raises(train.ResourceBoundError, match='earlier'):
        run(folder, make_data(rows, stats, log=again), epochs=epochs)
    assert no_work(again)


def test_a_time_overrun_during_the_checkpoint_write_counts_the_write_and_is_kept(tmp_path, rows, stats, monkeypatch):
    """The write of the second ``head.json`` takes (fake) 90,000 s: the checkpoint is complete, the stored failure holds the seconds that include the write, and a later
    call refuses before work although the clock is normal again."""
    folder, real = tmp_path / 'slow', train._write_atomic
    clock = Clock(monkeypatch)

    def write(path, payload):
        real(path, payload)
        if Path(path).name == 'head.json' and Path(path).parent.name == 'epoch_0002':
            clock.bump += 90_000.0

    monkeypatch.setattr(train, '_write_atomic', write)
    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(folder, make_data(rows, stats))
    assert [m['epoch'] for m, _ in checkpoints(folder)] == [1, 2]
    assert json.loads(next(folder.glob('resource_failure_*.json')).read_text(encoding='utf-8'))['seconds'] >= 90_000.0
    monkeypatch.setattr(train, '_write_atomic', real)
    clock.bump = 0.0
    log = []
    with pytest.raises(train.ResourceBoundError, match='earlier'):
        run(folder, make_data(rows, stats, log=log))
    assert no_work(log)


def test_the_preparation_and_the_restore_are_inside_the_measured_scope(tmp_path, rows, stats, cut_folder, monkeypatch):
    """The memory goes over the cap during the restore of a checkpoint (``torch.load``): the resume stops before any epoch. A call on a finished run is measured
    too."""
    done = tmp_path / 'done'
    run(done, make_data(rows, stats))
    state = {'high': False}
    injected_memory(monkeypatch, lambda: 41 * GIB if state['high'] else GIB)
    real = torch.load

    def load(*a, **k):
        state['high'] = True
        return real(*a, **k)

    monkeypatch.setattr(torch, 'load', load)
    for source in (cut_folder, done):
        folder = tmp_path / f'copy_{source.name}'
        shutil.copytree(source, folder)
        state['high'] = False
        log = []
        with pytest.raises(train.ResourceBoundError, match='40 GiB'):
            run(folder, make_data(rows, stats, log=log))
        assert no_work(log)


def test_loading_the_best_state_for_the_return_is_inside_the_measured_scope(tmp_path, rows, stats, monkeypatch):
    folder = tmp_path / 'ret'
    state = {'high': False}
    injected_memory(monkeypatch, lambda: 41 * GIB if state['high'] else GIB)
    real = torch.nn.Module.load_state_dict

    def load(self, *a, **k):
        out = real(self, *a, **k)
        if (folder / 'epoch_0002' / 'head.json').exists():
            state['high'] = True
        return out

    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', load)
    with pytest.raises(train.ResourceBoundError, match='40 GiB'):
        run(folder, make_data(rows, stats), epochs=2)


def test_one_model_is_allocated_for_a_run_and_for_a_finished_resume(tmp_path, rows, stats, monkeypatch):
    built = []
    real = heads.EPT.__init__
    monkeypatch.setattr(heads.EPT, '__init__', lambda self, *a, **k: (built.append(1), real(self, *a, **k))[1])
    folder = tmp_path / 'one'
    run(folder, make_data(rows, stats), epochs=2)
    assert len(built) == 1
    run(folder, make_data(rows, stats), epochs=2)
    assert len(built) == 2


# =====================================================================================================================================
# the identity of the inputs, the precision, the interface and the roles
# =====================================================================================================================================
def test_the_store_identity_is_recorded_and_bound(base):
    _, folder = base
    meta = checkpoints(folder)[-1][0]
    assert len(meta['identity']['s2_store_json_sha256']) == 64
    for key, want in (('device', 'cpu'), ('dtype', 'torch.float32'), ('deterministic_algorithms', True), ('threads', 8)):
        assert meta['identity'][key] == want, key
    assert meta['identity']['torch'] == torch.__version__


def test_the_store_identity_is_the_sha256_of_store_json_and_a_changed_store_is_refused(tmp_path):
    """R-S4a-18 (modified): the S2 store is bound by the bytes of its STORE.json (which holds the file hashes), nothing is read from the payload; a changed manifest is
    another store (resume refused before any work)."""
    w = build_ept_world(tmp_path / 'world', n_events=7, n_edges=5)
    st = data.fit_stats(w.s1, w.s2)
    r = {role: data.EngagementRows(data.ept_head_rows(w.s1, w.s2, w.labels, role)) for role in ('FIT', 'STOP')}
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(r, st), check=cut_on(3))
    assert checkpoints(folder)[-1][0]['identity']['s2_store_json_sha256'] == hashlib.sha256((w.s2.root / 'STORE.json').read_bytes()).hexdigest()
    w.s2.set_stage('probe', note=1)
    log = []
    with pytest.raises(ValueError, match='differ'):
        run(folder, make_data(r, st, log=log))
    assert no_work(log)


def test_a_run_reads_no_view_that_only_dev_selects_and_never_verifies_the_store(tmp_path, rows, stats, monkeypatch):
    from c_ept.data import store_core
    for name in ('verify', 'hashes', 'stored_files'):
        monkeypatch.setattr(store_core.Store, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('whole-store access')))
    seen, real = [], data.ViewTensors

    def spy(s2, views, st):
        seen.append(np.asarray(views).copy())
        return real(s2, views, st)

    monkeypatch.setattr(data, 'ViewTensors', spy)
    run(tmp_path / 'access', make_data(rows, stats), epochs=1)
    selected = np.unique(np.concatenate([b.views for r in ('FIT', 'STOP') for b in rows[r].blocks.values()]))
    dev_only = np.setdiff1d(np.unique(np.concatenate([b.views for b in rows['DEV'].blocks.values()])), selected)
    requested = np.unique(np.concatenate(seen))
    assert dev_only.size and np.isin(requested, selected).all() and not np.intersect1d(requested, dev_only).size


def test_fit_and_stop_must_use_the_same_fit_statistics(tmp_path, rows, stats):
    """A valid ``Stats`` of the same shape but other means on the STOP side would make the two sides read different inputs: refused before any update."""
    other = dataclasses.replace(stats, arrays={k: data.ColumnStats(cs.centre + 1.0, cs.scale, cs.count) for k, cs in stats.arrays.items()})
    assert other.sha256 != stats.sha256
    d = make_data(rows, stats)
    stop = SpyBatches(rows['STOP'], 100, 7, stats=other, training=False, name='stop', log=[])
    folder = tmp_path / 'stats'
    with pytest.raises(ValueError, match='statistics'):
        run(folder, train.TrainData(fit=d.fit, stop=stop, scales=d.scales, static=d.static))
    assert not checkpoints(folder)


def test_a_resume_on_another_device_or_dtype_is_refused_before_the_model_is_built(cut_folder, rows, stats, monkeypatch):
    """The checkpoint is a CPU float32 deterministic run: nothing says that another device or dtype gives the same bits, so it is another run."""
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'placeholder')                                # the cuda branch sets it: give it back afterwards
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    built = []
    real = heads.EPT.__init__
    monkeypatch.setattr(heads.EPT, '__init__', lambda self, *a, **k: (built.append(1), real(self, *a, **k))[1])
    with pytest.raises(ValueError, match='device'):
        train.train(make_cfg(), make_data(rows, stats), cut_folder, device='cuda', epochs=EPOCHS, check=lambda: None, seed=SEED)
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(ValueError, match='float32'):
            run(cut_folder, make_data(rows, stats))
    finally:
        torch.set_default_dtype(torch.float32)
    assert not built


def test_a_new_run_under_a_float64_default_is_refused_before_any_model_work(tmp_path, rows, stats, monkeypatch):
    folder = tmp_path / 'f64'
    log, built = [], []
    real = heads.EPT.__init__
    monkeypatch.setattr(heads.EPT, '__init__', lambda self, *a, **k: (built.append(1), real(self, *a, **k))[1])
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(ValueError, match='float32'):
            run(folder, make_data(rows, stats, log=log))
    finally:
        torch.set_default_dtype(torch.float32)
    assert no_work(log) and not built and not folder.exists()


def test_the_parameters_are_float32_and_the_precondition_is_recorded(base):
    result, folder = base
    assert all(p.dtype == torch.float32 for p in result.model.parameters())
    note = checkpoints(folder)[-1][0]['precondition']
    assert 'same implementation' in note and 'Task 7' in note and 'float32' in note


def test_the_fit_diagnostic_is_named_for_what_it_is(base):
    """``fit_batch_mean_losses`` is the mean of the batch losses of an epoch (training mode, dropout on), not a loss over the whole FIT data and not an evaluation;
    the STOP number is the evaluation of R-S4a-16 and is not comparable with it."""
    result, folder = base
    assert not hasattr(result, 'fit_losses') and len(result.fit_batch_mean_losses) == result.epochs_run
    note = checkpoints(folder)[-1][0]['losses']
    assert 'mean of batch' in note['fit_batch_mean_losses'] and 'R-S4a-16' in note['stop_losses'] and 'not comparable' in note['stop_losses']


def test_the_interface_is_the_one_of_the_brief():
    import inspect
    sig = inspect.signature(train.train)
    assert list(sig.parameters)[:3] == ['cfg', 'data', 'out_dir']
    assert all(sig.parameters[n].kind is inspect.Parameter.KEYWORD_ONLY for n in ('device', 'epochs', 'check', 'seed'))
    assert inspect.signature(train.deterministic_cpu).parameters['threads'].default == 8 and 'seed' in inspect.signature(train.deterministic_cpu).parameters
    assert {f.name for f in dataclasses.fields(train.TrainData)} == {'fit', 'stop', 'scales', 'static', 'static_source'}


def stub(role, training):
    """A stand-in for a ``Batches`` of a role: it has the two attributes that the role check reads and nothing else, so reading a batch fails."""
    return SimpleNamespace(rows=SimpleNamespace(role=role), training=training)


@pytest.mark.parametrize('role', ['DEV', 'CAL', 'AUDIT', 'TEST', 'EXTERNAL'])
@pytest.mark.parametrize('slot', ['fit', 'stop'])
def test_a_role_other_than_fit_and_stop_is_refused_before_any_work(tmp_path, rows, stats, role, slot):
    d = make_data(rows, stats)
    d = train.TrainData(fit=stub(role, True) if slot == 'fit' else d.fit, stop=stub(role, False) if slot == 'stop' else d.stop, scales=d.scales, static=d.static)
    folder = tmp_path / 'role'
    with pytest.raises((ValueError, TypeError), match='ruling R14'):
        run(folder, d)
    assert not checkpoints(folder)


def test_real_dev_batches_and_misplaced_slots_are_refused(tmp_path, rows, stats):
    d = make_data(rows, stats)
    dev = SpyBatches(rows['DEV'], 100, 7, stats=stats, training=False, name='dev', log=[])
    fit_as_stop = SpyBatches(rows['FIT'], 100, 7, stats=stats, training=False, name='x', log=[])
    fit_fixed = SpyBatches(rows['FIT'], 100, 7, stats=stats, training=False, name='y', log=[])
    for fit, stop in ((dev, d.stop), (d.fit, dev), (d.fit, fit_as_stop), (d.stop, d.stop), (fit_fixed, d.stop)):
        with pytest.raises((ValueError, TypeError)):
            run(tmp_path / 'slots', train.TrainData(fit=fit, stop=stop, scales=d.scales, static=d.static))
    assert not checkpoints(tmp_path / 'slots')


def test_batches_without_statistics_are_refused(tmp_path, rows):
    with pytest.raises(ValueError, match='statistics'):
        run(tmp_path / 'nostats', make_data(rows, None))


@pytest.mark.parametrize('bad', ['missing', 'extra', 'zero', 'nan'])
def test_the_scales_must_be_the_thirteen_finite_positive_ones(tmp_path, rows, stats, bad):
    scales = dict(SCALES)
    if bad == 'missing':
        scales.pop(loss.BLOCKS[0])
    elif bad == 'extra':
        scales['M9/masked'] = 1.0
    elif bad == 'zero':
        scales[loss.BLOCKS[0]] = 0.0
    else:
        scales[loss.BLOCKS[0]] = float('nan')
    with pytest.raises(ValueError, match='scale'):
        run(tmp_path / 'scales', make_data(rows, stats, scales=scales))


def test_the_training_loss_is_the_task_4_loss_with_the_equal_row_weights(tmp_path, rows, stats, monkeypatch):
    """The first epoch's training loss is the mean over its batches of ``loss.total_loss``; ``train`` calls that function, it does not re-implement it."""
    seen, real = [], loss.total_loss

    def spy(nlls, scales, targets):
        out = real(nlls, scales, targets)
        if step_guard.trusted():                      # the steps of the loop; the dry run of the data checks (batch_prep.validate_rows, S4b speed-ups item 3) and the STOP evaluation are untrusted calls
            seen.append(float(out.detach()))
        return out

    monkeypatch.setattr(loss, 'total_loss', spy)
    data_ = make_data(rows, stats)
    result = run(tmp_path / 'tl', data_, epochs=1)
    n = len(data_.fit)
    assert len(seen) == n and result.fit_batch_mean_losses[0] == pytest.approx(sum(seen[:n]) / n, rel=1e-12)


def test_train_py_reads_no_cohort_reader_or_holdout_flag():
    text = Path(train.__file__).read_text(encoding='utf-8')
    tree = ast.parse(text)
    names = {n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(part in ('cohort', 'cohort_policy', 'reader', 'store_build') for name in names for part in name.split('.'))
    assert 'allow_holdout' not in text and 'holdout' not in text.lower().replace('no holdout', '')
    assert not any(call in text for call in ('.unlink(', 'rmtree(', 'os.remove(', 'os.rmdir('))        # nothing is deleted by this module


if __name__ == '__main__':
    _child_main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))


@pytest.mark.parametrize('pending_delay,work_seconds,expected_beats', [
    (5.0, 54.0, 0), (5.0, 59.0, 1), (5.0, 60.0, 1), (65.0, 0.0, 1)])
def test_pending_write_time_is_in_the_heartbeat_origin_and_orphan_charge(tmp_path, monkeypatch, pending_delay, work_seconds, expected_beats):
    """The pending write belongs to the elapsed beat time: before first due, at the review's 86403 s cut, with a complete beat, and after a write longer than a period."""
    clock = [0.0]
    monkeypatch.setattr(train, '_CLOCK', lambda: clock[0])
    monkeypatch.setattr(train, '_memory_sample', lambda: (1024, 1024, 1024, 'private_bytes_windows'))
    res = train._Resources()
    res.sampler.sample()
    res.adopt(dict(seconds=86339.0, peak_bytes=1024, peak_upper_bytes=1024, peak_rss_bytes=1024, metric='private_bytes_windows'))
    original_write = train._write_atomic

    def delayed_pending(path, payload):
        original_write(path, payload)
        if path.name.endswith('.pending.json'):
            clock[0] += pending_delay

    monkeypatch.setattr(train, '_write_atomic', delayed_pending)
    train._Call(tmp_path).begin('synthetic-fix2-identity', res)
    clock[0] += work_seconds
    res.sampler.sample()
    finals, orphan = train._load_calls(tmp_path, 'synthetic-fix2-identity')
    assert orphan['pending']['start_accounting']['seconds'] == 86339.0
    assert orphan['beats'] == expected_beats
    if expected_beats:
        assert orphan['beat']['seconds'] == pending_delay + work_seconds
    adopted = train._Resources()
    adopted.sampler.sample()
    actual = res.seconds()
    if actual > train.WALL_CAP_SECONDS:
        with pytest.raises(train.ResourceBoundError, match='24|seconds|time|wall'):
            train._adopt_orphan(tmp_path, orphan, finals, None, adopted, 'synthetic-fix2-identity')
        record = json.loads((tmp_path / 'calls/call_0001.json').read_text(encoding='utf-8'))
        assert record['status'] == 'exceeded' and record['seconds'] >= actual
    else:
        record = train._adopt_orphan(tmp_path, orphan, finals, None, adopted, 'synthetic-fix2-identity')
        assert record['status'] == 'ok' and record['seconds'] >= actual


def test_pending_accounting_and_heartbeat_share_one_clock_sample(tmp_path, monkeypatch):
    """An advancing clock exposes drift from reading the origin separately from the accounting snapshot."""
    import itertools
    monkeypatch.setattr(train, '_CLOCK', iter(itertools.count(100)).__next__)
    monkeypatch.setattr(train, '_memory_sample', lambda: (1024, 1024, 1024, 'private_bytes_windows'))
    res = train._Resources()
    res.sampler.sample()
    train._Call(tmp_path).begin('synthetic-fix2-clock', res)
    pending = json.loads((tmp_path / 'calls/call_0001.pending.json').read_text(encoding='utf-8'))
    assert pending['start_accounting']['seconds'] == res.beat_origin - res.started
    assert res.beat_at == res.beat_origin


@pytest.mark.parametrize('head,values,matches,rows,expected_scale', [
    ('M3Kills', [2.0, 2.0, 8.0], [0, 0, 1], [0, 1, 2], 4.0),
    ('M3Gold', [2.0, 8.0, 20.0], [0, 0, 0], [0, 0, 1], 10.0)])
def test_identical_heterogeneous_b0_scores_need_not_give_a_unit_stop_ratio(head, values, matches, rows, expected_scale):
    """Equal per-entry units do not equate FIT entry averaging and STOP evaluation-row/match averaging, even on identical scores."""
    from c_ept.ept.scales import head_scale
    from c_ept.data.head_rows import HeadScores
    values, matches, rows = np.array(values), np.array(matches), np.array(rows)
    sizes = np.zeros(3, dtype=np.int64)
    scale, _, _ = head_scale(head, HeadScores(values, matches, rows), sizes)
    assert scale == expected_scale
    meter = train.StopMeter()
    meter.add(head + '/masked', values, np.ones(3, bool), matches, row=rows, size=sizes)
    assert meter.value({head + '/masked': scale}) == pytest.approx(1.25, rel=1e-12)


def test_q10_components_are_measured_and_checkpoint_schema_is_explicit(tmp_path, rows, stats):
    result = run(tmp_path / 'q10', make_data(rows, stats), epochs=1)
    assert train.SCHEMA == 'ept_train_checkpoint_v5'
    timing=result.timing_components
    for key in ('transfer_seconds','forward_seconds','backward_seconds','optimizer_seconds'):
        assert len(timing['epochs'][key]) == result.epochs_run
        assert all(math.isfinite(v) and v >= 0 for v in timing['epochs'][key])
        assert result.head['timing'][key] == timing['epochs'][key]
    assert len(timing['checkpoint_io_current_call']) == result.epochs_run
    assert all(x['seconds'] >= 0 for x in timing['checkpoint_io_current_call'])
    assert timing['checkpoint_io_scope'] == 'current call only; previous-attempt writes unavailable'


def test_checkpoint_missing_q10_components_is_refused_before_tensor_loading(tmp_path, rows, stats):
    folder=tmp_path / 'q10_missing'
    result=run(folder,make_data(rows,stats),epochs=1)
    head_path=result.checkpoint_dir/'head.json'
    doc=json.loads(head_path.read_text(encoding='utf-8'))
    doc['timing'].pop('forward_seconds')
    head_path.write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(RuntimeError,match='schema|telemetry'):
        train._latest_checkpoint(folder)


# =====================================================================================================================================
# AdamW parameter groups and the record of kappa (S4b preparation M4, decision D6, review finding A2-F2)
# =====================================================================================================================================
# The defect. One decay for every parameter pulls each log kappa toward 0 (kappa toward 1 s, a unit-dependent prior) and the FIT-initialised output biases toward 0: with 4,511
# steps an epoch the decay alone multiplies a log kappa by 0.543 after 18 epochs (80 s -> 10.8 s). ``decay_groups='split'`` gives no decay to every log kappa, every bias,
# LayerNorm and Embedding and the decay to the Linear weight matrices, the co-participation weights and the learned terrain field; ``'all'`` (the default) is the call of every
# earlier run. What the "default" tests prove: a test can only compare the code as it is now, so the ``'all'`` tests compare with the call written out in the test (the AdamW
# of every earlier run) and with an explicit ``'all'``; the evidence for the bits of the base commit is the proof script ``m4_proof_ept.py`` of the T012 session folder.
KAPPA2 = encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=2, field_channels=2)       # two decay scales: two kappa columns


def split_cfg(**over):
    return make_cfg(encoder=KAPPA2, decay_groups='split', **over)


def build_model(kind: str, seed: int = 5):
    """A seeded model of the training loop: the EPT of size S or M (3 decay scales, 8 learned terrain channels) or the N2 team-set model of the comparison."""
    torch.manual_seed(seed)
    if kind == 'N2':
        from c_ept.models.neural import TeamSetModel
        return TeamSetModel(encoder.SIZES['S'], toy_static(), hidden=32)
    return heads.EPT(encoder.SIZES[kind[-1]], toy_static())


def two_steps(model, optimizer):
    """Two optimizer steps on a loss that every parameter feeds with a gradient of its own size (no data, no randomness)."""
    for _ in range(2):
        total = sum((i + 1) * p.square().sum() for i, p in enumerate(model.parameters()))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
    return model


@pytest.fixture(scope='module')
def split_run(tmp_path_factory, rows, stats):
    """An uninterrupted run of EPOCHS epochs under ``decay_groups='split'`` with two decay scales."""
    folder = tmp_path_factory.mktemp('train_split') / 'run'
    return run(folder, make_data(rows, stats), split_cfg()), folder


@pytest.fixture(scope='module')
def all_run_k2(tmp_path_factory, rows, stats):
    """The same run under the default policy: the reference of the identity and of the curves."""
    folder = tmp_path_factory.mktemp('train_all_k2') / 'run'
    return run(folder, make_data(rows, stats), make_cfg(encoder=KAPPA2)), folder


@pytest.mark.parametrize('kind', ['EPT S', 'EPT M', 'N2'])
def test_split_groups_cover_every_parameter_once(kind):
    """Every parameter of the model lands in exactly one group (the groups partition ``model.parameters()``), whatever the model; the policy ``'all'`` is one group."""
    model = build_model(kind)
    everything = list(model.parameters())
    groups = train.param_groups(model, 0.01, 'split')
    flat = [p for g in groups for p in g['params']]
    assert len(flat) == len(everything) and len({id(p) for p in flat}) == len(flat) and {id(p) for p in flat} == {id(p) for p in everything}
    assert [g['weight_decay'] for g in groups] == [0.01, 0.0] and all(g['params'] for g in groups) and all(set(g) == {'params', 'weight_decay'} for g in groups)
    assert list(train.decay_assignment(model, 0.01)) == [name for name, _ in model.named_parameters()]        # a decay for each name, each name once
    one = train.param_groups(model, 0.01, 'all')
    assert len(one) == 1 and one[0]['weight_decay'] == 0.01 and [id(p) for p in one[0]['params']] == [id(p) for p in everything]


def test_split_groups_assign_decay_by_rule():
    """One name of each kind of the rule. The names are code (a rename of a model part is seen here), not data. The distance and geodesic networks sit in the module attribute
    ``bias`` of the encoder (``encoder.bias``): the rule is on the name of the parameter itself, so their weight matrices decay although the path of their names holds 'bias'."""
    model = build_model('EPT S')
    got = train.decay_assignment(model, 0.01)
    no_decay = ['encoder.bias.co.log_kappa',                                  # log kappa
                'encoder.bias.type_bias',                                     # type_bias
                'encoder.players_in.bias', 'encoder.blocks.0.qkv.bias',       # the bias of a Linear layer
                'heads.rate_net.bias',                                        # the hidden bias of the head networks
                'heads.rate_net.out.bias', 'heads.kill_out.bias', 'heads.gold_out.bias',     # output biases (FIT-initialised)
                'encoder.blocks.0.norm1.weight', 'encoder.blocks.0.norm1.bias', 'encoder.norm.weight',          # LayerNorm
                'encoder.zone_emb.weight', 'heads.bin_emb.weight', 'heads.gold_slot.weight', 'heads.mark.kind.weight']        # Embedding
    decayed = ['encoder.players_in.weight', 'encoder.blocks.0.qkv.weight', 'encoder.blocks.0.ffn.0.weight',          # Linear weight matrices
               'heads.rate_net.maps.0.weight', 'heads.rate_net.out.weight', 'heads.gold_in.weight', 'heads.mark.killer.weight', 'encoder.field_proj.weight',
               'encoder.bias.dist.0.net.0.weight', 'encoder.bias.geo.2.net.2.weight',             # Linear weights under the module attribute named 'bias'
               'encoder.bias.co.w', 'encoder.bias.co.w_back',                # the co-participation weights
               'encoder.field.learned']                                      # the learned terrain field (an L2 prior around 0)
    for name in no_decay:
        assert got[name] == 0.0, name
    for name in decayed:
        assert got[name] == 0.01, name
    # a second implementation written from the text of the rule, for every parameter of the model
    for name in got:
        parent, _, leaf = name.rpartition('.')
        zero = leaf in ('log_kappa', 'type_bias', 'bias') or isinstance(model.get_submodule(parent), (torch.nn.LayerNorm, torch.nn.Embedding))
        assert got[name] == (0.0 if zero else 0.01), name
        assert zero or leaf in ('weight', 'w', 'w_back', 'learned'), name          # what decays is a weight matrix, w, w_back or the field and nothing else
    assert got['encoder.bias.co.log_kappa'] != got['encoder.bias.co.w']
    # the decay is the weight decay that was asked for, not a constant of the module
    assert train.decay_assignment(model, 0.5)['encoder.players_in.weight'] == 0.5 and train.decay_assignment(model, 0.0)['encoder.players_in.weight'] == 0.0
    split = train.param_groups(model, 0.0, 'split')                    # a decay of 0 still gives the two groups (the policy, not the number, makes the groups)
    assert [g['weight_decay'] for g in split] == [0.0, 0.0] and sum(len(g['params']) for g in split) == len(list(model.parameters()))


def test_a_parameter_without_a_rule_is_a_value_error_naming_it_and_the_policy_name_is_checked():
    class Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.gamma = torch.nn.Parameter(torch.ones(2))

    with pytest.raises(ValueError, match='gamma'):
        train.param_groups(Odd(), 0.01, 'split')
    with pytest.raises(ValueError, match=r'conv\.weight'):
        train.param_groups(torch.nn.ModuleDict({'conv': torch.nn.Conv1d(1, 1, 1)}), 0.01, 'split')
    train.param_groups(Odd(), 0.01, 'all')                              # the policy 'all' has no rule to miss
    for bad in ('Split', 'none', '', None, 1, ('split',)):
        with pytest.raises(ValueError, match='decay_groups'):
            train.param_groups(build_model('EPT S'), 0.01, bad)


def test_all_policy_is_bit_identical(tmp_path, rows, stats, base):
    """The policy ``'all'`` is the AdamW call of every earlier run: one group of ``model.parameters()`` with the weight decay as a keyword of the optimizer, so the optimizer
    state after a step, the bytes of its state dict, the identity, the head and the epoch metrics are those of the code before this task."""
    cfg = make_cfg()
    model = build_model('EPT S')
    optimizer = train._optimizer(model, cfg)
    assert len(optimizer.param_groups) == 1 and optimizer.param_groups[0]['weight_decay'] == cfg.weight_decay
    two_steps(model, optimizer)
    old_model = build_model('EPT S')
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)       # the line of every earlier run, written out
    two_steps(old_model, old_optimizer)
    assert_same_state(model.state_dict(), old_model.state_dict(), 'model')
    assert_same_state(optimizer.state_dict(), old_optimizer.state_dict(), 'optimizer')
    assert train._torch_bytes(optimizer.state_dict()) == train._torch_bytes(old_optimizer.state_dict())       # the bytes too: the order of the keys of the group
    # a whole run: the keyword changes nothing, and nothing new is written under the default
    result, folder = base
    again = run(tmp_path / 'explicit', make_data(rows, stats), make_cfg(decay_groups='all'))
    assert_same_run(result, again)
    a, b = last_pt(folder, EPOCHS), last_pt(tmp_path / 'explicit', EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)
    assert 'kappa' not in a['loop'] and len(a['optimizer']['param_groups']) == 1 and a['optimizer']['param_groups'][0]['weight_decay'] == 0.01
    for head_ in (checkpoints(folder)[-1][0], again.head):
        assert 'decay_groups' not in head_ and head_['schema'] == 'ept_train_checkpoint_v6'
        assert 'decay_groups' not in head_['identity'] and 'decay_groups_sha256' not in head_['identity']
    assert again.head['identity'] == checkpoints(folder)[-1][0]['identity']
    snapshot = json.loads((folder / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.json').read_text(encoding='utf-8'))
    assert snapshot['schema'] == 'epoch-metrics-v1' and all(set(row) == set(train.epoch_metrics.COLUMNS) for row in snapshot['rows'])
    assert (folder / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.csv').read_text(encoding='utf-8').split('\n')[0] == ','.join(train.epoch_metrics.COLUMNS)


def test_the_split_groups_step_like_a_uniform_decay_of_their_own_value():
    """The decay of AdamW is decoupled and acts on each element alone, so the parameters of the no-decay group must move exactly as under a uniform decay of 0 and the
    others exactly as under a uniform 0.01: two steps, bit for bit. (Controls: the two uniform runs differ from each other, and log kappa moves differently under 0.01.)"""
    cfg = split_cfg()

    def stepped(make):
        model = build_model('EPT S')
        return two_steps(model, make(model))
    split = stepped(lambda m: train._optimizer(m, cfg))
    uniform = stepped(lambda m: torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=0.01))
    plain = stepped(lambda m: torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=0.0))
    rule = train.decay_assignment(build_model('EPT S'), 0.01)
    uniform_parameters, plain_parameters = dict(uniform.named_parameters()), dict(plain.named_parameters())
    for name, p in split.named_parameters():
        assert torch.equal(p, (uniform_parameters if rule[name] else plain_parameters)[name]), name
    assert not torch.equal(uniform_parameters['encoder.bias.co.log_kappa'], plain_parameters['encoder.bias.co.log_kappa'])
    assert not torch.equal(uniform_parameters['encoder.players_in.weight'], plain_parameters['encoder.players_in.weight'])
    assert not torch.equal(split.encoder.bias.co.log_kappa, uniform.encoder.bias.co.log_kappa)
    assert not torch.equal(split.encoder.players_in.weight, plain.encoder.players_in.weight)


def test_the_train_config_has_decay_groups_after_the_k_condition_with_the_default_all_and_refuses_another_name():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    assert names.index('decay_groups') == names.index('k_condition') + 1 and names[:6] == ['encoder', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'model_spec']
    small = encoder.EncoderConfig(d=16, layers=1, heads=2)
    assert train.TrainConfig(small, 1e-3, 0.0, 1).decay_groups == 'all' and train.TrainConfig(small, 1e-3, 0.0, 1, 2, None, 'poisson', 'raw').decay_groups == 'all'
    assert train.TrainConfig(small, 1e-3, 0.0, 1, decay_groups='split').decay_groups == 'split'
    assert train.DECAY_GROUPS == ('all', 'split')
    for bad in ('Split', 'none', '', None, 1):
        with pytest.raises(ValueError, match='decay_groups'):
            train.TrainConfig(small, 1e-3, 0.0, 1, decay_groups=bad)
    import inspect
    assert list(inspect.signature(train.param_groups).parameters) == ['model', 'weight_decay', 'policy']


def test_a_split_run_records_the_policy_and_the_hash_of_its_rule_in_the_identity_and_the_default_run_does_not(split_run, all_run_k2, monkeypatch):
    split, _ = split_run
    reference, _ = all_run_k2
    identity = split.head['identity']
    assert identity['decay_groups'] == 'split' and identity['decay_groups_sha256'] == train.decay_rule_sha256()
    assert len(identity['decay_groups_sha256']) == 64 and set(identity['decay_groups_sha256']) <= set('0123456789abcdef')
    assert {k: v for k, v in identity.items() if k not in ('decay_groups', 'decay_groups_sha256')} == reference.head['identity']          # nothing else moves
    assert 'decay_groups' not in reference.head['identity'] and 'decay_groups_sha256' not in reference.head['identity']              # absent means 'all'
    # the hash is the hash of the rule that is applied: another rule table is another hash and another assignment
    before = train.decay_rule_sha256()
    monkeypatch.setattr(train, 'DECAY_RULES', tuple((c, leaf, 'weight_decay') if leaf == 'log_kappa' else (c, leaf, kind) for c, leaf, kind in train.DECAY_RULES))
    assert train.decay_rule_sha256() != before
    assert train.decay_assignment(build_model('EPT S'), 0.01)['encoder.bias.co.log_kappa'] == 0.01              # log kappa decayed by that table
    monkeypatch.setattr(train, 'DECAY_RULES', tuple(rule for rule in train.DECAY_RULES if rule[1] != 'log_kappa'))
    with pytest.raises(ValueError, match='log_kappa'):                                                            # and without a rule it is refused, not guessed
        train.decay_assignment(build_model('EPT S'), 0.01)


def test_a_split_head_names_the_groups_it_trained_with(split_run, all_run_k2):
    split, folder = split_run
    record = split.head['decay_groups']
    assignment = train.decay_assignment(split.model, 0.01)
    assert record['policy'] == 'split' and record['rule_sha256'] == train.decay_rule_sha256()
    assert record['assignment_sha256'] == hashlib.sha256(json.dumps([[n, d] for n, d in sorted(assignment.items())]).encode()).hexdigest()
    sizes = {0.01: [p.numel() for n, p in split.model.named_parameters() if assignment[n]], 0.0: [p.numel() for n, p in split.model.named_parameters() if not assignment[n]]}
    assert record['groups'] == [{'weight_decay': 0.01, 'tensors': len(sizes[0.01]), 'elements': sum(sizes[0.01])},
                                {'weight_decay': 0.0, 'tensors': len(sizes[0.0]), 'elements': sum(sizes[0.0])}]
    assert split.head['schema'] == 'ept_train_checkpoint_v7' and checkpoints(folder)[-1][0] == split.head
    saved = last_pt(folder, EPOCHS)['optimizer']['param_groups']              # the optimizer state of the run has the two groups, the indices partition the parameters
    assert [g['weight_decay'] for g in saved] == [0.01, 0.0] and sorted(i for g in saved for i in g['params']) == list(range(len(assignment)))
    assert 'decay_groups' not in all_run_k2[0].head


def test_epoch_metrics_records_kappa_for_ept_and_blank_for_n2(tmp_path, rows, stats, split_run, base):
    """Under ``'split'`` every epoch row has one column for each decay scale: exp(log kappa) of the model at the end of that epoch (EPT), blank for N2, which has no decay scale.
    Under ``'all'`` there is no kappa column and the schema is the old one."""
    from c_ept.comparison import models as comparison_models
    result, folder = split_run
    em = train.epoch_metrics
    columns = (*em.COLUMNS, 'kappa_0', 'kappa_1')
    doc = train.checkpoints.latest_metadata(folder)[2]
    assert doc['schema'] == 'epoch-metrics-v2' and len(doc['rows']) == EPOCHS and all(set(row) == set(columns) for row in doc['rows'])
    for epoch, row in enumerate(doc['rows'], 1):
        want = torch.exp(last_pt(folder, epoch)['model']['encoder.bias.co.log_kappa']).tolist()            # read from the weights of that epoch's checkpoint
        assert [row['kappa_0'], row['kappa_1']] == want and all(isinstance(v, float) and v > 0 for v in want), epoch
    assert doc['rows'][0]['kappa_0'] != doc['rows'][-1]['kappa_0']                    # recorded at each epoch: the scale moves
    assert last_pt(folder, EPOCHS)['loop']['kappa'] == [[r['kappa_0'], r['kappa_1']] for r in doc['rows']]
    import csv
    parsed = list(csv.DictReader((folder / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.csv').read_text(encoding='utf-8').splitlines()))
    assert list(parsed[0]) == list(columns) and [float(r['kappa_1']) for r in parsed] == [r['kappa_1'] for r in doc['rows']]
    # the default policy: no kappa column, the schema of every earlier run
    old = train.checkpoints.latest_metadata(base[1])[2]
    assert old['schema'] == 'epoch-metrics-v1' and all(set(row) == set(em.COLUMNS) for row in old['rows'])
    # N2 has no co-participation: the columns are there (one for each configured scale) and blank
    spec = comparison_models.specification('n2', hidden=8)
    n2 = run(tmp_path / 'n2', make_data(rows, stats), split_cfg(model_spec=spec), epochs=2)
    n2_doc = train.checkpoints.latest_metadata(tmp_path / 'n2')[2]
    assert n2.head['schema'] == 'ept_train_checkpoint_v7' and n2.head['identity']['model_spec']['architecture'] == 'n2_teamset_v1'
    assert n2_doc['rows'] and all(row['kappa_0'] is None and row['kappa_1'] is None and set(row) == set(columns) for row in n2_doc['rows'])
    n2_csv = list(csv.DictReader((tmp_path / 'n2' / 'epoch_0002' / 'epoch_metrics.csv').read_text(encoding='utf-8').splitlines()))
    assert list(n2_csv[0]) == list(columns) and all(r['kappa_0'] == '' and r['kappa_1'] == '' for r in n2_csv)
    assert 'encoder.bias.co.log_kappa' not in n2.model.state_dict() and n2.head['decay_groups']['policy'] == 'split'
    assert not any('kappa' in name for name in n2.model.state_dict())


@pytest.mark.parametrize('first,second', [('all', 'split'), ('split', 'all')])
def test_resume_refuses_a_checkpoint_from_another_policy(tmp_path, rows, stats, first, second):
    """The optimizer state is indexed by the position of the parameters in its groups, which the policy changes, so a checkpoint of one policy is not resumed under the other:
    the identity differs (the keys ``decay_groups`` and ``decay_groups_sha256`` are absent under 'all'), before any work and with nothing written, in both directions."""
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), make_cfg(encoder=KAPPA2, decay_groups=first), check=cut_on(3))
    assert len(checkpoints(folder)) == 2
    before, log = file_hashes(folder), []
    with pytest.raises(ValueError, match='decay_groups') as raised:
        run(folder, make_data(rows, stats, log=log), make_cfg(encoder=KAPPA2, decay_groups=second))
    assert "['decay_groups', 'decay_groups_sha256'] differ" in str(raised.value)              # both keys are named, whichever side holds them
    assert no_work(log) and file_hashes(folder) == before


def test_a_split_run_resumes_with_the_same_bits_and_the_same_kappa_record(tmp_path, rows, stats, split_run):
    full, full_folder = split_run
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), split_cfg(), check=cut_on(3))
    resumed = run(folder, make_data(rows, stats), split_cfg())
    assert_same_run(full, resumed)
    a, b = last_pt(full_folder, EPOCHS), last_pt(folder, EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)
    assert a['loop']['kappa'] == b['loop']['kappa'] and len(b['loop']['kappa']) == EPOCHS
    for name in ('epoch_metrics.json', 'epoch_metrics.csv'):
        assert (folder / f'epoch_{EPOCHS:04d}' / name).read_bytes() == (full_folder / f'epoch_{EPOCHS:04d}' / name).read_bytes(), name
    assert resumed.head['decay_groups'] == full.head['decay_groups']


def test_a_resume_refuses_a_checkpoint_whose_group_assignment_is_not_the_one_of_the_code(tmp_path, rows, stats, monkeypatch):
    """The identity holds the hash of the rule table; the head holds the hash of the assignment that the rule gave to the parameters of the model. A code whose rule gives
    another assignment (the same table, another reading of it) is refused before the optimizer state is loaded into groups that do not mean what they meant."""
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), split_cfg(), check=cut_on(3))
    real, log = train.decay_assignment, []

    def other(model, weight_decay):
        return {**real(model, weight_decay), 'encoder.bias.co.log_kappa': weight_decay}              # as if log kappa were decayed

    monkeypatch.setattr(train, 'decay_assignment', other)
    with pytest.raises(ValueError, match='assignment'):
        run(folder, make_data(rows, stats, log=log), split_cfg())
    assert no_work(log) and len(checkpoints(folder)) == 2


# =====================================================================================================================================
# the terrain smoothness weight lambda_Omega (EPT-T012 S4b preparation M6, decision D8, review finding A2-F5)
# =====================================================================================================================================
SMOOTH = 0.5


def smooth_cfg(**over):
    return make_cfg(terrain_smooth_weight=SMOOTH, **over)


@pytest.fixture(scope='module')
def smooth_run(tmp_path_factory, rows, stats):
    """An uninterrupted run of EPOCHS epochs with lambda_Omega = 0.5, with the R_Omega of every optimizer step logged by a spy on ``training_step.terrain_penalty``."""
    folder = tmp_path_factory.mktemp('train_smooth') / 'run'
    seen = []
    real = train.training_step.terrain_penalty
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train.training_step, 'terrain_penalty', lambda fa: (seen.append(float(real(fa).detach())), real(fa))[1])
        result = run(folder, make_data(rows, stats), smooth_cfg())
    return result, folder, seen


def test_the_train_config_has_terrain_smooth_weight_after_the_population_with_the_default_zero_and_refuses_what_is_not_a_finite_number_that_is_not_negative():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    assert names.index('terrain_smooth_weight') == names.index('stats_population') + 1          # relative order: the fields of the next tasks (M7 swap_mode, eval_average) come after it
    small = encoder.EncoderConfig(d=16, layers=1, heads=2)
    assert train.TrainConfig(small, 1e-3, 0.0, 1).terrain_smooth_weight == 0.0 and train.TrainConfig(small, 1e-3, 0.0, 1, terrain_smooth_weight=0.25).terrain_smooth_weight == 0.25
    assert train.TrainConfig(small, 1e-3, 0.0, 1, terrain_smooth_weight=2).terrain_smooth_weight == 2
    for bad in (-0.1, float('nan'), float('inf'), -float('inf'), True, '0.5', None, [0.5]):
        with pytest.raises(ValueError, match='terrain_smooth_weight'):
            train.TrainConfig(small, 1e-3, 0.0, 1, terrain_smooth_weight=bad)


def test_zero_terrain_weight_is_bit_identical(tmp_path, rows, stats, base):
    """A weight of 0 is the default: no penalty in the loss, so the weights, the optimizer state, the random state, the identity, the head, the epoch metrics and the bytes of the metric
    files are those of the run without the setting (the K8 comparison reads ``epoch_metrics`` with ``==``); the loop of ``last.pt`` has no key for the penalty and the metrics no column."""
    result, folder = base
    again = run(tmp_path / 'explicit', make_data(rows, stats), make_cfg(terrain_smooth_weight=0.0))
    assert_same_run(result, again)
    a, b = last_pt(folder, EPOCHS), last_pt(tmp_path / 'explicit', EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)
    assert 'terrain_penalty' not in a['loop'] and 'terrain_penalty' not in b['loop'] and list(a['loop']) == list(b['loop'])
    for head_ in (checkpoints(folder)[-1][0], again.head):
        assert head_['schema'] == 'ept_train_checkpoint_v6' and 'terrain_smooth_weight' not in head_['identity']
    assert again.head['identity'] == checkpoints(folder)[-1][0]['identity']
    for name in ('epoch_metrics.json', 'epoch_metrics.csv', 'best.pt'):                    # last.pt holds the wall-clock timings of the loop, so only its parts are compared above
        first, second = folder / f'epoch_{EPOCHS:04d}' / name, tmp_path / 'explicit' / f'epoch_{EPOCHS:04d}' / name
        if first.exists() or second.exists():
            assert first.read_bytes() == second.read_bytes(), name
    snapshot = json.loads((tmp_path / 'explicit' / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.json').read_text(encoding='utf-8'))
    assert snapshot['schema'] == 'epoch-metrics-v1' and all(set(row) == set(train.epoch_metrics.COLUMNS) for row in snapshot['rows'])


def test_a_weighted_run_records_the_weight_in_the_identity_and_the_penalty_in_a_v8_head_and_the_metrics(smooth_run, base):
    result, folder, seen = smooth_run
    reference, _ = base
    identity = result.head['identity']
    assert identity['terrain_smooth_weight'] == SMOOTH and {k: v for k, v in identity.items() if k != 'terrain_smooth_weight'} == reference.head['identity']          # nothing else moves
    assert result.head['schema'] == 'ept_train_checkpoint_v8' and checkpoints(folder)[-1][0] == result.head
    doc = train.checkpoints.latest_metadata(folder)[2]
    columns = (*train.epoch_metrics.COLUMNS, 'terrain_penalty')
    assert doc['schema'] == 'epoch-metrics-v3' and len(doc['rows']) == EPOCHS and all(set(row) == set(columns) for row in doc['rows'])
    assert (folder / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.csv').read_text(encoding='utf-8').split('\n')[0] == ','.join(columns)
    steps = len(seen) // EPOCHS
    assert steps == FIT_BATCHES and len(seen) == EPOCHS * steps                                       # one R_Omega for every optimizer step
    for epoch, row in enumerate(doc['rows']):
        assert row['terrain_penalty'] == sum(seen[epoch * steps:(epoch + 1) * steps]) / steps         # the mean over the steps of that epoch, unweighted
        assert math.isfinite(row['terrain_penalty']) and row['terrain_penalty'] > 0
    assert seen[0] == 0.0                                                                             # the learned field starts at 0, so the first step has R_Omega = 0
    assert last_pt(folder, EPOCHS)['loop']['terrain_penalty'] == [row['terrain_penalty'] for row in doc['rows']]
    assert result.fit_batch_mean_losses != reference.fit_batch_mean_losses and not all(torch.equal(a, b) for a, b in zip(result.model.state_dict().values(), reference.model.state_dict().values()))      # the penalty changes the training (control)


def test_stop_value_excludes_the_penalty(tiny_model, rows, stats, smooth_run, monkeypatch):
    """The STOP value is the evaluation of the model on the STOP rows and nothing else: ``evaluate_stop`` has no weight, never asks for R_Omega, and the STOP value of a weighted run is the one of
    its own weights."""
    import inspect
    assert 'smooth_weight' not in inspect.signature(train.evaluate_stop).parameters and 'terrain_smooth_weight' not in inspect.signature(train.evaluate_stop).parameters
    batches = data.Batches(rows['STOP'], 100, 0, stats=stats, training=False)
    want = train.evaluate_stop(tiny_model, batches, SCALES)
    for owner, name in ((encoder.TerrainField, 'penalty'), (train.training_step, 'terrain_penalty')):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail('the penalty was asked for in the STOP evaluation'))
    assert train.evaluate_stop(tiny_model, batches, SCALES) == want
    monkeypatch.undo()
    result, folder, _ = smooth_run
    torch.manual_seed(0)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), toy_static())
    model.load_state_dict(last_state(folder, EPOCHS))
    again = train.evaluate_stop(model, data.Batches(rows['STOP'], len(rows['STOP'].units), 7, stats=stats, training=False), SCALES)
    tolerance = 1e-8 * abs(result.stop_losses[-1])                                 # the two evaluations differ by float32 reduction noise only (about 3e-10 relative in this toy run)
    assert abs(again - result.stop_losses[-1]) <= tolerance
    # the control: the term that must be missing (about 1e-5 absolute against a STOP value of about 30 in this toy run) is far above that tolerance, so this test can see it (a mutation that adds
    # lambda * R_Omega to ``stop_value`` in ``train`` fails the line above)
    assert SMOOTH * float(model.encoder.field.penalty().detach()) > 10 * tolerance


def test_the_terrain_weight_is_refused_when_the_model_has_no_learned_field_channel_before_any_work(tmp_path, rows, stats):
    from c_ept.comparison import models as comparison_models
    for name, cfg in (('ept', make_cfg(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=0), terrain_smooth_weight=SMOOTH)),
                      ('n2', make_cfg(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=0), terrain_smooth_weight=SMOOTH,
                                      model_spec=comparison_models.specification('n2', hidden=8)))):
        log = []
        with pytest.raises(ValueError, match='terrain_smooth_weight.*field_channels'):
            run(tmp_path / name, make_data(rows, stats, log=log), cfg)
        assert no_work(log) and checkpoints(tmp_path / name) == []
    run(tmp_path / 'zero', make_data(rows, stats), make_cfg(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=0)), epochs=1)      # weight 0, no channel: fine


@pytest.mark.parametrize('first,second', [(0.0, SMOOTH), (SMOOTH, 0.0), (SMOOTH, 0.25)])
def test_resume_refuses_a_checkpoint_under_another_terrain_weight_in_both_directions(tmp_path, rows, stats, first, second):
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), make_cfg(terrain_smooth_weight=first), check=cut_on(3))
    assert len(checkpoints(folder)) == 2
    before, log = file_hashes(folder), []
    with pytest.raises(ValueError, match='terrain_smooth_weight'):
        run(folder, make_data(rows, stats, log=log), make_cfg(terrain_smooth_weight=second))
    assert no_work(log) and file_hashes(folder) == before


def test_a_weighted_run_resumes_with_the_same_bits_and_the_same_metrics(tmp_path, rows, stats, smooth_run):
    full, full_folder, _ = smooth_run
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, make_data(rows, stats), smooth_cfg(), check=cut_on(3))
    resumed = run(folder, make_data(rows, stats), smooth_cfg())
    assert_same_run(full, resumed)
    a, b = last_pt(full_folder, EPOCHS), last_pt(folder, EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)
    assert a['loop']['terrain_penalty'] == b['loop']['terrain_penalty'] and len(b['loop']['terrain_penalty']) == EPOCHS
    for name in ('epoch_metrics.json', 'epoch_metrics.csv'):
        assert (folder / f'epoch_{EPOCHS:04d}' / name).read_bytes() == (full_folder / f'epoch_{EPOCHS:04d}' / name).read_bytes(), name


def test_a_weighted_split_run_has_the_kappa_columns_and_then_the_terrain_column_in_a_v8_head(tmp_path, rows, stats):
    result = run(tmp_path / 'run', make_data(rows, stats), split_cfg(terrain_smooth_weight=SMOOTH), epochs=2)
    doc = train.checkpoints.latest_metadata(tmp_path / 'run')[2]
    columns = (*train.epoch_metrics.COLUMNS, 'kappa_0', 'kappa_1', 'terrain_penalty')
    assert result.head['schema'] == 'ept_train_checkpoint_v8' and doc['schema'] == 'epoch-metrics-v3' and result.head['identity']['decay_groups'] == 'split'
    assert all(list(row) == list(columns) or set(row) == set(columns) for row in doc['rows']) and train.epoch_metrics.columns(result.head) == columns
    assert (tmp_path / 'run' / 'epoch_0002' / 'epoch_metrics.csv').read_text(encoding='utf-8').split('\n')[0] == ','.join(columns)
    assert all(row['kappa_0'] > 0 and row['terrain_penalty'] > 0 for row in doc['rows']) and 'decay_groups' in result.head


def test_a_team_set_run_smooths_its_field_too_and_records_a_number(tmp_path, rows, stats):
    from c_ept.comparison import models as comparison_models
    result = run(tmp_path / 'n2', make_data(rows, stats), make_cfg(terrain_smooth_weight=SMOOTH, model_spec=comparison_models.specification('n2', hidden=8)), epochs=2)
    doc = train.checkpoints.latest_metadata(tmp_path / 'n2')[2]
    assert result.head['schema'] == 'ept_train_checkpoint_v8' and all(isinstance(row['terrain_penalty'], float) and row['terrain_penalty'] > 0 for row in doc['rows'])


# =====================================================================================================================================
# the random view swap in training and the evaluation average (EPT-T012 S4b preparation M7, decisions D10 P1 and D17, plan Q3)
# =====================================================================================================================================
def coins(seed, epoch, index, views):
    """Plan Q3 written out: one fair coin for each distinct view of batch ``index`` of epoch ``epoch``, from SeedSequence([seed, epoch, index, 0x5157])."""
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index, 0x5157])).integers(0, 2, size=views).astype(bool).tolist()


def swap_cfg(mode, **over):
    return make_cfg(swap_mode=mode, **over)


def swap_data(rows, stats, mode, **over):
    return make_data(rows, stats, swap_mode=mode, **over)


def spy_encodes(patch):
    """A log of the calls of ``EPT.encode`` (the encoder of the model): ``(training mode, grad enabled, side markers of the views, number of views)``."""
    log, real = [], heads.EPT.encode

    def encode(self, batch):
        log.append((self.training, torch.is_grad_enabled(), batch['side'].clone(), int(len(batch['view']))))
        return real(self, batch)

    patch.setattr(heads.EPT, 'encode', encode)
    return log


@pytest.mark.parametrize('mode,calls', [('average', 2), ('random_view', 1), ('none', 1)])
def test_random_view_encodes_each_view_once_in_training(tmp_path, rows, stats, mode, calls):
    """The cost claim of D10 P1: the two-frame average runs the encoder twice for each training batch (both frames of every view), a random view or no swap once (one frame of every view).
    The coins of 'random_view' are those of plan Q3, they reach the encoder as the side marker of its views; 'none' encodes the recorded frame; the STOP evaluation keeps the average
    in every mode (``eval_average`` is True): two calls for its one batch."""
    with pytest.MonkeyPatch.context() as patch:
        log = spy_encodes(patch)
        run(tmp_path / mode, swap_data(rows, stats, mode), swap_cfg(mode), epochs=2)
    training, evaluation = [e for e in log if e[0]], [e for e in log if not e[0]]
    assert len(training) == calls * FIT_BATCHES * 2 and all(e[1] for e in training)                 # with gradients, in training mode
    assert len(evaluation) == 2 * 2 and not any(e[1] for e in evaluation)                           # STOP: one batch, two frames, no gradients, twice (two epochs)
    for e in evaluation[0::2]:
        assert not e[2].any()
    for e in evaluation[1::2]:
        assert e[2].all()
    if mode == 'average':
        assert all(not a[2].any() and b[2].all() and a[3] == b[3] for a, b in zip(training[0::2], training[1::2]))      # the recorded frame, then its swap
    elif mode == 'none':
        assert not any(e[2].any() for e in training)
    else:
        sides = [e[2].tolist() for e in training]
        assert sides == [[bool(x) for x in coins(7, epoch, index, len(sides[epoch * FIT_BATCHES + index]))] for epoch in range(2) for index in range(FIT_BATCHES)]
        assert any(any(s) for s in sides) and not all(all(s) for s in sides)                        # control: not a constant


def test_random_view_choice_is_seeded_and_independent_of_dropout(tmp_path, rows, stats):
    """The coins are drawn in the batches, not in the model: two runs with other dropout (other masks, other weights after the first step) see the same coins, the same coins as the
    generator of plan Q3 gives, and the same ones again in a second run with the same settings."""
    seen = {}
    for name, dropout in (('a', 0.1), ('b', 0.0), ('c', 0.1)):
        with pytest.MonkeyPatch.context() as patch:
            log = spy_encodes(patch)
            cfg = swap_cfg('random_view', encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=dropout, kappas=1, field_channels=2))
            run(tmp_path / name, swap_data(rows, stats, 'random_view'), cfg, epochs=2)
        seen[name] = [e[2].tolist() for e in log if e[0]]
    assert seen['a'] == seen['b'] == seen['c'] and len(seen['a']) == 2 * FIT_BATCHES
    assert seen['a'][0] == [float(x) for x in coins(7, 0, 0, len(seen['a'][0]))]


def test_rows_follow_their_view_frame_in_the_scored_blocks(tiny_model, rows, stats):
    """Plan M7: the blocks that ``score_blocks`` scores hold the rows in the frame of their view. A batch with some coins up is scored with the one-frame wrapper; for each of the 13 blocks
    the target fields are, bit for bit, the swapped ones where the coin of the row's view is up and the recorded ones where it is not, and the score of each row is the score it has
    in the fully swapped (coin up) or the fully recorded (coin down) batch, up to the rounding of a batch that holds other rows."""
    model = tiny_model.eval()
    fa = heads.FrameAveraged(model, average=False).eval()
    batch = next(iter(data.Batches(rows['FIT'], 100, 0, stats=stats, swap_mode='random_view').epoch(0)))
    flag = batch['swap_view']
    assert 0 < int(flag.sum()) < len(flag)
    plain = {k: v for k, v in batch.items() if k != 'swap_view'}
    whole = data.swap(plain)
    with torch.no_grad():
        mixed, recorded, every = (train.score_blocks(fa, fa.encode(b), b) for b in (batch, plain, whole))
    assert set(mixed) == set(recorded) == set(every) and len(mixed) == 13
    seen_up = seen_down = 0
    for name, (head, target, nll) in mixed.items():
        _, kind = loss.split_key(name)
        mask = flag[plain[data.row_key(head, kind, 'pos')]]
        seen_up, seen_down = seen_up + int(mask.sum()), seen_down + int((~mask).sum())
        for field, t in target.items():
            want = torch.where(mask.reshape(-1, *([1] * (t.dim() - 1))), every[name][1][field], recorded[name][1][field])
            assert torch.equal(t, want), (name, field)
        torch.testing.assert_close(nll, torch.where(mask, every[name][2], recorded[name][2]), rtol=1e-5, atol=1e-6)
    assert seen_up > 10 and seen_down > 10                                                          # control: rows of both frames were scored


def test_none_mode_never_swaps(tmp_path, rows, stats):
    """'none' (no swap augmentation): the batches carry no coin, the encoder sees only the recorded frame (one call for each training batch, every side marker 0) and the run records the
    mode; it is another run than the average and than the random swap (control)."""
    data_ = swap_data(rows, stats, 'none')
    assert all('swap_view' not in b for b in data_.fit.epoch(0))
    with pytest.MonkeyPatch.context() as patch:
        log = spy_encodes(patch)
        result = run(tmp_path / 'none', data_, swap_cfg('none'), epochs=2)
    assert len([e for e in log if e[0]]) == 2 * FIT_BATCHES and not any(e[2].any() for e in log if e[0])
    assert result.head['identity']['swap_mode'] == 'none' and 'eval_average' not in result.head['identity']
    other = run(tmp_path / 'average', swap_data(rows, stats, 'average'), make_cfg(), epochs=2)
    assert result.fit_batch_mean_losses != other.fit_batch_mean_losses


def test_average_default_is_bit_identical(tmp_path, rows, stats, base):
    """``swap_mode='average'`` and ``eval_average=True`` are the defaults: set explicitly they give the run of the earlier tasks bit for bit (the weights, the optimizer state, the random
    state, the identity, the head, the bytes of the metric files), and the identity has no key for either."""
    result, folder = base
    again = run(tmp_path / 'explicit', swap_data(rows, stats, 'average'), make_cfg(swap_mode='average', eval_average=True))
    assert_same_run(result, again)
    a, b = last_pt(folder, EPOCHS), last_pt(tmp_path / 'explicit', EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)
    assert list(a['loop']) == list(b['loop'])
    assert again.head['identity'] == checkpoints(folder)[-1][0]['identity'] and 'swap_mode' not in again.head['identity'] and 'eval_average' not in again.head['identity']
    assert again.head['schema'] == 'ept_train_checkpoint_v6'
    for name in ('epoch_metrics.json', 'epoch_metrics.csv', 'best.pt'):                    # last.pt holds the wall-clock timings of the loop, so only its parts are compared above; best.pt exists only in the folder of an epoch that improved
        first, second = folder / f'epoch_{EPOCHS:04d}' / name, tmp_path / 'explicit' / f'epoch_{EPOCHS:04d}' / name
        assert first.exists() == second.exists() and (not first.exists() or first.read_bytes() == second.read_bytes()), name
    assert (folder / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.json').exists()


def test_swap_settings_enter_the_identity_only_when_they_are_not_the_default(tmp_path, rows, stats, base):
    reference, _ = base
    for name, mode, over, keys in (('rv', 'random_view', {}, {'swap_mode': 'random_view'}), ('none', 'none', {}, {'swap_mode': 'none'}),
                                   ('noavg', 'average', {'eval_average': False}, {'eval_average': False}),
                                   ('d17', 'none', {'eval_average': False}, {'swap_mode': 'none', 'eval_average': False})):
        result = run(tmp_path / name, swap_data(rows, stats, mode), make_cfg(swap_mode=mode, **over))
        identity = result.head['identity']
        assert {k: identity[k] for k in keys} == keys and {k: v for k, v in identity.items() if k not in keys} == reference.head['identity'], name       # nothing else moves
        assert result.head['schema'] == 'ept_train_checkpoint_v6'                                  # no new column: the metric files are those of the earlier tasks


@pytest.mark.parametrize('first,second', [({'swap_mode': 'average'}, {'swap_mode': 'random_view'}), ({'swap_mode': 'random_view'}, {'swap_mode': 'average'}),
                                          ({'swap_mode': 'none'}, {'swap_mode': 'average'}), ({'swap_mode': 'average'}, {'swap_mode': 'none'}),
                                          ({'swap_mode': 'random_view'}, {'swap_mode': 'none'}),
                                          ({'eval_average': True}, {'eval_average': False}), ({'eval_average': False}, {'eval_average': True})])
def test_resume_refuses_a_checkpoint_under_another_swap_setting_in_both_directions(tmp_path, rows, stats, first, second):
    folder = tmp_path / 'run'
    name = next(iter(first))
    with pytest.raises(Cut):
        run(folder, swap_data(rows, stats, first.get('swap_mode', 'average')), make_cfg(**first), check=cut_on(3))
    assert len(checkpoints(folder)) == 2
    before, log = file_hashes(folder), []
    with pytest.raises(ValueError, match=name):
        run(folder, swap_data(rows, stats, second.get('swap_mode', 'average'), log=log), make_cfg(**second))
    assert no_work(log) and file_hashes(folder) == before


@pytest.mark.parametrize('settings,made_for', [('random_view', 'average'), ('average', 'random_view'), ('none', 'average'), ('average', 'none'), ('random_view', 'none'), ('none', 'random_view')])
def test_the_batches_and_the_settings_must_agree_on_the_swap_mode_before_any_work(tmp_path, rows, stats, settings, made_for):
    log = []
    with pytest.raises(ValueError, match='swap_mode'):
        run(tmp_path / 'run', swap_data(rows, stats, made_for, log=log), swap_cfg(settings))
    assert no_work(log) and checkpoints(tmp_path / 'run') == []


def test_the_train_config_has_swap_mode_and_eval_average_last_with_the_defaults_and_refuses_another_name_and_a_non_bool():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    assert names[-5:-2] == ['terrain_smooth_weight', 'swap_mode', 'eval_average'] and names[:6] == ['encoder', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'model_spec']
    small = encoder.EncoderConfig(d=16, layers=1, heads=2)
    cfg = train.TrainConfig(small, 1e-3, 0.0, 1)
    assert cfg.swap_mode == 'average' and cfg.eval_average is True
    assert train.TrainConfig(small, 1e-3, 0.0, 1, 2, None, 'poisson', 'raw', 'all', 'none', 'whole_fit', 0.0).swap_mode == 'average'         # the positional calls of the earlier tasks
    assert train.TrainConfig(small, 1e-3, 0.0, 1, swap_mode='random_view', eval_average=False).swap_mode == 'random_view'
    assert data.SWAP_MODES == ('average', 'random_view', 'none')
    for bad in ('Random_view', 'random', 'swap', '', None, 1, True):
        with pytest.raises(ValueError, match='swap_mode'):
            train.TrainConfig(small, 1e-3, 0.0, 1, swap_mode=bad)
    for bad in (None, 0, 1, 'yes', 'False', [True]):
        with pytest.raises(ValueError, match='eval_average'):
            train.TrainConfig(small, 1e-3, 0.0, 1, eval_average=bad)


@pytest.fixture(scope='module')
def swap_run(tmp_path_factory, rows, stats):
    """An uninterrupted run of EPOCHS epochs with the random view swap."""
    folder = tmp_path_factory.mktemp('train_random_view') / 'run'
    return run(folder, swap_data(rows, stats, 'random_view'), swap_cfg('random_view')), folder


def test_a_random_view_run_trains_and_is_another_run_than_the_average(swap_run, base):
    result, _ = swap_run
    reference, _ = base
    assert np.isfinite(result.stop_losses).all() and np.isfinite(result.fit_batch_mean_losses).all() and len(result.stop_losses) == EPOCHS
    assert result.head['identity']['swap_mode'] == 'random_view'
    assert result.fit_batch_mean_losses != reference.fit_batch_mean_losses
    assert not all(torch.equal(a, b) for a, b in zip(result.model.state_dict().values(), reference.model.state_dict().values()))      # the swap changes the training (control)


def test_a_random_view_run_resumes_with_the_same_bits(tmp_path, rows, stats, swap_run):
    """The coins are a function of the seed, the epoch and the batch index, so a run that was cut and resumed saw the same coins as the run that was not."""
    full, full_folder = swap_run
    folder = tmp_path / 'run'
    with pytest.raises(Cut):
        run(folder, swap_data(rows, stats, 'random_view'), swap_cfg('random_view'), check=cut_on(3))
    resumed = run(folder, swap_data(rows, stats, 'random_view'), swap_cfg('random_view'))
    assert_same_run(full, resumed)
    a, b = last_pt(full_folder, EPOCHS), last_pt(folder, EPOCHS)
    for part in ('model', 'optimizer', 'rng'):
        assert_same_state(a[part], b[part], part)


def test_the_stop_value_uses_the_evaluation_path_of_the_settings(tmp_path, rows, stats):
    """STOP value = the evaluation path: the two-frame average after a random-view run (``eval_average`` True), the recorded frame only for the D17 check ('none' with
    ``eval_average=False``); the recorded value is the pooled STOP number of the returned weights computed by the definition with the same wrapper."""
    data_ = swap_data(rows, stats, 'random_view')
    averaged = run(tmp_path / 'rv', data_, swap_cfg('random_view'), epochs=1)
    assert averaged.stop_losses[0] == pytest.approx(reference_stop(averaged.model, data_.stop, SCALES), rel=2e-5)
    d17_data = swap_data(rows, stats, 'none')
    d17 = run(tmp_path / 'd17', d17_data, make_cfg(swap_mode='none', eval_average=False), epochs=1)
    assert d17.stop_losses[0] == pytest.approx(reference_stop(d17.model, d17_data.stop, SCALES, average=False), rel=2e-5)
    assert abs(d17.stop_losses[0] - reference_stop(d17.model, d17_data.stop, SCALES)) > 1e-4 * d17.stop_losses[0]           # control: the average would be another number


def test_evaluate_stop_takes_eval_average_and_the_default_is_the_two_frame_average(tiny_model, rows, stats):
    import inspect
    assert inspect.signature(train.evaluate_stop).parameters['eval_average'].default is True
    batches = data.Batches(rows['STOP'], 100, 0, stats=stats, training=False)
    with pytest.MonkeyPatch.context() as patch:
        log = spy_encodes(patch)
        default = train.evaluate_stop(tiny_model, batches, SCALES)
        assert len(log) == 2 * len(batches) and train.evaluate_stop(tiny_model, batches, SCALES, eval_average=True) == default
        log.clear()
        single = train.evaluate_stop(tiny_model, batches, SCALES, eval_average=False)
        assert len(log) == len(batches) and not any(e[2].any() for e in log)                      # every view once, in the recorded frame
    assert single == pytest.approx(reference_stop(tiny_model, batches, SCALES, average=False), rel=2e-5) and single != default
    with pytest.raises(ValueError, match='eval_average'):
        train.evaluate_stop(tiny_model, batches, SCALES, eval_average=1)


def test_eval_average_true_gives_exact_swap_invariance_of_the_stop_value_and_false_does_not(tiny_model, rows, stats):
    """The invariance of the average is exact in every output (``test_ept_heads``); the STOP value adds the scores over permuted cells, so it is the same number up to the order of those sums.
    One frame alone gives another number for the swapped batch (control: the D17 check measures something)."""
    batch = next(iter(data.Batches(rows['STOP'], 100, 0, stats=stats, training=False).epoch(0)))
    swapped = data.swap(batch)
    a, b = (train.evaluate_stop(tiny_model, Fixed(x), SCALES) for x in (batch, swapped))
    assert a == pytest.approx(b, rel=1e-6)
    one_a, one_b = (train.evaluate_stop(tiny_model, Fixed(x), SCALES, eval_average=False) for x in (batch, swapped))
    assert abs(one_a - one_b) > 1e-4 * a


def test_the_loop_passes_eval_average_to_the_stop_evaluation_only_when_it_is_false(tmp_path, rows, stats, monkeypatch):
    kwargs = []
    monkeypatch.setattr(train, 'evaluate_stop', lambda *a, **k: (kwargs.append(sorted(k)), 2.0)[1])
    run(tmp_path / 'default', make_data(rows, stats), epochs=2)
    run(tmp_path / 'false', make_data(rows, stats), make_cfg(eval_average=False), epochs=2)
    assert kwargs[:2] == [['guard']] * 2 and kwargs[2:] == [['eval_average', 'guard']] * 2           # the default call is the call of every earlier run


def test_a_team_set_run_under_random_view_and_under_the_d17_settings_trains(tmp_path, rows, stats):
    from c_ept.comparison import models as comparison_models
    spec = comparison_models.specification('n2', hidden=8)
    for name, mode, over in (('rv', 'random_view', {}), ('d17', 'none', {'eval_average': False})):
        result = run(tmp_path / name, swap_data(rows, stats, mode), make_cfg(swap_mode=mode, model_spec=spec, **over), epochs=2)
        assert np.isfinite(result.stop_losses).all() and np.isfinite(result.fit_batch_mean_losses).all()
        assert result.head['identity']['model_spec']['architecture'] == 'n2_teamset_v1' and result.head['identity']['swap_mode'] == mode


# =====================================================================================================================================
# S4b preparation M8 (decision D10 P2, P7): the epoch budget and the behaviour at the wall cap
# =====================================================================================================================================
def cap_hook(clock, epoch_index=1):
    """The fake clock moves 90,000 s (more than the 24 h cap) while the second FIT batch of epoch ``epoch_index`` is fetched: no time is spent."""
    def hook(name, epoch, i):
        if (name, epoch, i) == ('fit', epoch_index, 1):
            clock.bump += 90_000.0
    return hook


def test_the_train_config_has_on_wall_cap_last_with_the_default_fail_and_refuses_another_name():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    assert names[-5:] == ['terrain_smooth_weight', 'swap_mode', 'eval_average', 'on_wall_cap', 'zone_mark'] and names[:6] == ['encoder', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'model_spec']       # M9 adds zone_mark after it
    small = encoder.EncoderConfig(d=16, layers=1, heads=2)
    assert train.TrainConfig(small, 1e-3, 0.0, 1).on_wall_cap == 'fail'
    assert train.TrainConfig(small, 1e-3, 0.0, 1, 2, None, 'poisson', 'raw', 'all', 'none', 'whole_fit', 0.0, 'average', True).on_wall_cap == 'fail'      # the positional calls of the earlier tasks
    assert train.TrainConfig(small, 1e-3, 0.0, 1, on_wall_cap='best_stop').on_wall_cap == 'best_stop'
    for bad in ('Best_Stop', 'stop', 'best', 'budget', '', None, 1, True):
        with pytest.raises(ValueError, match='on_wall_cap'):
            train.TrainConfig(small, 1e-3, 0.0, 1, on_wall_cap=bad)


def test_wall_cap_with_best_stop_ends_done_with_budget_reason(tmp_path, rows, stats, monkeypatch, base):
    """The cap is reached while epoch 2 (index 1) trains. Under 'best_stop' that epoch is finished (its FIT batches, its STOP pass, its checkpoint) and the run then ends with done=True
    and the reason 'budget': the best STOP weights are the returned model, the call marker is a normal completed one, and nothing is stored as a failure."""
    clock, log, folder = Clock(monkeypatch), [], tmp_path / 'budget'
    result = run(folder, make_data(rows, stats, log=log, hook=cap_hook(clock)), make_cfg(on_wall_cap='best_stop'))
    assert result.epochs_run == 2 and fit_batches_logged(log) == 2 * FIT_BATCHES and stop_batches_logged(log) == 2           # not between batches: the epoch is finished
    saved = checkpoints(folder)
    assert [head['epoch'] for head, _ in saved] == [1, 2] and 'budget_end' not in saved[0][0]
    head, last = saved[-1]
    assert head['done'] is True and head['stopped_early'] is False and head['budget_end'] is True and head['epochs'] == EPOCHS
    assert result.head['budget_end'] is True and result.head['identity']['on_wall_cap'] == 'best_stop'
    doc = json.loads((last / 'epoch_metrics.json').read_text(encoding='utf-8'))
    assert (doc['termination_reason'], doc['done'], doc['stopped_early'], doc['epoch_count']) == ('budget', True, False, 2)
    assert_same_state(result.model.state_dict(), torch.load(folder / head['best']['folder'] / 'best.pt', weights_only=True))      # the best STOP checkpoint, as it was
    for epoch in (1, 2):                                                                                                       # the budget run is the first two epochs of the default run
        assert_same_state(last_pt(folder, epoch)['model'], last_pt(base[1], epoch)['model'], f'epoch {epoch}')
    marker = call_marker(folder)
    assert (marker['outcome'], marker['status'], marker['epoch_done_end']) == ('completed', 'ok', 2) and marker['seconds'] >= 90_000.0
    assert not list(folder.glob('resource_failure_*.json'))
    before, again = file_hashes(folder), []                                                                                    # a call on the finished run does nothing, over the cap as it is
    repeated = run(folder, make_data(rows, stats, log=again), make_cfg(on_wall_cap='best_stop'))
    assert no_work(again) and file_hashes(folder) == before and repeated.epochs_run == 2 and repeated.head['budget_end'] is True
    assert_same_weights(repeated.model, result.model)


@pytest.mark.parametrize('case', ['early_stopping', 'max_epochs'])
def test_a_natural_end_in_the_epoch_of_the_cap_keeps_its_own_reason(tmp_path, rows, stats, monkeypatch, case):
    """The budget is the end that no other rule reached: when the patience runs out (or the horizon is reached) in the epoch that reaches the cap, the reason is that rule's."""
    clock = Clock(monkeypatch)
    if case == 'early_stopping':
        scripted(monkeypatch, [2.0, 3.0, 4.0, 5.0])
        result = run(tmp_path / case, make_data(rows, stats, hook=cap_hook(clock, 2)), make_cfg(on_wall_cap='best_stop', patience=2))
        assert result.epochs_run == 3 and result.stopped_early
    else:
        result = run(tmp_path / case, make_data(rows, stats, hook=cap_hook(clock, 1)), make_cfg(on_wall_cap='best_stop'), epochs=2)
        assert result.epochs_run == 2 and not result.stopped_early
    head, last = checkpoints(tmp_path / case)[-1]
    doc = json.loads((last / 'epoch_metrics.json').read_text(encoding='utf-8'))
    assert 'budget_end' not in head and head['done'] is True and doc['termination_reason'] == case


def test_best_stop_under_the_cap_is_the_default_run(tmp_path, rows, stats, base):
    """The cap is never reached: the run is the run of the default setting (weights, curves, metrics rows, no budget key), apart from the identity key that names the setting."""
    result = run(tmp_path / 'under', make_data(rows, stats), make_cfg(on_wall_cap='best_stop'))
    assert_same_run(base[0], result)
    head, last = checkpoints(tmp_path / 'under')[-1]
    assert 'budget_end' not in head and head['identity']['on_wall_cap'] == 'best_stop'
    assert json.loads((last / 'epoch_metrics.json').read_text(encoding='utf-8'))['termination_reason'] == 'max_epochs'
    assert (last / 'epoch_metrics.csv').read_bytes() == (base[1] / f'epoch_{EPOCHS:04d}' / 'epoch_metrics.csv').read_bytes()


@pytest.mark.parametrize('where', ['before_the_epochs', 'checkpoint_write', 'unmeasured'])
def test_memory_cap_still_fails_with_best_stop(tmp_path, rows, stats, monkeypatch, where):
    """P7 ends a run at the wall cap only: the memory bound (and a run that could not be measured) stays a failure with its stored record, and a later call is refused."""
    folder, log = tmp_path / where, []
    if where == 'before_the_epochs':
        injected_memory(monkeypatch, lambda: 41 * GIB)
        match = '40 GiB'
    elif where == 'checkpoint_write':
        injected_memory(monkeypatch, lambda: 41 * GIB if (folder / 'epoch_0002' / 'head.json').exists() else GIB)
        match = '40 GiB'
    else:
        monkeypatch.setattr(train, 'MEMORY_INTERVAL_S', 1e6)
        monkeypatch.setattr(train, '_memory_sample', lambda: (0, None, 0, 'unavailable'))
        match = 'measure'
    with pytest.raises(train.ResourceBoundError, match=match):
        run(folder, make_data(rows, stats, log=log), make_cfg(on_wall_cap='best_stop'))
    if where == 'checkpoint_write':
        assert [head['epoch'] for head, _ in checkpoints(folder)] == [1, 2] and 'budget_end' not in checkpoints(folder)[-1][0]
        assert list(folder.glob('resource_failure_*.json'))
        injected_memory(monkeypatch, lambda: GIB)
        again = []
        with pytest.raises(train.ResourceBoundError, match='earlier'):
            run(folder, make_data(rows, stats, log=again), make_cfg(on_wall_cap='best_stop'))
        assert no_work(again)
    else:
        assert no_work(log) and not checkpoints(folder)


def test_an_integrity_error_after_the_cap_is_not_a_budget_end_and_the_run_is_not_continued(tmp_path, rows, stats, monkeypatch):
    """The cap is reached and the STOP value of that epoch is not finite: the error is raised as it always was (no checkpoint, no 'done'), and the next call, which would have to train beyond
    the cap with no clean end behind it, is refused before any work."""
    clock, real, folder = Clock(monkeypatch), train.evaluate_stop, tmp_path / 'broken'
    scripted(monkeypatch, [float('nan')])
    with pytest.raises(FloatingPointError):
        run(folder, make_data(rows, stats, hook=cap_hook(clock, 0)), make_cfg(on_wall_cap='best_stop'))
    assert not checkpoints(folder) and not list(folder.glob('resource_failure_*.json')) and call_marker(folder)['outcome'] == 'interrupted'      # not yet a stored failure: the next call finds the run over the cap
    monkeypatch.setattr(train, 'evaluate_stop', real)
    log = []
    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(folder, make_data(rows, stats, log=log), make_cfg(on_wall_cap='best_stop'))
    assert no_work(log)


def test_fail_default_is_unchanged(tmp_path, rows, stats, monkeypatch, base):
    """The explicit 'fail' is the default: same weights, same checkpoint files (last.pt, the metrics snapshot), no new key in the head or the identity, and the cap is refused as before
    (the epoch in which it is reached is not finished, the failure is stored, a later call is refused)."""
    result = run(tmp_path / 'explicit', make_data(rows, stats), make_cfg(on_wall_cap='fail'))
    assert_same_run(base[0], result)
    for epoch in range(1, EPOCHS + 1):
        a, b = base[1] / f'epoch_{epoch:04d}', tmp_path / 'explicit' / f'epoch_{epoch:04d}'
        for part in ('model', 'optimizer', 'rng'):                                                                             # last.pt also holds the timings of the epochs, which are no bits to compare
            assert_same_state(last_pt(base[1], epoch)[part], last_pt(tmp_path / 'explicit', epoch)[part], f'epoch {epoch} {part}')
        assert (a / 'epoch_metrics.csv').read_bytes() == (b / 'epoch_metrics.csv').read_bytes() and json.loads((a / 'epoch_metrics.json').read_text(encoding='utf-8')) == json.loads((b / 'epoch_metrics.json').read_text(encoding='utf-8'))
    head = checkpoints(tmp_path / 'explicit')[-1][0]
    assert set(head) == set(base[0].head) and head['identity'] == base[0].head['identity'] and 'on_wall_cap' not in head['identity'] and 'budget_end' not in head
    clock, log, folder = Clock(monkeypatch), [], tmp_path / 'cap'
    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(folder, make_data(rows, stats, log=log, hook=cap_hook(clock)), make_cfg(on_wall_cap='fail'))
    assert [head['epoch'] for head, _ in checkpoints(folder)] == [1] and fit_batches_logged(log) == FIT_BATCHES + 2 and stop_batches_logged(log) == 1
    assert list(folder.glob('resource_failure_*.json')) and call_marker(folder)['status'] == 'exceeded'
    again = []
    with pytest.raises(train.ResourceBoundError, match='earlier'):
        run(folder, make_data(rows, stats, log=again), make_cfg(on_wall_cap='fail'))
    assert no_work(again)


def test_on_wall_cap_enters_the_identity_only_when_it_is_not_fail_and_a_resume_under_the_other_setting_is_refused_both_ways(tmp_path, rows, stats, base):
    assert 'on_wall_cap' not in base[0].head['identity']
    for first, second in (('best_stop', 'fail'), ('fail', 'best_stop')):
        folder = tmp_path / first
        with pytest.raises(Cut):
            run(folder, make_data(rows, stats), make_cfg(on_wall_cap=first), check=cut_on(2))
        identity = checkpoints(folder)[-1][0]['identity']
        assert identity.get('on_wall_cap') == ('best_stop' if first == 'best_stop' else None)
        before, log = file_hashes(folder), []
        with pytest.raises(ValueError, match='on_wall_cap'):
            run(folder, make_data(rows, stats, log=log), make_cfg(on_wall_cap=second))
        assert no_work(log) and file_hashes(folder) == before
        resumed = run(folder, make_data(rows, stats), make_cfg(on_wall_cap=first))
        assert resumed.epochs_run == EPOCHS
        assert_same_run(base[0], resumed)


def test_the_cap_boundary_fail_allows_exactly_24_h_and_best_stop_ends_at_it(tmp_path, rows, stats, monkeypatch):
    """M8 review m1: the seconds of the run are exactly the cap (a clock that is constant except for one step in epoch 2), then one tick below, then one tick above. 'fail' allows exactly the
    cap and refuses above it (inclusive, as before); 'best_stop' ends the run at the epoch that reaches the cap (>=) and not one tick below it."""
    cap = train.WALL_CAP_SECONDS

    def fixed_clock(level):
        state = {'t': 0.0}
        monkeypatch.setattr(train, '_clock', lambda: state['t'])

        def hook(name, epoch, i):
            if (name, epoch, i) == ('fit', 1, 1):
                state['t'] = level
        return hook

    for mode, level, epochs_run in (('best_stop', cap, 2), ('best_stop', cap - 1e-6, EPOCHS), ('fail', cap, EPOCHS), ('fail', cap - 1e-6, EPOCHS), ('best_stop', cap + 1e-6, 2)):
        folder = tmp_path / f'{mode}_{level}'
        result = run(folder, make_data(rows, stats, hook=fixed_clock(level)), make_cfg(on_wall_cap=mode))
        assert result.epochs_run == epochs_run, (mode, level)
        assert ('budget_end' in result.head) == (epochs_run != EPOCHS), (mode, level)
    with pytest.raises(train.ResourceBoundError, match='24 h'):
        run(tmp_path / 'fail_above', make_data(rows, stats, hook=fixed_clock(cap + 1e-6)), make_cfg(on_wall_cap='fail'))


def test_the_memory_bound_is_not_masked_by_the_soft_wall_cap(tmp_path, rows, stats, monkeypatch):
    """M8 review m5: the wall cap is passed in epoch 2 and the memory goes over its bound in the same epoch. The soft wall does not hide it: the run fails before the write of epoch 2, and no budget end is written."""
    clock, folder, state = Clock(monkeypatch), tmp_path / 'both', {'high': False}
    injected_memory(monkeypatch, lambda: 41 * GIB if state['high'] else GIB)

    def hook(name, epoch, i):
        if (name, epoch, i) == ('fit', 1, 1):
            clock.bump += 90_000.0
            state['high'] = True
    with pytest.raises(train.ResourceBoundError, match='40 GiB'):
        run(folder, make_data(rows, stats, hook=hook), make_cfg(on_wall_cap='best_stop'))
    assert [head['epoch'] for head, _ in checkpoints(folder)] == [1] and list(folder.glob('resource_failure_*.json'))


# =====================================================================================================================================
# the zone mark switch (S4b preparation M9, decision D9: A7 keeps the zone mark only if the shared M2Rate + M2Mark NLL is non-inferior and the mark gains on V_start CRPS and the masked M3 NLL)
# =====================================================================================================================================
SMALL_CFG = dict(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2)


def zone_models(seed: int = 5):
    """The same starting weights twice: the default EPT and the EPT without the zone mark."""
    cfg = encoder.EncoderConfig(**SMALL_CFG)
    torch.manual_seed(seed)
    on = heads.EPT(cfg, toy_static())
    torch.manual_seed(seed)
    off = heads.EPT(cfg, toy_static(), zone_mark=False)
    return on, off


def test_zone_mark_off_removes_the_m2zone_block_and_z_conditioning(rows, stats, monkeypatch):
    """A7: without the zone mark the M2Zone block is not scored (not in the loss, not in STOP) and M3, the objectives and M4 do not read z. The other twelve blocks are scored as before, the
    shared M2 call reads the views of the mark block only, the zone table and the zone head get no gradient, and the default model trains them."""
    on, off = zone_models()
    batch = next(iter(data.Batches(rows['FIT'], 100, 0, stats=stats, training=False).epoch(0)))
    calls, real = [], heads.FrameAveraged.m2
    monkeypatch.setattr(heads.FrameAveraged, 'm2', lambda self, enc, exposure, pos=None: (calls.append(None if pos is None else pos.clone()), real(self, enc, exposure, pos))[1])
    got = {}
    for name, model, extra in (('on', on, {}), ('off', off, {'zone_mark': False})):
        model.eval()                                                                              # no dropout: the two models see the same numbers
        fa = heads.FrameAveraged(model)
        calls.clear()
        scored = train.score_blocks(fa, fa.encode(batch), batch, **extra)
        got[name] = scored, list(calls)
        total = loss.total_loss({k: v[2] for k, v in scored.items()}, SCALES, {k: v[1] for k, v in scored.items()})
        total.backward()
        zone = {n: p.grad for n, p in model.heads.named_parameters() if n.startswith('zone_net.') or n == 'zone_emb.weight'}
        assert (name == 'on') == all(g is not None and g.abs().sum() > 0 for g in zone.values()), name       # the default trains every zone part; the others get nothing
        assert (name == 'off') == all(g is None for g in zone.values()), name
    (scored_on, calls_on), (scored_off, calls_off) = got['on'], got['off']
    assert 'M2Zone/masked' in scored_on and 'M2Zone/masked' not in scored_off and set(scored_off) == set(scored_on) - {'M2Zone/masked'} and len(scored_off) == 12
    mark_views = data.block(batch, 'M2Mark', 'masked')['pos']
    zone_views = data.block(batch, 'M2Zone', 'masked')['pos']
    assert len(calls_on) == 1 and torch.equal(calls_on[0], torch.unique(torch.cat([mark_views, zone_views])))
    assert len(calls_off) == 1 and torch.equal(calls_off[0], torch.unique(mark_views))
    for key in ('M2Rate/masked', 'M2Mark/masked'):                                           # the shared M2 blocks do not read z: the same bits whichever model
        assert torch.equal(scored_on[key][2].detach(), scored_off[key][2].detach()), key
    assert not torch.equal(scored_on['M3Size/masked'][2].detach(), scored_off['M3Size/masked'][2].detach())      # M3 read z in the default (control)
    # the STOP number leaves the zone block out as well: the number without it is the number with it minus the term of the block
    stop = data.Batches(rows['STOP'], 100, 0, stats=stats, training=False)
    full, without = train.evaluate_stop(on, stop, SCALES), train.evaluate_stop(on, stop, SCALES, zone_mark=False)
    term = train.evaluate_stop_blocks(on, stop, SCALES)['blocks']['M2Zone/masked']['term']
    assert term > 0 and without == pytest.approx(full - term, rel=1e-12) and without < full


def test_zone_mark_on_default_is_bit_identical(tmp_path, rows, stats, base):
    """zone_mark=True given explicitly is the default: the same blocks and losses, the same STOP number, the same run (weights, curves, every checkpoint part, the metric files, the head and the identity,
    which has no zone_mark key), and a run that does not pass the setting at all gives them too."""
    reference, folder = base
    on, _ = zone_models()
    batch = next(iter(data.Batches(rows['FIT'], 100, 0, stats=stats, training=False).epoch(0)))
    fa = heads.FrameAveraged(on.eval()).eval()
    with torch.no_grad():
        enc = fa.encode(batch)
        default, explicit = train.score_blocks(fa, enc, batch), train.score_blocks(fa, enc, batch, zone_mark=True)
    assert list(default) == list(explicit) and all(torch.equal(default[k][2], explicit[k][2]) for k in default)
    stop = data.Batches(rows['STOP'], 100, 0, stats=stats, training=False)
    assert train.evaluate_stop(on, stop, SCALES) == train.evaluate_stop(on, stop, SCALES, zone_mark=True)
    result = run(tmp_path / 'explicit', make_data(rows, stats), make_cfg(zone_mark=True))
    assert_same_run(reference, result)
    for epoch in range(1, EPOCHS + 1):
        a, b = folder / f'epoch_{epoch:04d}', tmp_path / 'explicit' / f'epoch_{epoch:04d}'
        for part in ('model', 'optimizer', 'rng'):                                                    # last.pt also holds the timings of the epochs, which are no bits to compare
            assert_same_state(last_pt(folder, epoch)[part], last_pt(tmp_path / 'explicit', epoch)[part], f'epoch {epoch} {part}')
        for name in ('epoch_metrics.csv', 'epoch_metrics.json'):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
        assert (a / 'best.pt').exists() == (b / 'best.pt').exists() and (not (a / 'best.pt').exists() or (a / 'best.pt').read_bytes() == (b / 'best.pt').read_bytes())
    head = checkpoints(tmp_path / 'explicit')[-1][0]
    assert set(head) == set(reference.head) and head['identity'] == reference.head['identity'] and 'zone_mark' not in head['identity'] and head['schema'] == 'ept_train_checkpoint_v6'
    assert train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1).zone_mark is True


def test_zone_mark_is_a_boolean_setting_that_the_n2_model_cannot_take_off(tmp_path, rows, stats):
    small = encoder.EncoderConfig(d=16, layers=1, heads=2)
    assert train.TrainConfig(small, 1e-3, 0.0, 1, zone_mark=False).zone_mark is False
    for bad in (None, 0, 1, 'no', 'False'):
        with pytest.raises(ValueError, match='zone_mark'):
            train.TrainConfig(small, 1e-3, 0.0, 1, zone_mark=bad)
    from c_ept.comparison import models as comparison_models
    log = []
    with pytest.raises(ValueError, match='zone_mark'):
        run(tmp_path / 'n2', make_data(rows, stats, log=log), make_cfg(zone_mark=False, model_spec=comparison_models.specification('n2', hidden=8)))
    assert no_work(log) and not (tmp_path / 'n2').exists()                                       # refused before any work, before the first marker of the call is written (not by the model registry later)
    ept = comparison_models.build_model(encoder.EncoderConfig(**SMALL_CFG), toy_static(), comparison_models.specification('ept'), zone_mark=False)
    assert isinstance(ept, heads.EPT) and ept.heads.zone_mark is False
    assert comparison_models.build_model(encoder.EncoderConfig(**SMALL_CFG), toy_static(), comparison_models.specification('ept')).heads.zone_mark is True
    with pytest.raises(ValueError, match='zone_mark'):
        comparison_models.build_model(encoder.EncoderConfig(**SMALL_CFG), toy_static(), comparison_models.specification('n2', hidden=8), zone_mark=False)


@pytest.fixture(scope='module')
def zone_off_run(tmp_path_factory, rows, stats):
    folder = tmp_path_factory.mktemp('zone_off') / 'run'
    return run(folder, make_data(rows, stats), make_cfg(zone_mark=False)), folder


def test_a_run_without_the_zone_mark_trains_no_zone_part_and_scores_no_zone_block(zone_off_run, base, rows, stats):
    """A7 run: the identity says zone_mark False and nothing else moves; no new column or schema; the zone table, the zone head's maps and its starting bias layer weights are what the seed made
    (they get no gradient and no decay), while the default run moved them; the recorded STOP numbers are the ones of the twelve blocks."""
    result, folder = zone_off_run
    reference, ref_folder = base
    identity = result.head['identity']
    assert identity['zone_mark'] is False and {k: v for k, v in identity.items() if k != 'zone_mark'} == reference.head['identity']
    assert result.head['schema'] == 'ept_train_checkpoint_v6' and set(result.head) == set(reference.head)
    last = EPOCHS
    assert (folder / f'epoch_{last:04d}' / 'epoch_metrics.csv').read_bytes().split(b'\n')[0] == (ref_folder / f'epoch_{last:04d}' / 'epoch_metrics.csv').read_bytes().split(b'\n')[0]     # the same columns
    train.deterministic_cpu(seed=SEED)
    initial = heads.EPT(encoder.EncoderConfig(**SMALL_CFG), toy_static()).state_dict()                  # the model the run started from, before the start biases
    moved = lambda model, key: not torch.equal(model.state_dict()[key], initial[key])                   # noqa: E731
    zone_keys = [k for k in initial if k.startswith('heads.zone_net.') and k != 'heads.zone_net.out.bias'] + ['heads.zone_emb.weight']
    assert zone_keys and not any(moved(result.model, k) for k in zone_keys)
    assert all(moved(reference.model, k) for k in zone_keys)                                          # control: the default run trains them
    assert all(math.isfinite(v) for v in result.stop_losses) and result.stop_losses != reference.stop_losses
    stop = make_data(rows, stats).stop
    report = train.evaluate_stop_blocks(result.model, stop, SCALES, zone_mark=False)
    assert 'M2Zone/masked' not in report['blocks'] and len(report['blocks']) == 12
    assert report['loss_stop'] == pytest.approx(result.stop_losses[result.best_epoch], rel=1e-9) and report['loss_stop'] == train.evaluate_stop(result.model, stop, SCALES, zone_mark=False)


def test_zone_mark_enters_the_identity_only_when_it_is_off_and_a_resume_under_the_other_setting_is_refused_both_ways(tmp_path, rows, stats, base, zone_off_run):
    reference, _ = base
    assert 'zone_mark' not in reference.head['identity'] and zone_off_run[0].head['identity']['zone_mark'] is False
    for first, second in ((False, True), (True, False)):
        folder = tmp_path / f'first_{first}'
        with pytest.raises(Cut):
            run(folder, make_data(rows, stats), make_cfg(zone_mark=first), check=cut_on(2))
        identity = checkpoints(folder)[-1][0]['identity']
        assert identity.get('zone_mark', True) is first
        before, log = file_hashes(folder), []
        with pytest.raises(ValueError, match='zone_mark'):
            run(folder, make_data(rows, stats, log=log), make_cfg(zone_mark=second))
        assert no_work(log) and file_hashes(folder) == before
        resumed = run(folder, make_data(rows, stats), make_cfg(zone_mark=first))
        assert resumed.epochs_run == EPOCHS
        assert_same_run(zone_off_run[0] if first is False else reference, resumed)


def test_shared_m2_blocks_are_reported_separately(rows, stats):
    """D9 judges the zone mark on the M2Rate and M2Mark blocks that the zone term does not touch. ``evaluate_stop_blocks`` returns the STOP value of every block separately (the mean L_h in nat of
    (TRN-2), its B0 scale and the term L_h / scale that Loss_STOP adds), the blocks add up to the STOP value of ``evaluate_stop`` bit for bit, ``shared_m2_blocks`` picks the two blocks, and the
    values of the two blocks are those of the independent reference. The report is a call on the finished model: no column of the epoch record and no schema changes."""
    on, off = zone_models()
    stop = data.Batches(rows['STOP'], 5, 0, stats=stats, training=False)
    report = train.evaluate_stop_blocks(on, stop, SCALES)
    assert set(report) == {'loss_stop', 'blocks'} and report['loss_stop'] == train.evaluate_stop(on, stop, SCALES)
    blocks = report['blocks']
    assert list(blocks) == [k for k in loss.BLOCKS if k in blocks] and {'M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked'} <= set(blocks)
    total = None
    for key, entry in blocks.items():
        assert set(entry) == {'mean', 'scale', 'term'} and entry['scale'] == SCALES[key] and entry['term'] == entry['mean'] / SCALES[key] and entry['mean'] > 0, key
        total = entry['term'] if total is None else total + entry['term']
    assert total == report['loss_stop']
    shared = train.shared_m2_blocks(report)
    assert set(shared) == {'M2Rate/masked', 'M2Mark/masked'} and shared['M2Rate/masked'] == blocks['M2Rate/masked'] and shared['M2Mark/masked'] == blocks['M2Mark/masked']
    assert shared['M2Rate/masked'] != shared['M2Mark/masked']                                   # two separate values, not a sum
    for key in ('M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked'):                              # the independent reference with every other block scaled away
        only = {k: (1.0 if k == key else 1e300) for k in loss.BLOCKS}
        assert blocks[key]['mean'] == pytest.approx(reference_stop(on, stop, only), rel=2e-5), key
    # the zone mark does not change the shared blocks of a model (same weights), and without the zone block the report has twelve
    without = train.evaluate_stop_blocks(off, stop, SCALES, zone_mark=False)
    assert 'M2Zone/masked' not in without['blocks'] and without['blocks']['M2Rate/masked'] == blocks['M2Rate/masked'] and without['blocks']['M2Mark/masked'] == blocks['M2Mark/masked']
    assert train.shared_m2_blocks(without) == shared
    with pytest.raises(ValueError, match='M2Mark'):
        train.shared_m2_blocks({'loss_stop': 1.0, 'blocks': {'M2Rate/masked': blocks['M2Rate/masked']}})


def test_the_stop_evaluation_refuses_a_model_that_declares_no_zone_mark_when_asked_to_score_the_zone_block(rows, stats):
    """M9 review m2: the model without the zone mark has an untrained M2Zone head; scoring it with the default zone_mark=True would add a meaningless block to the STOP value. Refused (for both
    functions); the other direction (a model with the zone mark, zone_mark=False) is the D9 comparison and stays legal."""
    on, off = zone_models()
    stop = data.Batches(rows['STOP'], 100, 0, stats=stats, training=False)
    for call in (train.evaluate_stop, train.evaluate_stop_blocks):
        with pytest.raises(ValueError, match='zone_mark'):
            call(off, stop, SCALES)
        assert call(off, stop, SCALES, zone_mark=False) is not None and call(on, stop, SCALES, zone_mark=False) is not None
