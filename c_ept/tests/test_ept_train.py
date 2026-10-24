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

from c_ept.ept import data, encoder, heads, loss, train
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

    def __init__(self, rows, batch_size, seed, *, stats, training, name, log, hook=None):
        super().__init__(rows, batch_size, seed, stats=stats, training=training)
        self.name, self.log, self.hook = name, log, hook

    def epoch(self, epoch, start=0):
        self.log.append(('epoch', self.name, epoch, start))
        for i, batch in enumerate(super().epoch(epoch, start), start):
            if self.hook is not None:
                self.hook(self.name, epoch, i)
            self.log.append(('batch', self.name))
            yield batch


def make_data(rows, stats, *, log=None, stop_rows=None, static=None, scales=None, fit_batches=FIT_BATCHES, static_source=None, stop_size=None, hook=None):
    """A ``TrainData``: FIT in ``fit_batches`` shuffled batches, STOP in one batch (or ``stop_size`` engagements per batch)."""
    log = [] if log is None else log
    fit_rows, stop_rows = rows['FIT'], rows['STOP'] if stop_rows is None else stop_rows
    size = max(1, math.ceil(len(fit_rows.units) / fit_batches))
    fit = SpyBatches(fit_rows, size, 7, stats=stats, training=True, name='fit', log=log, hook=hook)
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
    from c_ept import head_rows
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


def reference_stop(model, batches, scales) -> float:
    """The STOP number by the definition (R-S4a-16), from one pass over the entries: the scored entries of an evaluation row ``(match, row)`` of a block are averaged,
    each evaluation row is weighted 1 / n (n: the evaluation rows of the match in that block and size stratum), the weighted mean is taken in each stratum, the strata
    are averaged, divided by the scale and the blocks added."""
    model.eval()
    fa = heads.FrameAveraged(model)
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
        assert {p.name for p in path.iterdir()} == set(meta['files']) | {'head.json'}
        for name, entry in meta['files'].items():
            raw = (path / name).read_bytes()
            assert entry['sha256'] == hashlib.sha256(raw).hexdigest() and entry['bytes'] == len(raw)
    meta, path = checkpoints(folder)[-1]
    assert not list(Path(folder).rglob('*.tmp'))                                  # every temporary name was renamed
    assert meta['done'] and meta['schema'] == train.SCHEMA and result.head == meta and Path(result.checkpoint_dir) == path


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
    from c_ept.store_core import Store
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
    from c_ept import store_core
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
    from c_ept.head_rows import HeadScores
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
