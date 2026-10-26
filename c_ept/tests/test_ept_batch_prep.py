"""Tests of the CPU side of a training step (``c_ept/ept/batch_prep.py``; EPT-T012 S4b speed-ups SPD, author decision D32).

Item 1 (this block): the frame of every view is applied on the CPU, before the batch is moved to the device, and not inside ``FrameAveraged.encode`` on the device. What is pinned,
by behaviour:

* ``prepare_batch`` makes ``data.swap_views`` of a batch that carries coins and consumes ``swap_view``; a batch without coins (the modes 'average' and 'none', every STOP batch) is
  returned as it is;
* the tensors that reach the encoder and the targets that ``score_blocks`` scores are the same bits whether the frame is applied inside ``encode`` (the earlier way, still available for
  a batch that carries coins) or by ``prepare_batch`` before it;
* the training loop does the latter: ``swap_views`` is called by the loop before ``encode`` and ``encode`` never sees a coin.

Item 3 (the block "the data checks once per batch" below): ``Preparation`` makes on the CPU, once per batch, the data checks that the heads and the loss made on the device in every step, with the very
exception and message that the device checks raised (a differential test over every field of every block and many ways to be wrong), and the index lists that replace the boolean masks of the loss; the
step that reads such a batch runs trusted (``step_guard``) and gives the loss and every gradient of the untrusted step bit for bit, reads back from the device a handful of times instead of hundreds, and
refuses a model that breaks a check before backward with the message and the count of the earlier check.

The model is tiny (d = 16) and the world is the toy EPT world of ``ept_toys`` on the CPU. No real store, no GPU and no cohort module is read.
"""
# Ref: docs/superpowers/plans/2026-10-26-c-ept-s4b-prep.md Task M7 (the random view swap, the coins of SeedSequence([seed, epoch, batch, 0x5157])) and the audit
#      perf-audit.md items 1 and 3 (the swap on the CPU: loss and every gradient equal; the data checks once per batch on the CPU, the model checks as one flag).
from __future__ import annotations

import functools
import gc
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from c_ept.ept import batch_prep, data, encoder, heads, loss, step_guard, train, training_step
from c_ept.tests.ept_toys import build_ept_world, toy_static

SEED = 3
FIT_BATCHES = 3
SCALES = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}


@pytest.fixture(autouse=True)
def restore_torch_state():
    """``train`` changes process-wide settings (threads, deterministic algorithms, the global random streams); give them back after each test."""
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp('batch_prep_world')
    w = build_ept_world(folder, n_events=7, n_edges=5, consistent_first_kill=True)
    yield w
    del w
    gc.collect()                                      # nothing is deleted here: pytest keeps and rotates its own temporary folders


@pytest.fixture(scope='module')
def stats(world):
    return data.fit_stats(world.s1, world.s2)


@pytest.fixture(scope='module')
def rows(world):
    """``EngagementRows`` of the roles that the tests use (read only)."""
    return {role: data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role)) for role in ('FIT', 'STOP')}


def make_cfg(**over):
    fields = dict(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), lr=3e-3, weight_decay=0.01, warmup_steps=2, patience=10)
    fields.update(over)
    return train.TrainConfig(**fields)


def make_data(rows, stats, swap_mode='average'):
    size = max(1, math.ceil(len(rows['FIT'].units) / FIT_BATCHES))
    fit = data.Batches(rows['FIT'], size, 7, stats=stats, training=True, swap_mode=swap_mode)
    stop = data.Batches(rows['STOP'], len(rows['STOP'].units), 7, stats=stats, training=False)
    return train.TrainData(fit=fit, stop=stop, scales=dict(SCALES), static=toy_static())


def run(folder, data_, cfg, *, epochs=2):
    return train.train(cfg, data_, Path(folder), device='cpu', epochs=epochs, check=lambda: None, seed=SEED)


def a_batch_with_coins(rows, stats, epoch=0, index=0):
    """A FIT batch of the mode 'random_view' with some coins up and some down (so a wrong frame is seen), as ``Batches`` makes it."""
    batches = data.Batches(rows['FIT'], 100, 0, stats=stats, swap_mode='random_view')
    batch = next(iter(batches.epoch(epoch)))
    flag = batch['swap_view']
    assert 0 < int(flag.sum()) < len(flag)
    return batch


def same_tensors(a, b, where=''):
    assert list(a) == list(b), where
    for key in a:
        assert a[key].dtype == b[key].dtype and tuple(a[key].shape) == tuple(b[key].shape) and torch.equal(a[key], b[key]), (where, key)


# ---- item 1: the frame on the CPU ---------------------------------------------------------------------------------------------------------------
def test_prepare_batch_makes_the_frames_of_the_coins_and_consumes_swap_view(rows, stats):
    batch = a_batch_with_coins(rows, stats)
    before = {k: v.clone() for k, v in batch.items()}
    prepared = batch_prep.prepare_batch(batch)
    assert 'swap_view' not in prepared
    same_tensors(prepared, data.swap_views(batch), 'prepared')                       # the very frames of ``swap_views``
    same_tensors(batch, before, 'input')                                             # the input is not changed
    assert any(not torch.equal(prepared[k], before[k]) for k in prepared)           # control: some tensor really was swapped


@pytest.mark.parametrize('mode', ['average', 'none'])
def test_a_batch_without_coins_is_returned_as_it_is(rows, stats, mode):
    batch = next(iter(data.Batches(rows['FIT'], 100, 0, stats=stats, swap_mode=mode).epoch(0)))
    assert 'swap_view' not in batch
    assert batch_prep.prepare_batch(batch) is batch
    stop = next(iter(data.Batches(rows['STOP'], 100, 0, stats=stats, training=False).epoch(0)))
    assert batch_prep.prepare_batch(stop) is stop


def test_a_prepared_batch_prepared_again_is_the_same_batch(rows, stats):
    prepared = batch_prep.prepare_batch(a_batch_with_coins(rows, stats))
    assert batch_prep.prepare_batch(prepared) is prepared


@pytest.mark.parametrize('damage,word', [('not_bool', 'swap_view'), ('wrong_length', 'swap_view'), ('two_dim', 'swap_view')])
def test_a_malformed_swap_view_is_refused_with_the_message_of_swap_views(rows, stats, damage, word):
    batch = a_batch_with_coins(rows, stats)
    flag = batch['swap_view']
    batch['swap_view'] = {'not_bool': flag.long(), 'wrong_length': flag[:-1], 'two_dim': flag[:, None]}[damage]
    with pytest.raises(ValueError, match=word) as raised:
        data.swap_views(batch)
    with pytest.raises(ValueError, match=word) as ours:
        batch_prep.prepare_batch(batch)
    assert str(ours.value) == str(raised.value)


def test_the_frame_before_the_encoder_gives_the_same_inputs_targets_loss_and_gradients_as_the_frame_inside_it(rows, stats):
    """The same coins, the same rows swapped: the tensors that reach ``EPT.encode`` and the targets and row scores of the 13 blocks are the same bits, and so are the loss and every gradient
    of a training step (dropout on, the same seed for the dropout stream)."""
    batch = a_batch_with_coins(rows, stats)
    inputs = {}

    def step(prepared_first: bool):
        torch.manual_seed(5)
        model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), toy_static())
        model.train()
        fa = heads.FrameAveraged(model, average=False)
        seen = []
        real = heads.EPT.encode

        def spy(self, b):
            seen.append({k: v.clone() for k, v in b.items()})
            return real(self, b)

        heads.EPT.encode = spy
        try:
            given = batch_prep.prepare_batch(batch) if prepared_first else batch
            assert ('swap_view' not in given) == prepared_first                       # the frame was applied before the encoder, or is applied inside it
            torch.manual_seed(11)                                                     # the dropout stream
            enc = fa.encode(given)
            scored = train.score_blocks(fa, enc, given)
            total = loss.total_loss({k: v[2] for k, v in scored.items()}, SCALES, {k: v[1] for k, v in scored.items()})
            total.backward()
        finally:
            heads.EPT.encode = real
        inputs[prepared_first] = seen
        grads = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
        return total.detach().clone(), grads, {k: ({f: t.clone() for f, t in v[1].items()}, v[2].detach().clone()) for k, v in scored.items()}

    inside, outside = step(False), step(True)
    assert torch.equal(inside[0], outside[0]) and torch.isfinite(inside[0])
    assert len(inputs[False]) == len(inputs[True]) == 1
    same_tensors(inputs[False][0], inputs[True][0], 'the tensors that reach the encoder')
    assert list(inside[1]) == list(outside[1])
    for name in inside[1]:
        assert (inside[1][name] is None) == (outside[1][name] is None) and (inside[1][name] is None or torch.equal(inside[1][name], outside[1][name])), name
    assert list(inside[2]) == list(outside[2])
    for key in inside[2]:
        same_tensors(inside[2][key][0], outside[2][key][0], key)
        assert torch.equal(inside[2][key][1], outside[2][key][1]), key


def test_the_training_loop_frames_the_batch_before_the_encoder_and_the_encoder_never_sees_a_coin(tmp_path, rows, stats, monkeypatch):
    """The order of the calls of one training run in the mode 'random_view': ``data.swap_views`` (by the loop, on the CPU) comes first, then ``FrameAveraged.encode``, which gets a batch
    without ``swap_view``; the STOP evaluation (the average over both frames) does not call ``swap_views`` at all."""
    log = []
    real_swap_views, real_encode = data.swap_views, heads.FrameAveraged.encode

    def swap_views(batch):
        log.append(('swap_views', torch.is_grad_enabled()))
        return real_swap_views(batch)

    def encode(self, batch):
        log.append(('encode', 'swap_view' in batch, self.average, self.model.training))
        return real_encode(self, batch)

    monkeypatch.setattr(data, 'swap_views', swap_views)
    monkeypatch.setattr(heads.FrameAveraged, 'encode', encode)
    result = run(tmp_path / 'rv', make_data(rows, stats, 'random_view'), make_cfg(swap_mode='random_view'))
    training = [e for e in log if e[0] == 'swap_views' or (e[0] == 'encode' and e[3])]
    assert len(training) == 2 * result.epochs_run * FIT_BATCHES
    assert training[0::2] == [('swap_views', True)] * (result.epochs_run * FIT_BATCHES)                 # the loop frames first ...
    assert training[1::2] == [('encode', False, False, True)] * (result.epochs_run * FIT_BATCHES)          # ... then the encoder runs once, on a batch without coins
    assert all(e[0] == 'encode' and e[2] for e in log if e not in training)                                # the rest is the STOP evaluation: the two-frame average, no swap_views call


@pytest.mark.parametrize('mode', ['average', 'none'])
def test_the_modes_without_coins_call_swap_views_nowhere(tmp_path, rows, stats, monkeypatch, mode):
    calls = []
    real = data.swap_views
    monkeypatch.setattr(data, 'swap_views', lambda batch: (calls.append(1), real(batch))[1])
    run(tmp_path / mode, make_data(rows, stats, mode), make_cfg(swap_mode=mode), epochs=1)
    assert calls == []


# ---- item 3: the data checks once per batch, on the CPU (SPD decision D32) -------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def shifted_rows(world):
    return {role: data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role, kill_law='shifted')) for role in ('FIT', 'STOP')}


def a_model(seed=17):
    torch.manual_seed(seed)
    return heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), toy_static())


def a_batch(rows_by_role, stats, mode='random_view', size=100, index=0):
    """The batch ``index`` of epoch 0 of the FIT rows cut into batches of ``size`` engagements (the default: all of them in one batch)."""
    return next(itertools.islice(data.Batches(rows_by_role['FIT'], size, 0, stats=stats, swap_mode=mode).epoch(0), index, None))


def law_args(kill_law, zone_mark):
    return {**({} if kill_law == 'poisson' else {'kill_law': kill_law}), **({} if zone_mark else {'zone_mark': False})}


def legacy_error(model, batch, kill_law='poisson', zone_mark=True):
    """What the step of every earlier run made of ``batch``: the encoder, ``score_blocks`` and ``total_loss`` on the untrusted path, with the checks of the heads and of the loss on the device; the
    exception that it raised, or None."""
    fa = heads.FrameAveraged(model, average=False).eval()
    try:
        with torch.no_grad():
            framed = batch_prep.prepare_batch(batch)
            scored = train.score_blocks(fa, fa.encode(framed), framed, **law_args(kill_law, zone_mark))
            loss.total_loss({k: v[2] for k, v in scored.items()}, {k: 1.0 for k in loss.BLOCKS}, {k: v[1] for k, v in scored.items()})
    except Exception as err:                                                        # noqa: BLE001 - the type and the message are what is compared
        return err
    return None


def preparation_error(batch, kill_law='poisson', zone_mark=True):
    try:
        batch_prep.Preparation(kill_law=kill_law, zone_mark=zone_mark)(batch)
    except Exception as err:                                                        # noqa: BLE001
        return err
    return None


def same_error(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and type(a) is type(b) and str(a) == str(b))


def test_a_preparation_makes_a_prepared_batch_with_the_plan_and_the_settings_it_was_made_for(rows, stats):
    prep = batch_prep.Preparation()
    batch = a_batch_with_coins(rows, stats)
    prepared = prep(batch)
    assert isinstance(prepared, batch_prep.PreparedBatch) and isinstance(prepared, dict) and 'swap_view' not in prepared
    assert prepared.settings == {'kill_law': 'poisson', 'zone_mark': True} == prep.settings
    plan_keys = [k for k in prepared if k.startswith('plan/')]
    assert plan_keys and all(prepared[k].dtype == torch.int64 and prepared[k].ndim == 1 for k in plan_keys)
    framed = batch_prep.prepare_batch(batch)
    assert set(prepared) - set(plan_keys) == set(framed) and all(torch.equal(prepared[k], framed[k]) for k in framed)      # nothing but the plan is added to the framed batch
    assert prep(prepared) is prepared                                             # prepared once, prepared for good
    assert 'swap_view' in batch and all(not k.startswith('plan/') for k in batch)  # the input is not changed
    with pytest.raises(ValueError, match='settings'):
        batch_prep.Preparation(kill_law='shifted')(prepared)                      # a producer that prepared for other settings is refused, not trusted
    with pytest.raises(ValueError, match='settings'):
        batch_prep.Preparation(zone_mark=False)(prepared)
    for bad in ({'kill_law': 'negative_binomial'}, {'zone_mark': 1}, {'zone_mark': None}):
        with pytest.raises(ValueError):
            batch_prep.Preparation(**bad)


def test_the_shared_m2_call_of_the_plan_is_the_unique_views_of_the_rows_of_the_blocks_that_are_scored(rows, stats):
    batch = a_batch(rows, stats, 'average')
    with_zone, without = batch_prep.Preparation(zone_mark=True)(batch), batch_prep.Preparation(zone_mark=False)(batch)
    both = torch.unique(torch.cat([batch['M2Mark/masked/pos'], batch['M2Zone/masked/pos']]), return_inverse=True)
    only_mark = torch.unique(batch['M2Mark/masked/pos'], return_inverse=True)
    assert torch.equal(with_zone[batch_prep.m2_key(True, 'views')], both[0]) and torch.equal(with_zone[batch_prep.m2_key(True, 'inverse')], both[1])
    assert torch.equal(without[batch_prep.m2_key(False, 'views')], only_mark[0]) and torch.equal(without[batch_prep.m2_key(False, 'inverse')], only_mark[1])
    assert batch_prep.m2_key(True, 'views') not in without and batch_prep.m2_key(False, 'views') not in with_zone      # the key names the setting: a plan for the other setting is never read
    assert loss.plan_key('M2Zone', 'masked', 'counted') in with_zone and loss.plan_key('M2Zone', 'masked', 'counted') not in without


@pytest.mark.parametrize('law', ['poisson', 'shifted'])
def test_the_plan_lists_are_the_masks_of_head_losses_in_the_order_of_the_rows(rows, shifted_rows, stats, law):
    """For each of the 13 blocks the list of the rows that count is ``nonzero`` of ``scored_rows`` and, for the three experts, that of each size is the mask ``counted & (s == k)``: the rows that
    ``head_losses`` selects with its boolean masks, in the same order."""
    batch = a_batch(rows if law == 'poisson' else shifted_rows, stats)
    prepared = batch_prep.Preparation(kill_law=law)(batch)
    seen = 0
    for head, kind in data.ROW_FIELDS:
        block = data.block(prepared, head, kind)
        if len(block['y']) == 0:
            continue
        counted = loss.scored_rows(head, block)
        assert torch.equal(prepared[loss.plan_key(head, kind, 'counted')], counted.nonzero()[:, 0]), (head, kind)
        if head in loss.SIZE_NORMALISED:
            for k in range(4):
                assert torch.equal(prepared[loss.plan_key(head, kind, f'size{k}')], (counted & (block['s'] == k)).nonzero()[:, 0]), (head, kind, k)
        seen += 1
    assert seen == 13


def corruptions(key: str, t: torch.Tensor):
    """``[(name, corrupted tensor)]`` of one field of a block: the values and the types that a check of the heads or of the loss refuses (and, for the fields that no check reads, a control)."""
    head, kind, field = key.split('/')
    if t.numel() == 0:
        return []
    if field in ('match', 'row', 'unit'):
        return [('control', t.clone().fill_(-5))]                                 # no check reads these: the legacy path and the preparation both accept
    out = []
    if t.dtype == torch.int64 and t.ndim == 1:
        for name, value in (('high', 10_000), ('low', -7)):
            c = t.clone()
            c[len(c) // 2] = value
            out.append((name, c))
        out += [('float', t.to(torch.float32)), ('uint8', t.clamp(0, 5).to(torch.uint8)), ('short', t[:-1].clone())]
    elif t.dtype == torch.int64:                                                  # the counts y (n, cells) of the Poisson heads
        for name, value in (('negative', -1), ('big', 10 ** 6)):
            c = t.clone()
            c.view(-1)[0] = value
            out.append((name, c))
        out += [('fraction', t.to(torch.float64) + 0.5), ('bool', t.ne(0)), ('short', t[:-1].clone())]
    else:                                                                         # K, O, e: floats
        for name, value in (('negative', -1.0), ('nan', float('nan')), ('inf', float('inf'))):
            c = t.clone()
            c.view(-1)[0] = value
            out.append((name, c))
        out += [('short', t[:-1].clone())]
    return out


@pytest.mark.parametrize('law,mode', [('poisson', 'average'), ('shifted', 'random_view')])
def test_the_data_checks_on_the_cpu_refuse_what_the_device_checks_refused_with_the_same_exception_and_message(rows, shifted_rows, stats, law, mode):
    """The differential test of fail-closed: every field of every block, many ways to be wrong; the step of every earlier run (the untrusted device checks of the heads and of the loss) and ``Preparation``
    (the same functions on the CPU) raise the same type of exception with the same message, or neither raises."""
    model = a_model().eval()
    batch = a_batch(rows if law == 'poisson' else shifted_rows, stats, mode)
    assert legacy_error(model, batch, law) is None and preparation_error(batch, law) is None          # control: the clean batch passes both
    mismatches, raised, cases = [], 0, 0
    for key in sorted(k for k in batch if k.count('/') == 2):
        for name, corrupt in corruptions(key, batch[key]):
            case = dict(batch)
            case[key] = corrupt
            old, new = legacy_error(model, case, law), preparation_error(case, law)
            cases, raised = cases + 1, raised + (old is not None)
            if not same_error(old, new):
                mismatches.append((key, name, repr(old)[:100], repr(new)[:100]))
    assert not mismatches, mismatches[:8]
    assert cases > 150 and raised > 100, (cases, raised)                          # control: the table does hit the checks (a table that raised nowhere would pass for any code)


def test_a_batch_whose_rows_read_a_view_that_is_not_there_is_refused_like_before(rows, stats):
    model = a_model().eval()
    broken = dict(a_batch(rows, stats, 'average'))
    broken.update({k: broken[k][:3] for k in data.VIEW_KEYS})                      # the rows now read views that the batch does not hold
    old, new = legacy_error(model, broken), preparation_error(broken)
    assert old is not None and same_error(old, new)


def trusted_error(model, batch):
    """What a TRUSTED score of ``batch`` raises (the data checks of the heads and of the loss are not read back from the device), or None."""
    fa = heads.FrameAveraged(model, average=False).eval()
    try:
        with torch.no_grad(), step_guard.trust(step_guard.StepGuards()):
            framed = batch_prep.prepare_batch(batch)
            train.score_blocks(fa, fa.encode(framed), framed)
    except Exception as err:                                                        # noqa: BLE001
        return err
    return None


@pytest.mark.parametrize('key,value', [('M3Obj/masked/K', -1.0), ('M4Conv/masked/K', -1.0), ('M4Conv/masked/O', float('inf')), ('M2Rate/masked/e', -1.0), ('M4Conv/masked/e', -1.0)])
def test_a_trusted_score_does_not_read_back_the_data_checks_of_the_heads_that_an_untrusted_one_makes(rows, stats, key, value):
    model = a_model().eval()
    batch = dict(a_batch(rows, stats, 'average'))
    assert batch[key].numel() > 0 and legacy_error(model, batch) is None
    batch[key] = torch.full_like(batch[key], value)
    old = legacy_error(model, batch)
    assert isinstance(old, ValueError) and 'finite and not negative' in str(old)       # the check of the heads (or of the loss) refuses it on the device ...
    assert trusted_error(model, batch) is None                                         # ... and a trusted step, whose batch passed the same check on the CPU, does not read it back
    assert isinstance(preparation_error(batch), ValueError) and 'finite and not negative' in str(preparation_error(batch))


def noop_optimizer():
    return SimpleNamespace(step=lambda: None)


def a_step(model, batch, *, prepared: bool, kill_law='poisson', zone_mark=True, average=False, seed=11):
    """One ``optimize_batch`` from the same weights and the same dropout stream: the way of every earlier run (the raw batch, the checks on the device, untrusted) or the way of ``train._run`` now (a prepared
    batch, trusted, the model checks deferred and read once); the loss and every gradient."""
    model.zero_grad(set_to_none=True)
    fa = heads.FrameAveraged(model, average=average)
    law = law_args(kill_law, zone_mark)
    score = functools.partial(train.score_blocks, **law) if law else train.score_blocks
    components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
    torch.manual_seed(seed)
    if prepared:
        guards = step_guard.StepGuards()
        with step_guard.trust(guards):
            total = training_step.optimize_batch(fa, batch_prep.Preparation(kill_law=kill_law, zone_mark=zone_mark)(batch), noop_optimizer(), SCALES, 0, components, score=score, clock=lambda: 0.,
                                                 guards=guards)
    else:
        total = training_step.optimize_batch(fa, batch, noop_optimizer(), SCALES, 0, components, score=score, clock=lambda: 0.)
    return total.detach().clone(), {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize('law,mode,average,zone', [('poisson', 'random_view', False, True), ('shifted', 'random_view', False, True), ('poisson', 'none', False, False),
                                                   ('shifted', 'average', True, True), ('poisson', 'average', True, False)])
def test_a_trusted_step_from_a_prepared_batch_gives_the_loss_and_every_gradient_of_the_untrusted_step_bit_for_bit(rows, shifted_rows, stats, law, mode, average, zone):
    model = a_model()
    model.train()
    for index in range(3):                                                        # three different batches: other coins, other rows
        batch = a_batch(rows if law == 'poisson' else shifted_rows, stats, mode, size=9, index=index)
        old = a_step(model, batch, prepared=False, kill_law=law, zone_mark=zone, average=average)
        new = a_step(model, batch, prepared=True, kill_law=law, zone_mark=zone, average=average)
        assert torch.isfinite(old[0]) and torch.equal(old[0], new[0]), (index, old[0], new[0])
        assert list(old[1]) == list(new[1])
        for name in old[1]:
            assert (old[1][name] is None) == (new[1][name] is None) and (old[1][name] is None or torch.equal(old[1][name], new[1][name])), (index, name)


def test_a_model_that_breaks_a_check_is_refused_before_backward_with_the_message_and_the_count_of_the_device_check(rows, stats, monkeypatch):
    """The model check of the rates (a mean that is not positive and finite in an observed cell): the old step raised in the forward, the new one raises at the single read before backward, with the very
    message and count; no gradient is made and no optimizer step is taken."""
    model = a_model()
    model.train()
    batch = a_batch(rows, stats, 'average')
    real = heads.EPTHeads.m2_rate
    monkeypatch.setattr(heads.EPTHeads, 'm2_rate', lambda self, enc, exposure, pos=None: -real(self, enc, exposure, pos))        # every mean is negative
    seen = int((batch['M2Rate/masked/e'] > 0).sum())
    assert seen > 0
    for prepared in (False, True):
        steps = []
        model.zero_grad(set_to_none=True)
        fa = heads.FrameAveraged(model, average=True)
        components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
        optimizer = SimpleNamespace(step=lambda: steps.append(1))
        with pytest.raises(ValueError, match=rf'^{seen} observed cell\(s\) have a mean that is negative or not finite$'):
            if prepared:
                guards = step_guard.StepGuards()
                with step_guard.trust(guards):
                    training_step.optimize_batch(fa, batch_prep.Preparation()(batch), optimizer, SCALES, 0, components, score=train.score_blocks, clock=lambda: 0., guards=guards)
            else:
                training_step.optimize_batch(fa, batch, optimizer, SCALES, 0, components, score=train.score_blocks, clock=lambda: 0.)
        assert steps == [] and all(p.grad is None for p in model.parameters())


def test_a_training_run_with_a_broken_model_stops_at_the_first_step_and_writes_no_checkpoint(tmp_path, rows, stats, monkeypatch):
    real = heads.EPTHeads.m2_rate
    monkeypatch.setattr(heads.EPTHeads, 'm2_rate', lambda self, enc, exposure, pos=None: -real(self, enc, exposure, pos))
    steps = []
    real_step = torch.optim.AdamW.step
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda self, *a, **k: (steps.append(1), real_step(self, *a, **k))[1])
    with pytest.raises(ValueError, match=r'observed cell\(s\) have a mean that is negative or not finite'):
        run(tmp_path / 'broken', make_data(rows, stats), make_cfg(), epochs=1)
    assert steps == [] and not list((tmp_path / 'broken').glob('epoch_*'))


def device_reads(fn) -> dict:
    """The reads of a value out of a tensor that a device has to synchronise for, in ``fn``: the profiler's ``_local_scalar_dense``, ``nonzero`` and ``unique`` events and the ``tolist`` calls."""
    counts = {'tolist': 0}
    real = torch.Tensor.tolist

    def tolist(self, *a, **k):
        counts['tolist'] += 1
        return real(self, *a, **k)

    torch.Tensor.tolist = tolist
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            fn()
    finally:
        torch.Tensor.tolist = real
    for event in prof.events():
        if event.name in ('aten::_local_scalar_dense', 'aten::nonzero', 'aten::_unique2', 'aten::unique_dim', 'aten::masked_select'):
            counts[event.name] = counts.get(event.name, 0) + 1
    counts['total'] = sum(counts.values())
    return counts


@pytest.mark.parametrize('law', ['poisson', 'shifted'])
def test_a_trusted_step_reads_back_from_the_device_a_handful_of_times_and_the_untrusted_step_hundreds(rows, shifted_rows, stats, law):
    """The point of the item (counted on the CPU; on a device each of these reads waits for the whole queue): the step of every earlier run read the range or the finiteness of every input and the masks of
    every block back, about 100 to 300 times for a batch of this size; the trusted step reads the guards and the loss once."""
    model = a_model()
    model.train()
    batch = a_batch(rows if law == 'poisson' else shifted_rows, stats, 'random_view', size=32)
    old = device_reads(lambda: a_step(model, batch, prepared=False, kill_law=law))
    prepared = batch_prep.Preparation(kill_law=law)(batch)

    def trusted_step():
        fa = heads.FrameAveraged(model, average=False)
        guards = step_guard.StepGuards()
        components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
        with step_guard.trust(guards):
            training_step.optimize_batch(fa, prepared, noop_optimizer(), SCALES, 0, components, score=functools.partial(train.score_blocks, **law_args(law, True)) if law != 'poisson' else train.score_blocks,
                                         clock=lambda: 0., guards=guards)

    model.zero_grad(set_to_none=True)
    new = device_reads(trusted_step)
    assert old['total'] > 100, old
    assert new['tolist'] == 1 and new['total'] * 10 <= old['total'], (old, new)      # the one read of the guards and the loss; what is left is not the check of an input


@pytest.mark.parametrize('law,zone', [('poisson', True), ('shifted', False)])
def test_a_trusted_step_makes_no_union_and_no_mask_selection_on_the_device_path(rows, shifted_rows, stats, monkeypatch, law, zone):
    """The lists of the plan replace ``torch.unique`` of the shared M2 call and the boolean-mask selections (``nonzero``, ``masked_select``) of the loss: a trusted step on a prepared batch calls none of them."""
    model = a_model()
    model.train()
    prepared = batch_prep.Preparation(kill_law=law, zone_mark=zone)(a_batch(rows if law == 'poisson' else shifted_rows, stats, 'random_view', size=32))
    called = []
    for owner, name in ((torch, 'unique'), (torch.Tensor, 'nonzero'), (torch.Tensor, 'masked_select'), (torch.Tensor, '__getitem__')):
        real = getattr(owner, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            if _name != '__getitem__' or any(isinstance(a, torch.Tensor) and a.dtype == torch.bool for a in (args[1] if isinstance(args[1], tuple) else (args[1],))):
                called.append(_name)
            return _real(*args, **kwargs)
        monkeypatch.setattr(owner, name, spy)
    fa = heads.FrameAveraged(model, average=False)
    guards = step_guard.StepGuards()
    score = functools.partial(train.score_blocks, **law_args(law, zone)) if law_args(law, zone) else train.score_blocks
    components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
    with step_guard.trust(guards):
        training_step.optimize_batch(fa, prepared, noop_optimizer(), SCALES, 0, components, score=score, clock=lambda: 0., guards=guards)
    assert called == [], called


def test_a_batch_with_no_row_that_counts_is_refused_like_before(rows, stats):
    """The last check of the step is ``total_loss``: nothing to average is an error (``score_blocks`` found no block, or no block has a row that counts)."""
    model = a_model().eval()
    empty = {k: (v[:0] if k.count('/') == 2 else v) for k, v in a_batch(rows, stats, 'average').items()}
    old, new = legacy_error(model, empty), preparation_error(empty)
    assert isinstance(old, ValueError) and 'nothing to average' in str(old) and same_error(old, new)


def test_the_data_checks_are_made_whatever_the_scope_of_the_caller_says(rows, stats):
    """The dry run is never trusted: a preparation that is called inside a trusted scope (a caller's mistake) still refuses a bad batch."""
    batch = dict(a_batch(rows, stats, 'average'))
    batch['M3Obj/masked/K'] = torch.full_like(batch['M3Obj/masked/K'], -1.0)
    with step_guard.trust(step_guard.StepGuards()):
        assert isinstance(preparation_error(batch), ValueError)
    assert isinstance(preparation_error(batch), ValueError)


def test_the_zone_block_is_checked_only_when_it_is_scored(rows, stats):
    """With ``zone_mark=False`` the block M2Zone is not scored (``score_blocks``), so a field of it that only a check of that block would refuse is no reason to refuse the batch -- as before."""
    model = a_model().eval()
    batch = dict(a_batch(rows, stats, 'average'))
    batch['M2Zone/masked/m'] = torch.full_like(batch['M2Zone/masked/m'], 99)               # no such mark
    assert isinstance(legacy_error(model, batch, zone_mark=True), ValueError) and same_error(legacy_error(model, batch, zone_mark=True), preparation_error(batch, zone_mark=True))
    assert legacy_error(model, batch, zone_mark=False) is None and preparation_error(batch, zone_mark=False) is None
    plan = batch_prep.build_plan(batch, zone_mark=False)
    assert not any('M2Zone' in key for key in plan)                                        # nor is a list made for it


@pytest.mark.parametrize('law,zone', [('poisson', True), ('shifted', False)])
def test_the_training_loop_runs_every_fit_step_trusted_on_a_prepared_batch_and_the_stop_evaluation_untrusted(tmp_path, rows, shifted_rows, stats, monkeypatch, law, zone):
    """What the loop hands to the step: a batch of the preparation of this run (the index lists, the settings of the config, no coin), the guards of the step, a trusted scope; and the STOP evaluation, which
    is not prepared, with every check on."""
    seen, stop_scopes = [], []
    real_step, real_score = training_step.optimize_batch, train.score_blocks

    def step(fa, batch, optimizer, scales, n, components, **kw):
        seen.append((step_guard.trusted(), step_guard.current() is kw.get('guards'), isinstance(kw.get('guards'), step_guard.StepGuards), any(k.startswith('plan/') for k in batch),
                     'swap_view' in batch, getattr(batch, 'settings', None)))
        return real_step(fa, batch, optimizer, scales, n, components, **kw)

    def score(fa, enc, batch, *a, **k):
        if not torch.is_grad_enabled():
            stop_scopes.append((step_guard.trusted(), any(key.startswith('plan/') for key in batch)))
        return real_score(fa, enc, batch, *a, **k)

    monkeypatch.setattr(training_step, 'optimize_batch', step)
    monkeypatch.setattr(train, 'score_blocks', score)
    by_role = rows if law == 'poisson' else shifted_rows
    result = run(tmp_path / 'trusted', make_data(by_role, stats, 'random_view'), make_cfg(swap_mode='random_view', kill_law=law, zone_mark=zone), epochs=2)
    settings = {'kill_law': law, 'zone_mark': zone}
    assert len(seen) == result.epochs_run * FIT_BATCHES
    assert all(entry == (True, True, True, True, False, settings) for entry in seen), seen[:2]
    assert stop_scopes and all(entry == (False, False) for entry in stop_scopes)


# ---- item 4: the packed transfer (SPD decision D32) ---------------------------------------------------------------------------------------------------------------
class Counted:
    """Records the copies to a device: ``(dtype, number of elements, non_blocking)`` of every ``Tensor.to`` that carries a device."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = torch.Tensor.to

        def to(self_, *args, **kwargs):
            if (args and isinstance(args[0], (str, torch.device))) or 'device' in kwargs:
                self.calls.append((self_.dtype, self_.numel(), kwargs.get('non_blocking', False)))
            return real(self_, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, 'to', to)


def a_prepared_batch(rows_by_role, stats, mode='random_view', pack=True, **settings):
    batch = a_batch(rows_by_role, stats, mode)
    return batch, batch_prep.Preparation(pack=pack, **settings)(batch)


def test_the_pack_holds_one_buffer_for_each_dtype_and_every_tensor_in_its_own_aligned_slot(rows, stats):
    batch, prepared = a_prepared_batch(rows, stats)
    plain = batch_prep.Preparation(pack=False)(batch)
    packed = prepared.packed
    assert isinstance(packed, batch_prep.Packed) and set(packed.buffers) == {torch.float32, torch.int64, torch.bool}        # float32 inputs, int64 indices and rows, bool masks
    assert list(prepared) == list(plain)                                                                                    # the order of the keys is that of the unpacked batch
    spans = {}
    for key, tensor in prepared.items():
        assert tensor.dtype == plain[key].dtype and tensor.shape == plain[key].shape and torch.equal(tensor, plain[key]), key
        assert tensor.is_contiguous()
        if not tensor.numel():
            continue
        buffer = packed.buffers[tensor.dtype]
        offset = tensor.data_ptr() - buffer.data_ptr()
        assert offset % batch_prep.ALIGN_BYTES == 0 and offset + tensor.numel() * tensor.element_size() <= buffer.numel() * buffer.element_size(), key
        spans.setdefault(tensor.dtype, []).append((offset, offset + tensor.numel() * tensor.element_size()))
    for dtype, found in spans.items():                                                                                      # no two slots overlap
        found.sort()
        assert all(a[1] <= b[0] for a, b in zip(found, found[1:])), dtype
        assert all(b[0] - a[1] < batch_prep.ALIGN_BYTES for a, b in zip(found, found[1:])), dtype                           # and the padding is the least that aligns: the next slot starts at the first aligned offset
    assert batch_prep.ALIGN_BYTES == 512                                                                                   # the alignment of a block of the caching allocator of the device


def test_the_copy_to_the_device_is_one_call_for_each_dtype_instead_of_one_for_each_tensor(rows, stats, monkeypatch):
    batch, packed = a_prepared_batch(rows, stats, pack=True)
    unpacked = batch_prep.Preparation(pack=False)(batch)
    counted = Counted(monkeypatch)
    moved = batch_prep.to_device(packed, 'cpu')
    assert sorted(str(c[0]) for c in counted.calls) == sorted(str(d) for d in (torch.float32, torch.int64, torch.bool))
    assert list(moved) == list(packed) and all(torch.equal(moved[k], unpacked[k]) and moved[k].dtype == unpacked[k].dtype for k in moved)
    counted.calls.clear()
    assert batch_prep.to_device(unpacked, 'cpu') is unpacked and counted.calls == []                                        # on the CPU an unpacked batch needs no copy at all (as ``train._to``)
    assert len(unpacked) > 100                                                                                              # the tensors that the three copies replace, for a device: more than a hundred


def test_the_buffers_are_pinned_and_copied_without_waiting_only_for_a_cuda_device(rows, stats, monkeypatch):
    """The two lines that only a GPU can run, checked by their arguments: pinned host buffers for ``cuda`` (plain ones for the CPU) and ``non_blocking`` for a pinned buffer."""
    pinned, real_zeros = [], torch.zeros

    def zeros(*args, **kwargs):
        if 'pin_memory' in kwargs:
            pinned.append(kwargs.pop('pin_memory'))                                                                         # this machine has no device: the call is recorded, not made
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, 'zeros', zeros)
    batch = a_batch(rows, stats)
    batch_prep.Preparation(pack=True, device='cuda')(batch)
    assert pinned and all(pinned)
    pinned.clear()
    batch_prep.Preparation(pack=True, device='cpu')(batch)
    assert pinned and not any(pinned)
    monkeypatch.undo()
    prepared = batch_prep.Preparation(pack=True)(batch)
    sent, real_to = [], torch.Tensor.to

    def to(self_, *args, **kwargs):
        if args and args[0] == 'cuda':
            sent.append(kwargs.get('non_blocking'))
            return self_                                                                                                    # (no device here)
        return real_to(self_, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', to)
    monkeypatch.setattr(torch.Tensor, 'is_pinned', lambda self_: True)
    batch_prep.to_device(prepared, 'cuda')
    assert sent == [True, True, True]                                                                                       # a pinned buffer is copied without waiting for the host
    sent.clear()
    monkeypatch.setattr(torch.Tensor, 'is_pinned', lambda self_: False)
    batch_prep.to_device(prepared, 'cuda')
    assert sent == [False, False, False]                                                                                    # a pageable buffer is copied the plain way: the copy returns when the data is safe


def test_should_pack_is_true_for_a_cuda_device_only_and_the_preparation_asks_it(monkeypatch):
    assert batch_prep.should_pack(torch.device('cuda')) is True and batch_prep.should_pack('cuda') is True
    assert batch_prep.should_pack(torch.device('cpu')) is False and batch_prep.should_pack('cpu') is False and batch_prep.should_pack(None) is False
    assert batch_prep.Preparation().packed_for is False and batch_prep.Preparation(device='cuda').packed_for is True and batch_prep.Preparation(device='cpu').packed_for is False
    assert batch_prep.Preparation(pack=True).packed_for is True and batch_prep.Preparation(pack=False, device='cuda').packed_for is False
    monkeypatch.setattr(batch_prep, 'should_pack', lambda device: True)
    assert batch_prep.Preparation(device='cpu').packed_for is True                                                          # the proof script and the tests can ask for the packed path on the CPU


@pytest.mark.parametrize('law,mode', [('poisson', 'random_view'), ('shifted', 'average')])
def test_an_unpacked_view_is_a_value_for_value_copy_of_the_tensor_that_was_packed_and_slots_do_not_alias(rows, shifted_rows, stats, law, mode):
    by_role = rows if law == 'poisson' else shifted_rows
    batch = a_batch(by_role, stats, mode)
    unpacked = batch_prep.Preparation(kill_law=law, pack=False)(batch)
    moved = batch_prep.to_device(batch_prep.Preparation(kill_law=law, pack=True)(batch), 'cpu')
    assert list(moved) == list(unpacked)
    for key in unpacked:
        assert moved[key].dtype == unpacked[key].dtype and moved[key].shape == unpacked[key].shape and torch.equal(moved[key], unpacked[key]), key
    first, second = [k for k in moved if moved[k].dtype == torch.float32 and moved[k].numel()][:2]
    before = moved[second].clone()
    moved[first].zero_()                                                                                                    # writing one slot leaves the others as they were
    assert torch.equal(moved[second], before)


def test_a_tensor_that_is_not_contiguous_is_not_packed_but_copied_as_it_was(monkeypatch):
    """Only a contiguous tensor is laid in a buffer (it has the strides that its own copy would have); another keeps the copy of ``train._to`` and its strides, so no kernel sees another layout."""
    wide = torch.arange(12.0).view(3, 4)
    tensors = {'a': torch.arange(5), 'b': wide.t(), 'c': wide[:, 1:3], 'd': torch.tensor([True, False])}
    packed = batch_prep.pack(tensors, pin=False)
    assert list(packed.to('cpu')) == ['a', 'b', 'c', 'd'] and set(packed.buffers) == {torch.int64, torch.bool}
    counted = Counted(monkeypatch)
    out = packed.to('cpu')
    assert len(counted.calls) == 2 + 2                                                                                      # a copy for each of the two buffers and one for each of the two loose tensors
    for key, t in tensors.items():
        assert out[key].dtype == t.dtype and out[key].shape == t.shape and torch.equal(out[key], t) and out[key].stride() == t.stride(), key


def test_an_unpacked_batch_is_copied_tensor_by_tensor_to_a_device_and_not_at_all_to_the_cpu(rows, stats, monkeypatch):
    unpacked = batch_prep.Preparation(pack=False)(a_batch(rows, stats))
    sent, real_to = [], torch.Tensor.to

    def to(self_, *args, **kwargs):
        if args and args[0] == 'cuda':
            sent.append(kwargs.get('non_blocking', False))
            return self_ + 0                                                                                                # (no device here: a fresh tensor stands for the copy)
        return real_to(self_, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', to)
    moved = batch_prep.to_device(unpacked, 'cuda')
    assert len(sent) == len(unpacked) and set(sent) == {False} and list(moved) == list(unpacked) and moved is not unpacked        # one plain copy for each tensor, as train._to made
    assert batch_prep.to_device(unpacked, 'cpu') is unpacked


def test_empty_tensors_and_other_dtypes_round_trip():
    tensors = {'a': torch.zeros(0, 18), 'b': torch.arange(7), 'c': torch.ones(3, 2, dtype=torch.float64), 'd': torch.tensor([True, False]), 'e': torch.zeros(0, dtype=torch.bool), 'f': torch.arange(5.0),
               'g': torch.tensor(3.5), 'h': torch.zeros(0, dtype=torch.int32)}
    packed = batch_prep.pack(tensors, pin=False)
    assert set(packed.buffers) == {torch.float32, torch.int64, torch.float64, torch.bool, torch.int32}
    out = packed.to('cpu')
    assert list(out) == list(tensors)
    for key, t in tensors.items():
        assert out[key].dtype == t.dtype and out[key].shape == t.shape and torch.equal(out[key], t), key


@pytest.mark.parametrize('law,mode,zone', [('poisson', 'random_view', True), ('shifted', 'random_view', False), ('poisson', 'average', True), ('poisson', 'none', True)])
def test_a_training_run_on_the_packed_path_gives_the_state_of_the_unpacked_one_bit_for_bit(tmp_path, rows, shifted_rows, stats, monkeypatch, law, mode, zone):
    by_role = rows if law == 'poisson' else shifted_rows
    states = {}
    for name, packed in (('plain', False), ('packed', True)):
        monkeypatch.setattr(batch_prep, 'should_pack', lambda device, _p=packed: _p)
        result = run(tmp_path / name, make_data(by_role, stats, mode), make_cfg(swap_mode=mode, kill_law=law, zone_mark=zone), epochs=2)
        folder = Path(result.checkpoint_dir)
        assert result.epochs_run == 2
        last = torch.load(folder / 'last.pt', weights_only=True, map_location='cpu')
        states[name] = (result.fit_batch_mean_losses, result.stop_losses, last['model'], last['optimizer']['state'], result.model.state_dict())          # the weights of the last epoch, the Adam state, and the weights of the best STOP epoch
    plain, packed = states['plain'], states['packed']
    assert plain[0] == packed[0] and plain[1] == packed[1]
    for a, b in ((plain[2], packed[2]), (plain[4], packed[4])):
        assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert list(plain[3]) == list(packed[3]) and all(torch.equal(plain[3][i][k], packed[3][i][k]) for i in plain[3] for k in plain[3][i])


def test_the_loop_moves_a_packed_batch_with_one_copy_for_each_dtype(tmp_path, rows, stats, monkeypatch):
    seen = []
    real_to_device = batch_prep.to_device
    monkeypatch.setattr(batch_prep, 'should_pack', lambda device: True)
    monkeypatch.setattr(batch_prep, 'to_device', lambda batch, device: (seen.append((type(batch).__name__, batch.packed is not None, str(device))), real_to_device(batch, device))[1])
    result = run(tmp_path / 'p', make_data(rows, stats, 'random_view'), make_cfg(swap_mode='random_view'), epochs=1)
    assert seen == [('PreparedBatch', True, 'cpu')] * (result.epochs_run * FIT_BATCHES)
