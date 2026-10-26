"""Tests of the checks of a training step (``c_ept/ept/step_guard.py`` and the check functions of ``heads``, ``loss``, ``killlaw`` and ``training_step`` that use it; EPT-T012 S4b speed-ups SPD, author
decision D32, item 3).

What is pinned, by behaviour:

* ``trust`` is a scope of the thread: ``trusted()`` is True inside it, False outside (also after an exception), False in another thread, False inside ``untrusted``;
* outside a trusted scope every check function is what it was (the tests of ``heads``, ``loss`` and ``train`` run there); inside it the data checks are not read back from the device and the model
  checks are deferred (nothing raises) into the ``StepGuards`` of the step, with the message that the check raised;
* ``StepGuards.check`` reads every deferred count and the finiteness of the loss in ONE transfer, raises the ValueError of the first failing check with its count before the FloatingPointError of
  a loss that is not finite, and a refused step has run no backward and no optimizer step;
* the row means of ``loss.head_losses`` from the index lists of the plan are the very bits of the means from the boolean masks.
"""
# Ref: the performance audit of the S4b hot path (perf-audit.md, item 3).
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from c_ept.ept import loss, step_guard, training_step
from c_ept.models import killlaw


def guards_with(*counts_and_messages):
    guards = step_guard.StepGuards()
    for count, message in counts_and_messages:
        guards.defer(torch.tensor(count), message)
    return guards


# ---- the scope ----------------------------------------------------------------------------------------------------------------------------------
def test_trust_is_a_scope_of_the_thread_that_is_given_back_also_after_an_error():
    assert not step_guard.trusted() and step_guard.current() is None
    guards = step_guard.StepGuards()
    with step_guard.trust(guards) as inside:
        assert inside is guards and step_guard.trusted() and step_guard.current() is guards
        seen = []
        thread = threading.Thread(target=lambda: seen.append((step_guard.trusted(), step_guard.current())))
        thread.start()
        thread.join()
        assert seen == [(False, None)]                                          # a producer thread or a test in another thread is not trusted by this scope
        with step_guard.untrusted():
            assert not step_guard.trusted() and step_guard.current() is None    # the dry run of the data checks is never trusted
        assert step_guard.current() is guards
    assert not step_guard.trusted()
    with pytest.raises(KeyError):
        with step_guard.trust(guards):
            raise KeyError('x')
    assert not step_guard.trusted()


def test_trust_needs_the_guards_of_the_step():
    for bad in (None, True, {}, 'guards'):
        with pytest.raises(TypeError):
            with step_guard.trust(bad):
                pass


# ---- the single read -----------------------------------------------------------------------------------------------------------------------------
class Reads:
    """Counts the reads of a value out of a tensor (``item``, ``tolist``, ``bool``, ``int``, ``float``): what a device has to synchronise for."""

    def __init__(self, monkeypatch):
        self.n = 0
        for name in ('item', 'tolist', '__bool__', '__int__', '__float__'):
            real = getattr(torch.Tensor, name)

            def counted(self_, *a, _real=real, **k):
                self.n += 1
                return _real(self_, *a, **k)
            monkeypatch.setattr(torch.Tensor, name, counted)


def test_deferring_reads_nothing_and_the_check_reads_once(monkeypatch):
    guards = step_guard.StepGuards()
    reads = Reads(monkeypatch)
    for _ in range(7):
        guards.defer(torch.tensor(0), 'check {}')
    assert reads.n == 0 and len(guards) == 7                                    # recording a model check is not a read
    guards.check(torch.tensor(True), 0)
    assert reads.n == 1                                                         # all of them and the finiteness of the loss in one transfer


def test_the_first_failing_guard_raises_with_its_count_before_a_loss_that_is_not_finite():
    guards = guards_with((0, 'first {}'), (3, '{} cells of the second'), (2, '{} of the third'))
    with pytest.raises(ValueError, match='^3 cells of the second$'):
        guards.check(torch.tensor(False), 4)                                    # the loss is not finite too: the guard wins, as the check that ran in the forward did
    guards = guards_with((0, 'a'), (0, 'b'))
    with pytest.raises(FloatingPointError, match='^the training loss is not finite at step 17$'):
        guards.check(torch.tensor(False), 17)
    guards.check(torch.tensor(True), 17)                                        # nothing failed: no error
    with pytest.raises(ValueError, match='^a text without the count$'):
        guards_with((1, 'a text without the count')).check(torch.tensor(True), 0)


def test_a_guard_is_recorded_without_a_graph():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    guards = step_guard.StepGuards()
    guards.defer((x > 0).sum(), '{}')
    guards.defer((x * 2).sum().to(torch.int64), '{}')
    assert not any(count.requires_grad for count, _ in guards._items)


# ---- the check functions: outside the scope they are what they were, inside they skip the data checks and defer the model checks ------------------------------
# (through the public entry points of ``loss`` and ``killlaw``: the private check functions of another package are not imported by a test, tools/check_project_boundaries.py)
def inside():
    guards = step_guard.StepGuards()
    return guards, step_guard.trust(guards)


def category_out(n, tail):
    return torch.log(torch.full((n, *tail), 1.0 / tail[-1]))


def test_the_data_checks_of_the_loss_are_not_made_on_the_device_inside_a_trusted_step_and_are_outside():
    q_out, fractional = torch.tensor([0, -3, 1]), torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    mark_target = {'q': q_out, 'y': torch.tensor([0, 1, 2])}
    bad_cases = [
        ('q must lie in 0 .. 17', lambda: loss.head_nll('M2Mark', category_out(3, (18, 52)), mark_target)),
        ('not non-negative whole numbers', lambda: loss.head_nll('M3Kills', torch.ones(1, 4, 2), {'s': torch.tensor([0]), 'y': fractional})),
        ('finite and not negative', lambda: loss.head_nll('M2Rate', torch.ones(1, 18), {'y': torch.zeros(1, 18), 'e': torch.full((1, 18), -1.0)})),
        ('finite and not negative', lambda: loss.head_nll('M4Conv', torch.ones(1, 18), {'y': torch.zeros(1, 18), 'e': torch.tensor([float('nan')])})),
        ('lie outside', lambda: loss.head_nll('M3Size', torch.zeros(1, 4), {'y': torch.tensor([-2])})),
    ]
    for message, call in bad_cases:
        with pytest.raises(ValueError, match=message):
            call()
    guards, scope = inside()
    with scope:
        for message, call in bad_cases:
            assert call().shape[0] in (1, 3)                       # the batch passed these checks on the CPU: not read back from the device again (the score of a junk input is not looked at)
        with pytest.raises(TypeError, match='integer tensor'):
            loss.head_nll('M2Mark', category_out(3, (18, 52)), {'q': torch.zeros(3, dtype=torch.uint8), 'y': torch.tensor([0, 1, 2])})      # the checks that need no read (type, shape) stay
        with pytest.raises(ValueError, match='shape'):
            loss.head_nll('M2Mark', category_out(3, (18, 52)), {'q': torch.zeros(4, dtype=torch.long), 'y': torch.tensor([0, 1, 2])})


def test_the_marks_and_the_first_teams_of_the_shifted_law_are_not_read_back_inside_a_trusted_step():
    marks = torch.tensor([3, 70])
    with pytest.raises(ValueError, match='a mark is 0 .. 51'):
        killlaw.torch_first_team(marks)
    with pytest.raises(ValueError, match='the first team d is 0'):
        killlaw.torch_logpmf(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([4]), 'shifted')
    assert killlaw.torch_first_team(marks, check_range=False).tolist() == [0, -1]         # the formula is the same; only the read of the range is not made
    assert killlaw.torch_logpmf(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([4]), 'shifted', check_range=False).shape == (1, 2)
    with pytest.raises(TypeError, match='integer tensor'):
        killlaw.torch_first_team(torch.zeros(2, dtype=torch.uint8), check_range=False)        # the type check stays
    target = {'m': marks, 's': torch.tensor([0, 0]), 'y': torch.tensor([[1, 0], [1, 1]])}
    with pytest.raises(ValueError, match='the mark m must lie in 0 .. 51'):
        loss.head_nll('M3Kills', torch.ones(2, 4, 2), target, kill_law='shifted')
    guards, scope = inside()
    with scope:                                                                           # the loss passes the switch for a trusted step and only then
        score = loss.head_nll('M3Kills', torch.ones(2, 4, 2), target, kill_law='shifted')
    assert score.shape == (2,) and len(guards) == 1
    guards.check(torch.tensor(True), 0)
    open_target = {'d': torch.tensor([4]), 's': torch.tensor([0]), 'y': torch.tensor([[1, 0]])}      # an open block names the first team itself: the range of d is a data check too
    with pytest.raises(ValueError, match='the first-kill team d must lie in -1 .. 1'):
        loss.head_nll('M3Kills', torch.ones(1, 4, 2), open_target, kill_law='shifted')
    guards, scope = inside()
    with scope:
        assert loss.head_nll('M3Kills', torch.ones(1, 4, 2), open_target, kill_law='shifted').shape == (1,)      # not read back (``killlaw.torch_logpmf`` made the same read)


def test_a_model_check_raises_at_once_outside_a_trusted_step_and_is_deferred_inside_it_with_the_same_message():
    rates = torch.ones(2, 18)
    rates[0, 1], rates[1, 2] = float('nan'), -1.0
    target = {'y': torch.zeros(2, 18), 'e': torch.ones(2, 18)}
    with pytest.raises(ValueError, match=r'^2 observed cell\(s\) have a mean that is negative or not finite$'):
        loss.head_nll('M2Rate', rates, target)
    guards, scope = inside()
    with scope:
        cells = loss.head_nll('M2Rate', rates, target)                                    # nothing raised
    assert cells.shape == (2,) and len(guards) == 1
    with pytest.raises(ValueError, match=r'^2 observed cell\(s\) have a mean that is negative or not finite$'):
        guards.check(torch.tensor(True), 0)                                               # the very message, with the count of the device
    unseen = {'y': torch.zeros(2, 18), 'e': torch.zeros(2, 18)}
    loss.head_nll('M2Rate', rates, unseen)                                                # a bad mean in a cell that is not observed is not a failure (as before)
    count_without_exposure = {'y': torch.ones(1, 18), 'e': torch.zeros(1, 18)}
    with pytest.raises(ValueError, match=r'^18 cell\(s\) have a count but no exposure$'):
        loss.head_nll('M2Rate', torch.ones(1, 18), count_without_exposure)
    guards, scope = inside()
    with scope:
        loss.head_nll('M2Rate', torch.ones(1, 18), count_without_exposure)                # a data check: made on the CPU
    assert len(guards) == 1                                                               # only the model check of the mean was deferred


def test_the_shifted_kill_rates_check_is_deferred_with_the_message_of_the_mean_check():
    target = {'m': torch.tensor([3, 3]), 's': torch.tensor([0, 1]), 'y': torch.tensor([[1, 0], [2, 1]])}
    rates = torch.ones(2, 4, 2)
    rates[1, 1, 0] = float('nan')
    with pytest.raises(ValueError, match=r'^1 observed cell\(s\) have a mean that is negative or not finite$'):
        loss.head_nll('M3Kills', rates, target, kill_law='shifted')
    guards, scope = inside()
    with scope:
        loss.head_nll('M3Kills', rates, target, kill_law='shifted')
    with pytest.raises(ValueError, match=r'^1 observed cell\(s\) have a mean that is negative or not finite$'):
        guards.check(torch.tensor(True), 0)


def test_the_log_law_checks_of_the_category_rows_are_deferred_with_their_messages_and_give_the_same_row_scores():
    good = category_out(3, (4,))
    y = torch.tensor([2, -1, 0])
    outside = loss.head_nll('M3Size', good, {'y': y})
    guards, scope = inside()
    with scope:
        within = loss.head_nll('M3Size', good, {'y': y})
    assert torch.equal(outside, within) and len(guards) == 2
    guards.check(torch.tensor(True), 0)                                                   # a valid law: nothing fails
    for law, row_y, message in ((torch.log(torch.tensor([[0.25, float('nan'), 0.25, 0.5]])), torch.tensor([0]), 'holds NaN or \\+infinity in a row that is read'),
                                (torch.tensor([[0.0, -1.0, -1.0, -1.0]]), torch.tensor([1]), 'does not add up to 1')):
        with pytest.raises(ValueError, match=f'^a log law of M3Size {message}'):
            loss.head_nll('M3Size', law, {'y': row_y})
        guards, scope = inside()
        with scope:
            loss.head_nll('M3Size', law, {'y': row_y})
        with pytest.raises(ValueError, match=f'^a log law of M3Size {message}'):
            guards.check(torch.tensor(True), 0)
    for law in (torch.tensor([[0.0, -1.0, -1.0, -1.0]]), torch.log(torch.tensor([[0.25, float('nan'), 0.25, 0.5]])), torch.tensor([[float('inf'), 0.0, 0.0, 0.0]])):
        loss.head_nll('M3Size', law, {'y': torch.tensor([-1])})                           # a law that fails in a row that is NOT read (y = -1): no failure, inside or outside
        guards, scope = inside()
        with scope:
            loss.head_nll('M3Size', law, {'y': torch.tensor([-1])})
        guards.check(torch.tensor(True), 0)
    two = torch.stack([torch.tensor([0.0, -1.0, -1.0, -1.0]), torch.log(torch.tensor([0.25, float('nan'), 0.25, 0.5]))])       # the first row does not add up, the second holds a NaN
    with pytest.raises(ValueError, match='^a log law of M3Size holds NaN'):
        loss.head_nll('M3Size', two, {'y': torch.tensor([1, 0])})                         # the NaN check comes first, over all the rows that are read
    guards, scope = inside()
    with scope:
        loss.head_nll('M3Size', two, {'y': torch.tensor([1, 0])})
    with pytest.raises(ValueError, match='^a log law of M3Size holds NaN'):
        guards.check(torch.tensor(True), 0)                                               # and so it does when deferred: the guards keep the order of the checks


# ---- the row means from the index lists --------------------------------------------------------------------------------------------------------------
def a_block(head, n, seed):
    """Fields of a block of ``n`` rows and a row score for each, deterministic."""
    g = torch.Generator().manual_seed(seed)
    size = torch.randint(0, 4, (n,), generator=g)
    y = {'M3Kills': torch.randint(0, 4, (n, 2), generator=g), 'M3Obj': torch.randint(0, 3, (n, 18), generator=g), 'M3Gold': torch.randint(-1, 31, (n,), generator=g)}[head]       # M3Gold: -1 = no gold bin, the row does not count
    if head == 'M3Gold' and n > 1:
        y[::3] = -1                                                                      # rows that do not count, for sure
    rows = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    return {'s': size, 'y': y}, rows.requires_grad_()


@pytest.mark.parametrize('head', ['M3Kills', 'M3Obj', 'M3Gold'])
@pytest.mark.parametrize('n', [1, 7, 40])
def test_the_means_from_the_plan_are_the_bits_of_the_means_from_the_masks_and_so_are_their_gradients(head, n):
    target, rows = a_block(head, n, seed=n)
    key = f'{head}/masked'
    expected = loss.head_losses({key: rows}, {key: target})[key]
    expected.backward()
    want_grad, rows.grad = rows.grad.clone(), None
    plan = loss.counting_plan(head, 'masked', target)
    assert set(plan) == {loss.plan_key(head, 'masked', name) for name in ('counted', 'size0', 'size1', 'size2', 'size3')}
    block = {**target, **plan}
    guards, scope = inside()
    with scope:
        got = loss.head_losses({key: rows}, {key: block})[key]
    got.backward()
    assert torch.equal(got, expected) and torch.equal(rows.grad, want_grad)
    assert len(guards) == 0
    plain = loss.head_losses({key: rows}, {key: block})[key]                              # outside a trusted step the plan is not used: the masks again, the same bits
    assert torch.equal(plain, expected)
    wrong = {**block, 's': torch.full_like(block['s'], 9)}                                # ... and the checks are made: a size out of range is refused, whatever lists the target carries
    with pytest.raises(ValueError, match='the size class s must lie in 0 .. 3'):
        loss.head_losses({key: rows}, {key: wrong})
    only_counted = {k: v for k, v in block.items() if not k.startswith(loss.PLAN_PREFIX) or k.endswith('/counted')}      # a plan that is not whole is no plan: the masks again, the same bits
    guards, scope = inside()
    with scope:
        assert torch.equal(loss.head_losses({key: rows}, {key: only_counted})[key], expected)
    if n > 1:
        guards, scope = inside()
        with scope, pytest.raises(ValueError, match=rf'the block {key} has 1 scores but its target has {n} rows'):
            loss.head_losses({key: rows[:1]}, {key: block})                               # scores and rows of another count are refused with the plan too


def test_a_block_without_a_row_that_counts_is_left_out_with_the_plan_too():
    target = {'y': torch.tensor([-1, -1, -1])}
    rows = torch.rand(3, dtype=torch.float64)
    plan = loss.counting_plan('M2Mark', 'masked', target)
    assert plan[loss.plan_key('M2Mark', 'masked', 'counted')].numel() == 0 and loss.head_losses({'M2Mark/masked': rows}, {'M2Mark/masked': target}) == {}
    guards, scope = inside()
    with scope:
        assert loss.head_losses({'M2Mark/masked': rows}, {'M2Mark/masked': {**target, **plan}}) == {}


# ---- the step ----------------------------------------------------------------------------------------------------------------------------------------
def a_world(monkeypatch):
    """A stand-in model of one weight and a score that records its model check in the trusted scope, as the loss functions do; ``total_loss`` returns the one block."""
    class Fa(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([3.0]))

        def encode(self, batch):
            return 'encoded'

    fa = Fa()
    monkeypatch.setattr(loss, 'total_loss', lambda nlls, scales, targets: nlls['block'])
    return fa


def step(fa, failing, finite=True, guards=True):
    seen = []

    def score(fa_, encoded, batch):
        current = step_guard.current()
        if current is not None:
            current.defer(torch.tensor(failing), '{} cell(s) of the model are not acceptable')
        nll = 2.0 * fa_.w.sum() * (1.0 if finite else float('inf'))
        return {'block': (None, 'mask', nll)}

    components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
    optimizer = SimpleNamespace(step=lambda: seen.append('step'))
    given = step_guard.StepGuards()
    if guards:
        with step_guard.trust(given):
            total = training_step.optimize_batch(fa, 'batch', optimizer, {'block': 1.0}, 3, components, score=score, clock=lambda: 0., guards=given)
    else:                                                                                 # the step of every earlier run: no scope, no guards
        total = training_step.optimize_batch(fa, 'batch', optimizer, {'block': 1.0}, 3, components, score=score, clock=lambda: 0.)
    return total, seen


def test_a_step_whose_guards_are_not_those_of_the_active_scope_fails_closed_before_it_does_anything(monkeypatch):
    """Review m1: a trusted scope whose guards the step does not read would let the deferred model checks pass unread."""
    fa = a_world(monkeypatch)
    encoded = []
    fa.encode = lambda batch: encoded.append(1)
    components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
    optimizer = SimpleNamespace(step=lambda: encoded.append('step'))
    scope_guards, other = step_guard.StepGuards(), step_guard.StepGuards()

    def call(**kw):
        return training_step.optimize_batch(fa, 'batch', optimizer, {'block': 1.0}, 3, components, score=lambda *a: {}, clock=lambda: 0., **kw)

    with step_guard.trust(scope_guards):
        with pytest.raises(RuntimeError, match='StepGuards of the active'):
            call()                                                                        # a trusted scope, no guards
        with pytest.raises(RuntimeError, match='StepGuards of the active'):
            call(guards=other)                                                            # a trusted scope, other guards
    with pytest.raises(RuntimeError, match='StepGuards of the active'):
        call(guards=scope_guards)                                                         # guards, no scope
    assert encoded == [] and fa.w.grad is None


def test_a_step_with_nothing_to_refuse_runs_backward_and_the_optimizer_exactly_as_one_without_guards(monkeypatch):
    fa = a_world(monkeypatch)
    total, seen = step(fa, 0)
    first = fa.w.grad.clone()
    fa.zero_grad(set_to_none=True)
    again, seen_plain = step(fa, 0, guards=False)
    assert total.item() == again.item() == 6.0 and seen == seen_plain == ['step'] and torch.equal(fa.w.grad, first)


def test_a_refused_step_has_run_no_backward_and_no_optimizer_step_and_raises_the_message_of_the_check(monkeypatch):
    fa = a_world(monkeypatch)
    with pytest.raises(ValueError, match=r'^2 cell\(s\) of the model are not acceptable$'):
        step(fa, 2)
    assert fa.w.grad is None                                                              # no backward
    seen = []
    with pytest.raises(ValueError, match='not acceptable'):
        total, seen = step(fa, 1, finite=False)                                           # a failing guard wins over a loss that is not finite
    with pytest.raises(FloatingPointError, match='^the training loss is not finite at step 3$'):
        step(fa, 0, finite=False)
    assert fa.w.grad is None
