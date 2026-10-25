"""Synthetic accounting/prefetch and fake Job/process tests; no real child."""
from __future__ import annotations

import io
import json
from pathlib import Path
import threading

import pytest

from c_ept.comparison.infrastructure import (
    BudgetExceeded, CacheIdentity, CampaignBudget, CampaignError, CampaignLedger,
    HOUR, IdentityCache, MODELS, OrderedPrefetch, PHASE_CAPS, PrefetchError,
    PRIVATE_BYTES_LIMIT, ResourcePlan, TOTAL_SECONDS, phase_plan,
    check_worker_budget,
    worker_budget_check,
)
from tools.ept_comparison_controller import run_owned


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_budget(tmp_path):
    clock = Clock()
    ledger = CampaignLedger(tmp_path / "campaign.json", clock=clock, wall_clock=clock)
    return CampaignBudget(ledger), clock


def good_cleanup():
    return {"tree_gone": True, "job_gone": True, "job_closed": True}


def complete(budget, clock, phase, model=None, seconds=1.):
    budget.begin(phase, model)
    clock.advance(seconds)
    assert budget.finish(success=True, receipt=good_cleanup())


def test_fixed_three_model_schedule_and_reservations():
    plan = phase_plan()
    assert len(plan) == 8
    assert sum(item.seconds for item in plan) == TOTAL_SECONDS == 104 * HOUR
    assert PHASE_CAPS == {"prepare": 4 * HOUR, "fit": 72 * HOUR,
                          "evaluate": 24 * HOUR, "finalize": 4 * HOUR}
    assert [item.model for item in plan if item.phase == "fit"] == list(MODELS)
    assert all(item.seconds == 24 * HOUR for item in plan if item.phase == "fit")
    assert all(item.seconds == 8 * HOUR for item in plan if item.phase == "evaluate")
    for models in (("ept", "n2"), ("ept", "ept", "b1"), ("ept", "n2", "other")):
        with pytest.raises(CampaignError):
            phase_plan(models)


def test_full_ordered_campaign_persists_and_terminal_time_freezes(tmp_path):
    budget, clock = make_budget(tmp_path)
    for item in phase_plan():
        complete(budget, clock, item.phase, item.model, seconds=2.)
    saved = json.loads(budget.ledger.path.read_text(encoding="utf-8"))
    assert saved["state"] == "complete"
    assert [row["elapsed_seconds"] for row in saved["records"]] == [2.] * 8
    assert saved["records"][0]["future_reserved_seconds"] == 100 * HOUR
    elapsed = budget.ledger.elapsed_seconds()
    clock.advance(100 * HOUR)
    assert budget.ledger.elapsed_seconds() == elapsed == 16.
    reopened = CampaignLedger(budget.ledger.path, clock=clock, wall_clock=clock)
    assert reopened.snapshot()["reopened_read_only"]
    assert reopened.snapshot()["state"] == "complete"
    with pytest.raises(CampaignError, match="inspection-only"):
        CampaignBudget(reopened).begin("prepare")


def test_no_resume_unfinished_or_retry_failed(tmp_path):
    budget, clock = make_budget(tmp_path)
    budget.begin("prepare")
    reopened = CampaignLedger(budget.ledger.path, clock=clock, wall_clock=clock)
    with pytest.raises(CampaignError, match="resume/retry"):
        CampaignBudget(reopened).begin("prepare")
    clock.advance(3)
    assert not budget.finish(success=False, receipt=good_cleanup())
    assert budget.ledger.snapshot()["records"][0]["elapsed_seconds"] == 3
    with pytest.raises(CampaignError, match="resume/retry"):
        budget.begin("prepare")


def test_phase_order_and_one_active_phase(tmp_path):
    budget, clock = make_budget(tmp_path)
    with pytest.raises(CampaignError, match="phase order"):
        budget.begin("fit", "ept")
    budget.begin("prepare")
    with pytest.raises(CampaignError, match="running"):
        budget.begin("prepare")
    clock.advance(1)
    budget.finish(success=True, receipt=good_cleanup())
    with pytest.raises(CampaignError, match="phase order"):
        budget.begin("fit", "b1")
    with pytest.raises(CampaignError, match="phase order"):
        budget.begin("evaluate", "ept")


def test_setup_time_and_failed_cleanup_are_charged(tmp_path):
    budget, clock = make_budget(tmp_path)
    clock.advance(17)
    limit = budget.begin("prepare")
    assert limit == 4 * HOUR - 17
    clock.advance(5)
    receipt = good_cleanup()
    receipt["job_closed"] = False
    assert not budget.finish(success=True, receipt=receipt)
    snapshot = budget.ledger.snapshot()
    assert snapshot["total_elapsed_seconds"] == 22
    assert snapshot["records"][0]["elapsed_seconds"] == 5
    assert snapshot["records"][0]["cleanup_confirmed"] is False
    assert snapshot["records"][0]["receipt"]["state"] == "failed"


def test_per_fit_deadline_includes_cleanup_and_failed_run(tmp_path):
    budget, clock = make_budget(tmp_path)
    complete(budget, clock, "prepare")
    budget.begin("fit", "ept")
    clock.advance(24 * HOUR - 9)
    with pytest.raises(BudgetExceeded, match="cleanup included"):
        budget.check(reserve_seconds=10)
    clock.advance(11)
    assert not budget.finish(success=False, receipt=good_cleanup())
    row = budget.ledger.snapshot()["records"][-1]
    assert row["elapsed_seconds"] == 24 * HOUR + 2
    assert row["within_budget"] is False


def test_phase_reserves_cannot_be_transferred_to_fit(tmp_path):
    budget, clock = make_budget(tmp_path)
    complete(budget, clock, "prepare", seconds=1)
    complete(budget, clock, "fit", "ept", seconds=1)
    complete(budget, clock, "fit", "n2", seconds=1)
    assert budget.begin("fit", "b1") == 24 * HOUR
    clock.advance(24 * HOUR + 1)
    with pytest.raises(BudgetExceeded):
        budget.check()
    assert not budget.finish(success=True, receipt=good_cleanup())


def test_evaluation_per_run_cap_and_final_reserve(tmp_path):
    budget, clock = make_budget(tmp_path)
    complete(budget, clock, "prepare")
    for model in MODELS:
        complete(budget, clock, "fit", model)
    assert budget.begin("evaluate", "ept") == 8 * HOUR
    assert budget.ledger.snapshot()["records"][-1]["future_reserved_seconds"] == 20 * HOUR
    clock.advance(8 * HOUR + 1)
    assert not budget.finish(success=True, receipt=good_cleanup())


def test_idle_wall_time_protects_all_future_reservations(tmp_path):
    budget, clock = make_budget(tmp_path)
    complete(budget, clock, "prepare", seconds=1)
    clock.advance(4 * HOUR)
    with pytest.raises(BudgetExceeded, match="remaining phase budgets"):
        budget.begin("fit", "ept")


def test_ledger_tamper_and_snapshot_mutation(tmp_path):
    budget, _ = make_budget(tmp_path)
    snapshot = budget.ledger.snapshot()
    snapshot["phase_caps_seconds"]["fit"] = 999999
    assert budget.ledger.snapshot()["phase_caps_seconds"]["fit"] == 72 * HOUR
    budget.ledger.path.write_text("{}", encoding="utf-8")
    with pytest.raises(CampaignError, match="changed outside"):
        budget.begin("prepare")


def test_read_only_worker_budget_client_and_wrong_phase(tmp_path):
    budget, clock = make_budget(tmp_path)
    with pytest.raises(CampaignError, match="active"):
        check_worker_budget(budget.ledger.path, phase="prepare", wall_clock=clock)
    budget.begin("prepare")
    clock.advance(7)
    assert check_worker_budget(budget.ledger.path, phase="prepare", wall_clock=clock) == 4 * HOUR - 7
    with pytest.raises(CampaignError, match="active phase/model"):
        check_worker_budget(budget.ledger.path, phase="fit", model="ept", wall_clock=clock)
    clock.advance(4 * HOUR)
    with pytest.raises(BudgetExceeded):
        check_worker_budget(budget.ledger.path, phase="prepare", wall_clock=clock)


def test_finish_rejects_wall_time_that_consumes_future_reserves(tmp_path):
    budget, clock = make_budget(tmp_path)
    budget.begin("prepare")
    # A wall-clock jump can consume future reservations even if monotonic phase
    # time remains short. The final receipt must fail too, without relying on check().
    budget.ledger.wall_clock = lambda: clock.now + 5 * HOUR
    assert not budget.finish(success=True, receipt=good_cleanup())
    assert budget.ledger.snapshot()["records"][0]["within_budget"] is False


def test_positional_worker_callback_and_lead_model_order(tmp_path):
    ledger = CampaignLedger(tmp_path / "lead-order.json", models=("b1", "ept", "n2"))
    budget = CampaignBudget(ledger)
    budget.begin("prepare")
    before = ledger.path.read_bytes()
    assert 0 < worker_budget_check(ledger.path, "prepare", None) <= 4 * HOUR
    assert ledger.path.read_bytes() == before
    assert budget.finish(success=True, receipt=good_cleanup())
    assert budget.begin("fit", "b1") <= 24 * HOUR
    assert worker_budget_check(ledger.path, "fit", "b1") > 0
    assert budget.finish(success=True, receipt=good_cleanup())


def test_resource_policy_and_immutable_identity_cache():
    resource = ResourcePlan().describe()
    assert resource["b1_native_threads"] == 8
    assert resource["concurrent_gpu_fits"] == 1
    assert resource["prefetch_cpu_producers"] == 1
    assert resource["prefetch_batches"] == 2
    assert resource["dtype"] == "float32"
    assert "sampled" in resource["memory_enforcement"]
    assert "not an OS continuous" in resource["memory_enforcement"]
    for kwargs in ({"amp": True}, {"tf32": True}, {"compile": True}, {"autobatch": True},
                   {"concurrent_gpu_fits": 2}, {"prefetch_batches": 3}, {"b1_native_threads": 16}):
        with pytest.raises(CampaignError):
            ResourcePlan(**kwargs)
    identity = CacheIdentity("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    cache = IdentityCache(identity)
    with pytest.raises(AttributeError):
        cache.identity = CacheIdentity("e" * 64, "b" * 64, "c" * 64, "d" * 64)
    cache.put("stats", b"immutable", identity=identity)
    assert cache.get("stats", identity=identity) == b"immutable"
    assert len(identity.digest) == 64
    with pytest.raises(TypeError):
        cache.put("mutable", bytearray(b"x"), identity=identity)
    with pytest.raises(CampaignError, match="cannot be replaced"):
        cache.put("stats", b"changed", identity=identity)
    for index in range(4):
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        hashes[index] = "e" * 64
        changed = CacheIdentity(*hashes)
        with pytest.raises(CampaignError, match="identity mismatch"):
            cache.get("stats", identity=changed)
        with pytest.raises(CampaignError, match="identity mismatch"):
            cache.put("value", b"new", identity=changed)
    with pytest.raises(CampaignError):
        CacheIdentity("mutable-path", "b" * 64, "c" * 64, "d" * 64)


class Rows:
    def __init__(self, count=12):
        self.rows = [{"index": i} for i in range(count)]
        self.metadata = {"source": "synthetic"}

    def describe(self):
        return self.metadata

    def __iter__(self):
        yield from self.rows


def test_ordered_prefetch_matches_serial_and_forwards_original_metadata():
    source = Rows()
    with OrderedPrefetch(source) as prefetched:
        assert prefetched.rows is source.rows
        assert prefetched.describe() is source.describe()
        values = list(prefetched)
        assert values == list(source)
        assert all(a is b for a, b in zip(values, source.rows))
        assert list(prefetched) == source.rows


def test_epoch_api_preserves_metadata_offset_values_and_local_rng():
    import random

    class Batches(Rows):
        training = True
        batch_size = 4
        seed = 20261022
        stats = object()

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            raise AssertionError("trainer must call epoch, not iter(fit)")

        def epoch(self, epoch, start=0):
            order = list(range(len(self.rows)))
            rng = random.Random((self.seed << 32) + epoch)
            rng.shuffle(order)
            for index in order[start:]:
                yield self.rows[index]

    source = Batches()
    before = random.getstate()
    with OrderedPrefetch(source) as wrapped:
        assert len(wrapped) == len(source)
        assert wrapped.rows is source.rows and wrapped.stats is source.stats
        assert (wrapped.training, wrapped.batch_size, wrapped.seed) == (True, 4, 20261022)
        for epoch in range(3):
            assert list(wrapped.epoch(epoch)) == list(source.epoch(epoch))
            assert list(wrapped.epoch(epoch, start=3)) == list(source.epoch(epoch, start=3))
        active = wrapped.epoch(3)
        with pytest.raises(PrefetchError, match="one CPU producer"):
            wrapped.epoch(4)
        active.close()
    assert random.getstate() == before


def test_prefetch_reserves_slot_before_read_no_third_read_and_one_producer():
    observed = []
    ready = threading.Event()

    def values():
        for value in range(100):
            observed.append(value)
            if value == 1:
                ready.set()
            yield value

    with OrderedPrefetch(values()) as prefetched:
        iterator = iter(prefetched)
        assert ready.wait(2)
        assert observed == [0, 1]
        with pytest.raises(PrefetchError, match="one CPU producer"):
            iter(prefetched)
        assert next(iterator) == 0
        iterator.close()
        assert not iterator._thread.is_alive()
        assert len(observed) <= 3


@pytest.mark.parametrize("max_prefetch", [0, 3, -1, True, 2.5])
def test_invalid_prefetch_bound(max_prefetch):
    with pytest.raises(ValueError):
        OrderedPrefetch([], max_prefetch=max_prefetch)


def test_prefetch_source_exception_is_not_hidden():
    error = RuntimeError("synthetic producer failure")

    def values():
        yield "first"
        raise error

    prefetched = OrderedPrefetch(values())
    iterator = iter(prefetched)
    assert next(iterator) == "first"
    with pytest.raises(RuntimeError, match="synthetic producer failure") as found:
        next(iterator)
    assert found.value is error
    assert not iterator._thread.is_alive()


def test_prefetch_iter_failure_and_close_failure_propagate():
    class BrokenIter:
        def __iter__(self):
            raise ValueError("cannot create iterator")

    with pytest.raises(ValueError, match="cannot create iterator"):
        list(OrderedPrefetch(BrokenIter()))

    class BrokenClose:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise ValueError("cannot close iterator")

    with pytest.raises(ValueError, match="cannot close iterator"):
        list(OrderedPrefetch(BrokenClose()))


def test_prefetch_cancellation_joins_and_runs_source_finally():
    closed = threading.Event()

    def values():
        try:
            yield from range(10000)
        finally:
            closed.set()

    prefetched = OrderedPrefetch(values())
    iterator = iter(prefetched)
    assert next(iterator) == 0
    prefetched.close()
    assert closed.is_set()
    assert not iterator._thread.is_alive()
    assert list(iterator) == []


def test_prefetch_blocking_source_reports_unconfirmed_join_without_replacement():
    entered, unblock = threading.Event(), threading.Event()

    def blocking():
        entered.set()
        unblock.wait(2)
        yield 1

    prefetched = OrderedPrefetch(blocking(), join_seconds=.01)
    iterator = iter(prefetched)
    assert entered.wait(1)
    try:
        with pytest.raises(PrefetchError, match="did not join"):
            iterator.close()
        with pytest.raises(PrefetchError, match="one CPU producer"):
            iter(prefetched)
    finally:
        unblock.set()
        iterator._thread.join(2)
        iterator.close()
    assert not iterator._thread.is_alive()


class FakeProcess:
    def __init__(self, events, clock, *, exit_after=2, exit_code=0):
        self.events, self.clock = events, clock
        self.stdin = io.BytesIO()
        self.pid = 12345
        self.running = True
        self.released = False
        self.exit_after, self.exit_code = exit_after, exit_code
        self.samples = 0
        self.code = None

    def poll(self):
        if self.released and self.samples >= self.exit_after and self.running:
            self.running = False
            self.code = self.exit_code
        return None if self.running else self.code

    def kill(self):
        self.events.append("kill-exact-popen")
        self.running, self.code = False, 124

    def wait(self, timeout):
        self.events.append("wait")
        if self.poll() is None:
            raise TimeoutError("synthetic process still live")
        return self.code


class FakeJob:
    def __init__(self, events, clock, process, *, assign_error=False, stop_error=False,
                 close_error=False, stuck=False, descendants=False, cleanup_time=2.):
        self.events, self.clock, self.process = events, clock, process
        self.handle = 1
        self.assigned = False
        self.assign_error, self.stop_error = assign_error, stop_error
        self.close_error, self.stuck = close_error, stuck
        self.descendants, self.cleanup_time = descendants, cleanup_time

    def assign(self, process):
        assert process is self.process and not process.released
        self.events.append("assign")
        if self.assign_error:
            raise OSError("synthetic assignment failure")
        self.assigned = True

    def release(self, process):
        assert self.assigned and process is self.process
        process.stdin.write(b"G")
        assert process.stdin.getvalue() == b"G"
        process.stdin.close()
        process.released = True
        self.events.append("G")

    def gone(self):
        return not self.stuck and not self.descendants and (not self.assigned or not self.process.running)

    def stop(self, timeout):
        self.events.append("stop-job")
        self.clock.advance(min(timeout, self.cleanup_time))
        if self.stop_error:
            raise OSError("synthetic stop failure")
        if not self.stuck:
            self.process.running, self.process.code = False, 124
            self.descendants = False
        return self.gone()

    def close(self):
        self.events.append("close-job")
        if self.close_error:
            raise OSError("synthetic CloseHandle failure")
        self.handle = None
        self.process.running, self.process.code = False, self.process.code or 124


def command(phase="prepare", model=None):
    result = ["C:/Users/todtj/anaconda3/python.exe", "-B", "-m", "c_ept.comparison.operator",
              "--config", "synthetic-only.json", "--action", "Worker", "--stage", phase]
    if model is not None:
        result += ["--model", model]
    return result + ["--await-job-binding"]


def fake_run(tmp_path, *, job_kwargs=None, process_kwargs=None, memory=None, check=None, poll_seconds=1.,
             before=None,accept=None):
    budget, clock = make_budget(tmp_path)
    events = []
    process = FakeProcess(events, clock, **(process_kwargs or {}))
    job = FakeJob(events, clock, process, **(job_kwargs or {}))

    def launch(args, **kwargs):
        assert args == command()
        assert kwargs["stdin"] is not None and kwargs["shell"] is False
        assert not process.released
        events.append("popen-dormant")
        return process

    def sample(current_job):
        assert current_job is job and job.assigned and process.released
        events.append("sample")
        process.samples += 1
        if isinstance(memory, BaseException):
            raise memory
        return 123 if memory is None else memory

    def guard():
        events.append("check")
        if check is not None:
            check(events)

    receipt = run_owned(command(), cwd=tmp_path, budget=budget, phase="prepare", check=guard,
                        job_factory=lambda: job, popen_factory=launch, memory_sampler=sample,
                        sleep=clock.advance, poll_seconds=poll_seconds,
                        before=(lambda:before(clock,events)) if before else None,
                        accept=(lambda:accept(clock,events)) if accept else None)
    return receipt, events, budget, clock, process, job


def test_owned_gate_assigns_before_G_and_persists_exact_cleanup(tmp_path):
    receipt, events, budget, _, _, _ = fake_run(tmp_path)
    assert events.index("popen-dormant") < events.index("assign") < events.index("G") < events.index("sample")
    assert events[events.index("assign") + 1] == "check"
    assert receipt["state"] == "completed"
    assert receipt["returncode"] == 0
    assert receipt["private_sample_count"] == 2
    assert receipt["peak_sampled_private_bytes"] == 123
    assert all(receipt[key] is True for key in good_cleanup())
    assert "stop-job" not in events and "kill-exact-popen" not in events
    assert budget.ledger.snapshot()["records"][0]["receipt"] == receipt


def test_parent_acceptance_is_charged_after_cleanup_before_terminal_finish(tmp_path):
    def before(clock,events):
        events.append('parent-boundary');clock.advance(7)
    def accept(clock,events):
        assert events[-1]=='check' and 'close-job' in events
        events.append('parent-receipt');clock.advance(13)
    receipt,events,budget,_,_,_=fake_run(tmp_path,before=before,accept=accept)
    assert receipt['state']=='completed'
    assert events.index('parent-boundary')<events.index('G')<events.index('parent-receipt')
    assert receipt['elapsed_seconds']==21
    assert budget.ledger.snapshot()['records'][0]['elapsed_seconds']==21


@pytest.mark.parametrize('failure',('overbudget','receipt'))
def test_parent_acceptance_cannot_escape_budget_or_succeed_after_invalid_receipt(tmp_path,failure):
    def accept(clock,events):
        if failure=='overbudget':clock.advance(TOTAL_SECONDS+1)
        else:raise ValueError('synthetic invalid artifact')
    receipt,_,budget,_,_,_=fake_run(tmp_path,accept=accept)
    assert receipt['state']=='failed' and budget.ledger.snapshot()['state']=='failed'
    assert receipt['tree_gone'] and receipt['job_closed']
    assert ('budget' if failure=='overbudget' else 'invalid artifact') in receipt['error']


def test_startup_preflight_is_charged_against_preparation_reserve(tmp_path):
    clock=Clock();began=clock();clock.advance(30)
    ledger=CampaignLedger(tmp_path/'startup.json',clock=clock,wall_clock=clock,
                          started_at=began,started_wall=began)
    budget=CampaignBudget(ledger)
    assert ledger.elapsed_seconds()==30
    assert budget.begin('prepare')==4*HOUR-30
    clock.advance(4*HOUR-30)
    with pytest.raises(BudgetExceeded):budget.check()


@pytest.mark.parametrize('location',('between','after_last','active'))
def test_parent_failure_latches_durable_ledger_and_charges_elapsed(tmp_path,location):
    budget,clock=make_budget(tmp_path)
    if location=='after_last':
        for item in phase_plan():complete(budget,clock,item.phase,item.model)
    elif location=='between':complete(budget,clock,'prepare')
    else:budget.begin('prepare')
    clock.advance(19)
    budget.abort('synthetic parent failure',phase='finalize' if location=='after_last' else 'prepare')
    saved=json.loads(budget.ledger.path.read_text())
    assert saved['state']=='failed' and saved['parent_failure']['error']=='synthetic parent failure'
    assert saved['terminal_elapsed_seconds']==19+(8 if location=='after_last' else 1 if location=='between' else 0)
    if location=='active':assert saved['records'][-1]['state']=='failed'
    with pytest.raises(CampaignError,match='no resume/retry'):budget.begin('fit','ept')


@pytest.mark.parametrize('marker_seconds',(13,5*HOUR,TOTAL_SECONDS+1))
def test_successful_final_marker_io_is_charged_and_cannot_borrow_from_other_phases(tmp_path,marker_seconds):
    budget,clock=make_budget(tmp_path)
    for item in phase_plan():complete(budget,clock,item.phase,item.model)
    clock.advance(marker_seconds)
    if marker_seconds>4*HOUR:
        with pytest.raises(BudgetExceeded,match='final marker'):budget.seal_terminal()
    else:assert budget.seal_terminal()==8+marker_seconds
    saved=budget.ledger.snapshot()
    assert saved['total_elapsed_seconds']==8+marker_seconds
    assert saved['records'][-1]['terminal_parent_seconds']==marker_seconds
    assert saved['state']==('complete' if marker_seconds<4*HOUR else 'failed')


def test_assignment_failure_never_releases_and_kills_only_launch_handle(tmp_path):
    receipt, events, budget, _, _, _ = fake_run(tmp_path, job_kwargs={"assign_error": True})
    assert receipt["state"] == "failed"
    assert "G" not in events and "sample" not in events
    assert "kill-exact-popen" in events and "close-job" in events
    assert all(receipt[key] is True for key in good_cleanup())
    assert budget.ledger.snapshot()["state"] == "failed"


def test_identity_failure_before_G_stops_owned_dormant_child(tmp_path):
    def check(events):
        if "assign" in events:
            raise CampaignError("synthetic source drift")
    receipt, events, _, _, _, _ = fake_run(tmp_path, check=check)
    assert "G" not in events
    assert "synthetic source drift" in receipt["error"]
    assert receipt["tree_gone"] is True


def test_failed_preflight_launches_nothing_and_is_retained(tmp_path):
    def check(events):
        raise CampaignError("future authorization absent")
    receipt, events, budget, _, _, _ = fake_run(tmp_path, check=check)
    assert "popen-dormant" not in events
    assert receipt["state"] == "failed"
    assert all(receipt[key] is True for key in good_cleanup())
    assert budget.ledger.snapshot()["state"] == "failed"


@pytest.mark.parametrize("memory", [PRIVATE_BYTES_LIMIT, PRIVATE_BYTES_LIMIT + 1, -1, True,
                                    None, OSError("synthetic sample unavailable")])
def test_sampled_memory_failure_stops_whole_job(tmp_path, memory):
    # None means normal memory in fake_run; use a separate unavailable-None test below.
    if memory is None:
        memory = float("nan")
    receipt, events, budget, _, _, _ = fake_run(tmp_path, memory=memory)
    assert receipt["state"] == "failed"
    assert "stop-job" in events and "close-job" in events
    assert receipt["elapsed_seconds"] >= 2.  # cleanup is charged
    assert all(receipt[key] is True for key in good_cleanup())
    assert budget.ledger.snapshot()["records"][0]["elapsed_seconds"] >= 2.


def test_timeout_stops_job_and_cleanup_counts_against_same_phase(tmp_path):
    receipt, events, budget, _, _, _ = fake_run(tmp_path, process_kwargs={"exit_after": 999999},
                                              poll_seconds=4 * HOUR)
    assert receipt["state"] == "failed" and "BudgetExceeded" in receipt["error"]
    assert "stop-job" in events
    row = budget.ledger.snapshot()["records"][0]
    assert row["elapsed_seconds"] == 4 * HOUR - 8
    assert receipt["cleanup_seconds"] == 2.


def test_worker_nonzero_never_retries(tmp_path):
    receipt, events, budget, _, _, _ = fake_run(tmp_path, process_kwargs={"exit_code": 7})
    assert receipt["state"] == "failed" and receipt["returncode"] == 7
    assert events.count("popen-dormant") == 1
    with pytest.raises(CampaignError, match="resume/retry"):
        budget.begin("prepare")


@pytest.mark.parametrize("kwargs", [{"close_error": True}, {"stuck": True}, {"stop_error": True}])
def test_cleanup_uncertainty_cannot_be_success_or_disappear_from_ledger(tmp_path, kwargs):
    receipt, events, budget, _, _, _ = fake_run(tmp_path, job_kwargs=kwargs,
                                              memory=PRIVATE_BYTES_LIMIT)
    assert receipt["state"] == "failed"
    assert "manual_attention" in receipt
    assert "close-job" in events
    assert not all(receipt[key] is True for key in good_cleanup())
    assert budget.ledger.snapshot()["records"][0]["receipt"] == receipt


def test_orphan_descendant_is_stopped_and_not_reported_as_clean_success(tmp_path):
    receipt, events, _, _, _, _ = fake_run(tmp_path, job_kwargs={"descendants": True})
    assert receipt["state"] == "failed"
    assert "active Job members" in receipt["error"]
    assert "stop-job" in events
    assert all(receipt[key] is True for key in good_cleanup())


@pytest.mark.parametrize("stage", ["job", "popen"])
def test_creation_failures_leave_failed_persistent_receipts_without_G(tmp_path, stage):
    budget, clock = make_budget(tmp_path)
    events = []
    process = FakeProcess(events, clock)
    job = FakeJob(events, clock, process)

    def create_job():
        if stage == "job":
            raise OSError("synthetic Job construction failure")
        return job

    def create_child(*args, **kwargs):
        raise OSError("synthetic Popen failure")

    receipt = run_owned(command(), cwd=tmp_path, budget=budget, phase="prepare", check=lambda: None,
                        job_factory=create_job, popen_factory=create_child, sleep=clock.advance)
    assert receipt["state"] == "failed"
    assert not receipt["released"] and not receipt["job_assigned"]
    assert budget.ledger.snapshot()["records"][0]["receipt"] == receipt
    if stage == "popen":
        assert all(receipt[key] is True for key in good_cleanup())
        assert events == ["close-job"]
    else:
        assert "manual_attention" in receipt  # constructor did not return ownership evidence


def test_cleanup_overrun_is_charged_and_cannot_be_completed(tmp_path):
    budget, clock = make_budget(tmp_path)
    events = []
    process = FakeProcess(events, clock, exit_after=999999)
    job = FakeJob(events, clock, process)

    def sample(current_job):
        process.samples += 1
        raise CampaignError("synthetic stop request")

    def stop(timeout):
        clock.advance(4 * HOUR + 1)  # faulty cleanup exceeds its promised bound
        process.running, process.code = False, 124
        return True

    job.stop = stop
    receipt = run_owned(command(), cwd=tmp_path, budget=budget, phase="prepare", check=lambda: None,
                        job_factory=lambda: job, popen_factory=lambda *args, **kwargs: process,
                        memory_sampler=sample, sleep=clock.advance)
    assert receipt["state"] == "failed"
    assert receipt["elapsed_seconds"] == 4 * HOUR + 1
    row = budget.ledger.snapshot()["records"][0]
    assert row["elapsed_seconds"] == receipt["elapsed_seconds"]
    assert row["within_budget"] is False


@pytest.mark.parametrize("mutate", [
    lambda args: args[:-1],
    lambda args: ["python"] + args[1:],
    lambda args: [item if item != "Worker" else "Start" for item in args],
    lambda args: args + ["--await-job-binding"],
    lambda args: args + ["--other", "value"],
])
def test_ungated_nonliteral_or_wrong_worker_commands_are_refused(tmp_path, mutate):
    budget, _ = make_budget(tmp_path)
    with pytest.raises(ValueError):
        run_owned(mutate(command()), cwd=tmp_path, budget=budget, phase="prepare", check=lambda: None)
    assert budget.ledger.snapshot()["records"] == []
