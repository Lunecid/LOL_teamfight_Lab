"""Software-only campaign accounting and deterministic input plumbing.

No datasets, scientific models, services or accelerator runtimes are imported.
The ledger is single-owner and single-attempt: reopening permits inspection only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import uuid


HOUR = 3600
MODELS = ("ept", "n2", "b1")
PHASE_CAPS = {"prepare": 4 * HOUR, "fit": 72 * HOUR,
              "evaluate": 24 * HOUR, "finalize": 4 * HOUR}
TOTAL_SECONDS = 104 * HOUR
PRIVATE_BYTES_LIMIT = 40 * 1024 ** 3


class CampaignError(RuntimeError):
    """An identity, accounting or single-attempt contract failed."""


class BudgetExceeded(CampaignError):
    pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise CampaignError(f"{name} must be a finite nonnegative number")
    return float(value)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


@dataclass(frozen=True)
class Reservation:
    phase: str
    model: str | None
    seconds: int


def phase_plan(models=MODELS):
    models = tuple(models)
    if len(models) != 3 or set(models) != set(MODELS):
        raise CampaignError("exactly ept, n2 and b1 are required, once each")
    return (Reservation("prepare", None, 4 * HOUR),
            *(Reservation("fit", model, 24 * HOUR) for model in models),
            *(Reservation("evaluate", model, 8 * HOUR) for model in models),
            Reservation("finalize", None, 4 * HOUR))


class CampaignLedger:
    """Atomic JSON receipts, with immutable fixed caps and no resume/retry.

    The first creator owns mutations. Existing ledgers are read-only, including
    interrupted ledgers. Never delete a failed ledger to retry the same campaign.
    Clocks are injectable for synthetic tests; production uses monotonic + UTC.
    """

    def __init__(self, path, *, models=MODELS, clock=time.monotonic, wall_clock=time.time,
                 started_at=None, started_wall=None):
        models = tuple(models)
        self.path = Path(path)
        self.clock, self.wall_clock = clock, wall_clock
        self.plan = phase_plan(models)
        self._lock = threading.RLock()
        self._owner = False
        opened = _number(clock(), "monotonic clock")
        self._opened_at = opened if started_at is None else _number(started_at, "start clock")
        created_wall = _number(wall_clock() if started_wall is None else started_wall, "wall clock")
        if self._opened_at > opened or created_wall > wall_clock():
            raise CampaignError("campaign start is in the future")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema": "ept-comparison-ledger-v1", "campaign_id": uuid.uuid4().hex,
                    "created_wall": created_wall,
                    "startup_elapsed_seconds": max(opened-self._opened_at, wall_clock()-created_wall),
                    "models": list(models), "total_cap_seconds": TOTAL_SECONDS,
                    "phase_caps_seconds": dict(PHASE_CAPS),
                    "plan": [asdict(item) for item in self.plan],
                    "state": "ready", "records": [], "revision": 0}
        try:
            with self.path.open("xb") as handle:
                self._raw = _json_bytes(document)
                handle.write(self._raw)
                handle.flush()
                os.fsync(handle.fileno())
            self._owner = True
        except FileExistsError:
            self._raw = self.path.read_bytes()
            document = json.loads(self._raw)
        self._document = document
        if (document.get("schema") != "ept-comparison-ledger-v1"
                or document.get("plan") != [asdict(item) for item in self.plan]
                or document.get("models") != list(models)
                or document.get("total_cap_seconds") != TOTAL_SECONDS
                or document.get("phase_caps_seconds") != PHASE_CAPS):
            raise CampaignError("ledger identity/plan/caps mismatch")
        _number(document["created_wall"], "created_wall")
        self._elapsed_at_open = 0. if self._owner else max(0., wall_clock() - document["created_wall"])

    def _verify(self):
        if self.path.read_bytes() != self._raw:
            raise CampaignError("ledger changed outside its owning controller")

    def _save(self):
        self._verify()
        self._document["revision"] += 1
        raw = _json_bytes(self._document)
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._raw = raw
        finally:
            temporary.unlink(missing_ok=True)

    def elapsed_seconds(self):
        if "terminal_elapsed_seconds" in self._document:
            return self._document["terminal_elapsed_seconds"]
        monotonic = _number(self.clock(), "monotonic clock") - self._opened_at
        wall = _number(self.wall_clock(), "wall clock") - self._document["created_wall"]
        if monotonic < 0 or wall < 0:
            raise CampaignError("clock moved backwards")
        charged = sum(record.get("elapsed_seconds", 0.) for record in self._document["records"])
        return max(wall, self._elapsed_at_open + monotonic, charged)

    def snapshot(self):
        with self._lock:
            self._verify()
            result = json.loads(_json_bytes(self._document))
            result["total_elapsed_seconds"] = self.elapsed_seconds()
            result["reopened_read_only"] = not self._owner
            return result


def check_worker_budget(path, *, phase, model=None, wall_clock=time.time):
    """Read-only worker-side deadline check; the parent alone mutates the ledger.

    Wall time supplements the parent's monotonic enforcement. This callback does
    not check source/config/input identity or authorize a run: compose those
    checks in the worker. Missing/stale/terminal accounting fails closed.
    """
    state = json.loads(Path(path).read_bytes())
    plan = phase_plan(state.get("models", ()))
    if (state.get("schema") != "ept-comparison-ledger-v1"
            or state.get("total_cap_seconds") != TOTAL_SECONDS
            or state.get("phase_caps_seconds") != PHASE_CAPS
            or state.get("plan") != [asdict(item) for item in plan]
            or state.get("state") != "running"):
        raise CampaignError("worker requires the active fixed-budget parent ledger")
    records = state.get("records", [])
    if not 1 <= len(records) <= len(plan):
        raise CampaignError("worker ledger record count mismatch")
    for index, record in enumerate(records):
        expected = plan[index]
        if (record.get("phase"), record.get("model")) != (expected.phase, expected.model):
            raise CampaignError("worker ledger phase order mismatch")
        if index < len(records) - 1 and record.get("state") != "completed":
            raise CampaignError("worker ledger contains a failed/unfinished prior phase")
    record = records[-1]
    if record.get("state") != "running" or (record["phase"], record["model"]) != (phase, model):
        raise CampaignError("worker does not own the active phase/model")
    now = _number(wall_clock(), "wall clock")
    elapsed = now - _number(record["started_wall"], "phase start")
    total_elapsed = now - _number(state["created_wall"], "campaign start")
    future = sum(item.seconds for item in plan[len(records):])
    limit = _number(record["limit_seconds"], "phase limit")
    if (elapsed < 0 or total_elapsed < 0 or limit > plan[len(records) - 1].seconds
            or record.get("future_reserved_seconds") != future):
        raise CampaignError("worker ledger clock/caps/reservations mismatch")
    remaining = min(limit - elapsed, TOTAL_SECONDS - total_elapsed - future)
    if remaining <= 0:
        raise BudgetExceeded("worker active phase/campaign deadline reached")
    return remaining


def worker_budget_check(path, phase, model=None):
    """Positional worker callback API; read-only, no scientific imports."""
    return check_worker_budget(path, phase=phase, model=model)


class CampaignBudget:
    """Reserve future phases before launch; charge failures and cleanup in full."""

    def __init__(self, ledger: CampaignLedger):
        self.ledger = ledger
        self.clock = ledger.clock
        self._started = None
        self._limit = None

    def begin(self, phase, model=None):
        with self.ledger._lock:
            self.ledger._verify()
            state = self.ledger._document
            if not self.ledger._owner:
                raise CampaignError("existing ledger is inspection-only; no resume/retry")
            if state["state"] != "ready" or self._started is not None:
                raise CampaignError("campaign is running, failed or complete; no resume/retry")
            index = len(state["records"])
            if index >= len(self.ledger.plan):
                raise CampaignError("campaign is complete")
            current = self.ledger.plan[index]
            if (phase, model) != (current.phase, current.model):
                raise CampaignError(f"phase order requires {current.phase}/{current.model}")
            elapsed = self.ledger.elapsed_seconds()
            future = sum(item.seconds for item in self.ledger.plan[index + 1:])
            used = sum(record["elapsed_seconds"] for record in state["records"] if record["phase"] == phase)
            available = min(current.seconds, PHASE_CAPS[phase] - used,
                            TOTAL_SECONDS - elapsed - future)
            # Preparation starts at campaign creation, including controller setup.
            if phase != "prepare" and available < current.seconds:
                raise BudgetExceeded("cannot reserve the full current and remaining phase budgets")
            if available <= 0:
                raise BudgetExceeded("campaign/phase budget exhausted before launch")
            self._started = self.clock()
            self._limit = available
            state["records"].append({"phase": phase, "model": model, "state": "running",
                                     "started_wall": self.ledger.wall_clock(),
                                     "limit_seconds": available, "future_reserved_seconds": future})
            state["state"] = "running"
            self.ledger._save()
            return available

    def check(self, *, reserve_seconds=0.):
        with self.ledger._lock:
            if self._started is None:
                raise CampaignError("no active phase")
            self.ledger._verify()
            reserve = _number(reserve_seconds, "cleanup reserve")
            elapsed = _number(self.clock() - self._started, "phase elapsed")
            future = self.ledger._document["records"][-1]["future_reserved_seconds"]
            if elapsed + reserve >= self._limit:
                raise BudgetExceeded("per-run/phase wall budget reached (cleanup included)")
            if self.ledger.elapsed_seconds() + reserve + future >= TOTAL_SECONDS:
                raise BudgetExceeded("campaign wall budget reached; future reservations protected")
            return self._limit - elapsed

    def finish(self, *, success, receipt):
        with self.ledger._lock:
            if self._started is None:
                raise CampaignError("no active phase to finish")
            elapsed = _number(self.clock() - self._started, "phase elapsed")
            record = self.ledger._document["records"][-1]
            cleanup = all(receipt.get(key) is True for key in ("tree_gone", "job_gone", "job_closed"))
            in_budget = (elapsed <= self._limit
                         and self.ledger.elapsed_seconds() + record["future_reserved_seconds"] <= TOTAL_SECONDS)
            completed = success is True and cleanup and in_budget
            saved_receipt = json.loads(_json_bytes(receipt))
            saved_receipt["state"] = "completed" if completed else "failed"
            record.update(state="completed" if completed else "failed", elapsed_seconds=elapsed,
                          ended_wall=self.ledger.wall_clock(), within_budget=in_budget,
                          cleanup_confirmed=cleanup, receipt=saved_receipt)
            state = self.ledger._document
            state["state"] = ("complete" if len(state["records"]) == len(self.ledger.plan) else "ready") if completed else "failed"
            if state["state"] in ("complete", "failed"):
                state["terminal_elapsed_seconds"] = self.ledger.elapsed_seconds()
            self.ledger._save()
            self._started = self._limit = None
            return completed

    def seal_terminal(self):
        """Charge the successful FINAL-envelope write before returning success.

        CAMPAIGN is authoritative; a FINAL ledger copy is explicitly a snapshot
        before that write. The accounting seal itself is a final cooperative I/O
        operation, not a hard real-time filesystem deadline guarantee.
        """
        with self.ledger._lock:
            self.ledger._verify()
            state=self.ledger._document
            if not self.ledger._owner or self._started is not None or state['state']!='complete':
                raise CampaignError('terminal seal requires the completed owning campaign')
            before=state.pop('terminal_elapsed_seconds')
            elapsed=self.ledger.elapsed_seconds()
            extra=max(0.,elapsed-before)
            record=state['records'][-1]
            record['elapsed_seconds']+=extra
            record['terminal_parent_seconds']=record.get('terminal_parent_seconds',0.)+extra
            record['ended_wall']=self.ledger.wall_clock()
            in_budget=(elapsed<=TOTAL_SECONDS and record['elapsed_seconds']<=PHASE_CAPS['finalize'])
            record['within_budget']=in_budget
            if not in_budget:
                record['state']=state['state']='failed'
            state['terminal_elapsed_seconds']=elapsed
            self.ledger._save()
            if not in_budget:
                raise BudgetExceeded('final marker exceeded finalization/campaign budget')
            return elapsed

    def abort(self, error, *, phase=None, model=None):
        """Latch parent/controller errors too, including after the last child.

        A completed child is evidence only: parent acceptance must already have
        run inside its active phase. No error reopens the campaign for retry.
        """
        with self.ledger._lock:
            self.ledger._verify()
            if not self.ledger._owner:
                raise CampaignError("inspection-only ledger cannot be aborted")
            state = self.ledger._document
            state.pop("terminal_elapsed_seconds", None)
            elapsed = self.ledger.elapsed_seconds()
            if self._started is not None:
                record = state["records"][-1]
                record.update(state="failed", elapsed_seconds=_number(self.clock()-self._started,"phase elapsed"),
                              ended_wall=self.ledger.wall_clock(), within_budget=False,
                              parent_error=str(error))
            state.update(state="failed", parent_failure={"error": str(error), "phase": phase,
                         "model": model, "ended_wall": self.ledger.wall_clock()},
                         terminal_elapsed_seconds=elapsed)
            self.ledger._save()
            self._started = self._limit = None


@dataclass(frozen=True)
class ResourcePlan:
    """Inventory and fixed policy, not a benchmark or continuously enforced cap."""

    gpu_name: str = "NVIDIA GeForce RTX 5080"
    gpu_memory_mib: int = 16303
    physical_cpus: int = 8
    logical_cpus: int = 16
    host_ram_bytes: int = 66206961664
    b1_native_threads: int = 8
    prefetch_batches: int = 2
    prefetch_cpu_producers: int = 1
    concurrent_gpu_fits: int = 1
    dtype: str = "float32"
    deterministic: bool = True
    amp: bool = False
    tf32: bool = False
    compile: bool = False
    autobatch: bool = False
    pinned_memory: bool = False

    def __post_init__(self):
        if (self.b1_native_threads != 8 or self.prefetch_batches != 2
                or self.prefetch_cpu_producers != 1 or self.concurrent_gpu_fits != 1
                or self.dtype != "float32" or self.deterministic is not True
                or any(flag is not False for flag in (self.amp, self.tf32, self.compile, self.autobatch))):
            raise CampaignError("resource policy must preserve the fixed float32 sequential contract")

    def describe(self):
        return {**asdict(self), "memory_limit_bytes": PRIVATE_BYTES_LIMIT,
                "memory_enforcement": "sampled child-tree private; not an OS continuous limit",
                "performance": "unmeasured", "batch_order": "unchanged"}


@dataclass(frozen=True)
class CacheIdentity:
    source_sha256: str
    subset_sha256: str
    stats_sha256: str
    inputs_sha256: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise CampaignError(f"{name} requires a lowercase SHA256")

    @property
    def digest(self):
        return hashlib.sha256(_json_bytes(asdict(self))).hexdigest()


class IdentityCache:
    """Cache immutable serialized values under one source/subset/Stats/input ID.

    Callers must hash immutable input snapshots; live datasets are not cache keys.
    Decoding belongs to the caller, so mutable arrays/Stats cannot alias the cache.
    """

    def __init__(self, identity: CacheIdentity):
        if not isinstance(identity, CacheIdentity):
            raise TypeError("CacheIdentity required")
        self._identity = identity
        self._values = {}

    @property
    def identity(self):
        return self._identity

    def _check(self, identity):
        if identity != self.identity:
            raise CampaignError("cache source/subset/Stats/input identity mismatch")

    def put(self, key: str, value: bytes, *, identity: CacheIdentity):
        self._check(identity)
        if not isinstance(key, str) or type(value) is not bytes:
            raise TypeError("cache accepts string keys and immutable bytes only")
        if key in self._values and self._values[key] != value:
            raise CampaignError("immutable cache entry cannot be replaced")
        self._values[key] = value

    def get(self, key: str, *, identity: CacheIdentity):
        self._check(identity)
        return self._values[key]


class PrefetchError(RuntimeError):
    pass


class OrderedPrefetch:
    """One CPU producer; at most two outstanding reads, in original order.

    Use as a context manager or explicitly close an abandoned iterator. A source
    whose next() blocks indefinitely cannot be forcibly interrupted: cancellation
    raises if the producer cannot join, and never starts a replacement producer.
    All source metadata, including rows/describe, is forwarded unchanged.
    """

    def __init__(self, source, max_prefetch=2, *, join_seconds=5.):
        if type(max_prefetch) is not int or not 1 <= max_prefetch <= 2:
            raise ValueError("max_prefetch must be 1 or 2")
        if _number(join_seconds, "join_seconds") <= 0:
            raise ValueError("join_seconds must be positive")
        self._source, self._size, self._join_seconds = source, max_prefetch, join_seconds
        self._producer_lock = threading.Lock()
        self._current = None

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __len__(self):
        return len(self._source)

    def __iter__(self):
        return self._start(lambda: iter(self._source))

    def epoch(self, epoch, start=0):
        """Preserve Batches.epoch's local RNG/order and resume-offset meaning.

        This is an input offset, not permission to resume a scientific campaign.
        The original epoch iterator is constructed inside the sole CPU producer.
        """
        return self._start(lambda: self._source.epoch(epoch, start=start))

    def _start(self, source_factory):
        if not self._producer_lock.acquire(blocking=False):
            raise PrefetchError("only one CPU producer may be active")
        try:
            self._current = _PrefetchIterator(self, source_factory)
            return self._current
        except BaseException:
            self._producer_lock.release()
            raise

    def close(self):
        if self._current is not None:
            self._current.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _PrefetchIterator:
    def __init__(self, owner, source_factory):
        self._owner = owner
        self._source_factory = source_factory
        self._queue = queue.Queue(maxsize=owner._size)
        self._slots = threading.Semaphore(owner._size)
        self._cancel = threading.Event()
        self._closed = False
        self._producer_error = None
        self._thread = threading.Thread(target=self._produce, name="ept-ordered-prefetch", daemon=True)
        self._thread.start()

    def _produce(self):
        iterator = None
        terminal = None
        slot_owned = False
        try:
            iterator = iter(self._source_factory())
            while not self._cancel.is_set():
                if not self._slots.acquire(timeout=.05):
                    continue
                slot_owned = True
                if self._cancel.is_set():
                    break
                try:
                    item = ("value", next(iterator))
                except StopIteration:
                    terminal = ("end", None)
                    break
                except BaseException as exc:
                    self._producer_error = exc
                    terminal = ("error", exc)
                    break
                self._queue.put_nowait(item)  # slot acquired BEFORE reading next()
                slot_owned = False
        except BaseException as exc:
            self._producer_error = exc
            terminal = ("error", exc)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as exc:
                self._producer_error = exc
                terminal = ("error", exc)
            if terminal is not None:
                while not self._cancel.is_set():
                    if slot_owned or self._slots.acquire(timeout=.05):
                        self._queue.put_nowait(terminal)
                        break

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        kind, value = self._queue.get()
        self._slots.release()
        if kind == "value":
            return value
        self.close()
        if kind == "error":
            raise value
        raise StopIteration

    def close(self):
        if self._closed:
            return
        self._cancel.set()
        self._thread.join(timeout=self._owner._join_seconds)
        if self._thread.is_alive():
            raise PrefetchError("CPU producer did not join; cancellation unconfirmed")
        self._closed = True
        self._owner._producer_lock.release()
        if self._producer_error is not None:
            raise self._producer_error
