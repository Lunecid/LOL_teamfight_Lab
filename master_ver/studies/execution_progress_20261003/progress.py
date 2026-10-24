"""Small standard-library progress state writer and terminal watch command.

The writer owns one JSON state file for one run.  A state update is committed
with a temporary file and ``os.replace`` so a reader sees either the previous
complete document or the next complete document.  ETA values are deliberately
fail-closed: they use observed count increases and active time only, and are
``None`` when the observations or the stage plan do not support an estimate.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
import uuid


STATE_SCHEMA = "execution_progress_20261003_v1"
DEFAULT_TERMINAL_STATUSES = frozenset(
    {"completed", "success", "succeeded", "failed", "cancelled", "canceled", "stopped"}
)
_UNSET = object()
_DEFAULT_STALE_AFTER_SECONDS = 120.0
_MAX_OBSERVATIONS = 512
_ATOMIC_REPLACE_RETRY_WINERRORS = frozenset({5, 32})
_ATOMIC_REPLACE_RETRY_ATTEMPTS = 41
_ATOMIC_REPLACE_RETRY_DELAY_SECONDS = 0.05


class ProgressStateError(ValueError):
    """A progress update would violate the persisted state contract."""


class StateReadError(RuntimeError):
    """The state file could not be read as one complete JSON document."""


def utc_now() -> str:
    """Return an ASCII UTC timestamp suitable for state JSON."""

    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _coerce_utc_timestamp(value: Any) -> _datetime.datetime:
    if isinstance(value, _datetime.datetime):
        parsed = value
    else:
        if not isinstance(value, str) or not value:
            raise ValueError("timestamp must be a non-empty ISO-8601 string")
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = _datetime.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
    return parsed.astimezone(_datetime.timezone.utc)


def _normalise_utc(value: Any) -> str:
    if isinstance(value, _datetime.datetime):
        parsed = value
    else:
        parsed = _coerce_utc_timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
    return (
        parsed.astimezone(_datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete JSON document and replace the destination atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(_ATOMIC_REPLACE_RETRY_ATTEMPTS):
            try:
                os.replace(str(temporary), str(path))
                break
            except OSError as exc:
                if (
                    getattr(exc, "winerror", None) not in _ATOMIC_REPLACE_RETRY_WINERRORS
                    or attempt + 1 >= _ATOMIC_REPLACE_RETRY_ATTEMPTS
                ):
                    raise
                time.sleep(_ATOMIC_REPLACE_RETRY_DELAY_SECONDS)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_state(
    path: Union[Path, str],
    *,
    retries: int = 3,
    retry_delay: float = 0.02,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Read an atomically written state document, retrying transient reads.

    A malformed or disappearing file never becomes a successful empty state.
    ``StateReadError`` retains the final exception text for the watch command.
    """

    target = Path(path)
    attempts = max(0, int(retries)) + 1
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            with target.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError("state JSON root must be an object")
            return value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts and retry_delay > 0:
                sleep(float(retry_delay))
    raise StateReadError("could not read complete state %s: %s" % (target, last_error)) from last_error


def _positive_pid(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProgressStateError("pid must be a positive integer")
    return int(value)


def is_pid_alive(pid: Any) -> bool:
    """Return a best-effort process liveness result on Windows and POSIX."""

    try:
        value = _positive_pid(pid)
    except ProgressStateError:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) sends CTRL_C_EVENT on Windows; never use it as a probe.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, value)  # SYNCHRONIZE only
        if not handle:
            return ctypes.get_last_error() == 5  # access denied still implies existence
        try:
            result = kernel.WaitForSingleObject(handle, 0)
            if result == 0xFFFFFFFF:
                raise OSError(ctypes.get_last_error(), "process liveness wait failed")
            return result == 0x00000102  # WAIT_TIMEOUT: not signalled/exited
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(value, 0)
    except PermissionError:
        # Permission to signal is enough evidence that the process exists.
        return True
    except OSError as exc:
        # Windows reports access denied as winerror 5 in some Python builds.
        if getattr(exc, "winerror", None) == 5:
            return True
        return False
    return True


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgressStateError("%s must be a non-negative number" % name)
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ProgressStateError("%s must be a non-negative finite number" % name)
    return result


def _unit_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgressStateError("%s must be a non-negative integer" % name)
    return int(value)


def _validate_counts(completed: Any, total: Any) -> Tuple[int, Optional[int]]:
    done = _unit_count(completed, "completed")
    if total is None:
        return done, None
    ceiling = _unit_count(total, "total")
    if done > ceiling:
        raise ProgressStateError("completed cannot exceed total")
    return done, ceiling


def _normalise_stage_plan(
    stage_plan: Optional[Mapping[str, Any]],
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    if stage_plan is None:
        return [], {}
    if not isinstance(stage_plan, Mapping) or not stage_plan:
        raise ProgressStateError("stage_plan must be a non-empty mapping when supplied")
    order: List[str] = []
    estimates: Dict[str, Optional[float]] = {}
    for raw_name, raw_value in stage_plan.items():
        name = str(raw_name).strip()
        if not name:
            raise ProgressStateError("stage names must be non-empty")
        if name in estimates:
            raise ProgressStateError("stage names must be unique")
        if isinstance(raw_value, Mapping):
            raw_value = raw_value.get("estimated_seconds")
        if raw_value is None:
            estimate = None
        else:
            estimate = _nonnegative_number(raw_value, "stage estimate")
        order.append(name)
        estimates[name] = estimate
    return order, estimates


def _empty_metrics(timestamp: str) -> Dict[str, Any]:
    return {
        "active_seconds": 0.0,
        "observations": [
            {"active_seconds": 0.0, "completed": 0, "recorded_at_utc": timestamp}
        ],
    }


def _error_payload(error: Any, traceback_text: Optional[str] = None) -> Dict[str, str]:
    if isinstance(error, BaseException):
        message = str(error)
        error_type = type(error).__name__
        if traceback_text is None:
            if error.__traceback__ is not None:
                traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            else:
                traceback_text = "".join(traceback.format_stack()) + "%s: %s\n" % (error_type, message)
    else:
        message = str(error)
        error_type = "RuntimeError"
        if traceback_text is None:
            traceback_text = "".join(traceback.format_stack()) + "%s: %s\n" % (error_type, message)
    return {
        "type": error_type,
        "message": message,
        "traceback": str(traceback_text or ""),
    }


class ProgressStateWriter:
    """Persist one run's progress state with validated monotonic updates.

    The constructor creates a new state file immediately.  It refuses to
    overwrite an existing path; callers start a fresh run with a fresh path.
    ``stage_plan`` is an ordered mapping from stage name to a known future
    duration in seconds, or ``None`` when that future duration is unknown.
    """

    def __init__(
        self,
        path: Union[Path, str],
        *,
        run_id: str,
        stage: str,
        total_units: Optional[int] = None,
        pid: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        utc_clock: Callable[[], Any] = utc_now,
        stage_plan: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise ProgressStateError("state path already exists: %s" % self.path)
        self.run_id = str(run_id).strip()
        self.stage = str(stage).strip()
        if not self.run_id:
            raise ProgressStateError("run_id must be non-empty")
        if not self.stage:
            raise ProgressStateError("stage must be non-empty")
        self._clock = clock
        self._utc_clock = utc_clock
        self._last_mono = float(clock())
        self._elapsed_seconds = 0.0
        plan_order, estimates = _normalise_stage_plan(stage_plan)
        if stage_plan is not None and self.stage not in plan_order:
            raise ProgressStateError("initial stage is missing from stage_plan")
        if self.stage not in plan_order:
            plan_order.append(self.stage)
            estimates[self.stage] = None
        completed, total = _validate_counts(0, total_units)
        timestamp = _normalise_utc(utc_clock())
        self._state: Dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "stage": self.stage,
            "stage_order": plan_order,
            "stage_plan_complete": stage_plan is not None,
            "stage_estimates_seconds": estimates,
            "status": "running",
            "pid": _positive_pid(os.getpid() if pid is None else pid),
            "completed": completed,
            "total": total,
            "started_at_utc": timestamp,
            "updated_at_utc": timestamp,
            "finished_at_utc": None,
            "elapsed_seconds": 0.0,
            "checkpoint": None,
            "last_error": None,
            "stage_metrics": {name: _empty_metrics(timestamp) for name in plan_order if name == self.stage},
        }
        self._write()

    @property
    def state(self) -> Dict[str, Any]:
        """Return a detached copy of the most recently committed state."""

        return copy.deepcopy(self._state)

    def _write(self) -> None:
        _atomic_write_json(self.path, self._state)

    def _now(self) -> Tuple[float, str]:
        return float(self._clock()), _normalise_utc(self._utc_clock())

    def _advance_time(self, state: Dict[str, Any], now_mono: float, *, active: bool) -> None:
        delta = max(0.0, now_mono - self._last_mono)
        self._last_mono = now_mono
        if state.get("status") in DEFAULT_TERMINAL_STATUSES:
            return
        state["elapsed_seconds"] = float(state.get("elapsed_seconds", 0.0)) + delta
        if not active or state.get("status") not in {"running", "starting"}:
            return
        metrics = state["stage_metrics"][state["stage"]]
        metrics["active_seconds"] = float(metrics.get("active_seconds", 0.0)) + delta

    def _record_observation(self, state: Dict[str, Any], timestamp: str) -> None:
        metrics = state["stage_metrics"][state["stage"]]
        observations = metrics.setdefault("observations", [])
        current = {
            "active_seconds": float(metrics.get("active_seconds", 0.0)),
            "completed": int(state["completed"]),
            "recorded_at_utc": timestamp,
        }
        if not observations or observations[-1].get("completed") != current["completed"]:
            observations.append(current)
            if len(observations) > _MAX_OBSERVATIONS:
                del observations[:-_MAX_OBSERVATIONS]

    def _reject(self, error: BaseException) -> None:
        """Persist a fail-closed error while retaining the prior counters."""

        if self._state.get("status") in DEFAULT_TERMINAL_STATUSES:
            raise error
        try:
            now_mono, timestamp = self._now()
            candidate = copy.deepcopy(self._state)
            self._advance_time(candidate, now_mono, active=False)
            candidate["status"] = "failed"
            candidate["updated_at_utc"] = timestamp
            candidate["finished_at_utc"] = timestamp
            candidate["last_error"] = _error_payload(error)
            self._state = candidate
            self._write()
        except Exception:
            # The original validation error remains the useful caller error.
            pass
        raise error

    def update(
        self,
        *,
        stage: Any = _UNSET,
        status: Any = _UNSET,
        completed: Any = _UNSET,
        total: Any = _UNSET,
        checkpoint: Any = _UNSET,
        last_error: Any = _UNSET,
        stage_estimate_seconds: Any = _UNSET,
        active: bool = True,
    ) -> Dict[str, Any]:
        """Commit one validated update and return the detached state.

        ``completed`` is monotonic within one stage.  A counter reset is only
        valid through a forward ``stage`` transition.  Invalid updates are
        recorded as failed state before raising ``ProgressStateError``.
        """

        try:
            if self._state.get("status") in DEFAULT_TERMINAL_STATUSES:
                raise ProgressStateError("terminal state cannot be updated")
            now_mono, timestamp = self._now()
            candidate = copy.deepcopy(self._state)
            self._advance_time(candidate, now_mono, active=bool(active))
            previous_stage = str(candidate["stage"])
            previous_total = candidate.get("total")
            requested_stage = previous_stage if stage is _UNSET else str(stage).strip()
            if not requested_stage:
                raise ProgressStateError("stage must be non-empty")
            if requested_stage != previous_stage:
                order = list(candidate.get("stage_order", []))
                if requested_stage in order:
                    if order.index(requested_stage) <= order.index(previous_stage):
                        raise ProgressStateError("stage regression is not allowed")
                else:
                    order.append(requested_stage)
                    candidate["stage_plan_complete"] = False
                candidate["stage_order"] = order
                candidate["stage"] = requested_stage
                candidate.setdefault("stage_metrics", {})[requested_stage] = _empty_metrics(timestamp)
                if stage_estimate_seconds is not _UNSET:
                    candidate.setdefault("stage_estimates_seconds", {})[requested_stage] = (
                        None
                        if stage_estimate_seconds is None
                        else _nonnegative_number(stage_estimate_seconds, "stage estimate")
                    )
                # A new stage starts its own counter.  Omitted total means
                # unknown for the new stage, never the previous stage total.
                candidate["completed"] = 0
                candidate["total"] = None if total is _UNSET else total
            else:
                if stage_estimate_seconds is not _UNSET:
                    candidate.setdefault("stage_estimates_seconds", {})[requested_stage] = (
                        None
                        if stage_estimate_seconds is None
                        else _nonnegative_number(stage_estimate_seconds, "stage estimate")
                    )
                if total is not _UNSET:
                    candidate["total"] = total
                    if (
                        previous_total is not None
                        and total is not None
                        and isinstance(total, int)
                        and total < previous_total
                    ):
                        raise ProgressStateError("total cannot regress within a stage")
                if completed is not _UNSET:
                    previous_completed = candidate["completed"]
                    candidate["completed"] = completed
                    if isinstance(completed, int) and completed < previous_completed:
                        raise ProgressStateError("completed cannot regress within a stage")
            if completed is not _UNSET and requested_stage != previous_stage:
                candidate["completed"] = completed
            candidate["completed"], candidate["total"] = _validate_counts(
                candidate["completed"], candidate["total"]
            )
            if status is not _UNSET:
                value = str(status).strip().lower()
                if not value:
                    raise ProgressStateError("status must be non-empty")
                candidate["status"] = value
            if checkpoint is not _UNSET:
                candidate["checkpoint"] = copy.deepcopy(checkpoint)
            if last_error is not _UNSET:
                if last_error is None:
                    candidate["last_error"] = None
                elif isinstance(last_error, Mapping):
                    candidate["last_error"] = copy.deepcopy(dict(last_error))
                else:
                    candidate["last_error"] = _error_payload(last_error)
            candidate["updated_at_utc"] = timestamp
            if candidate["status"] in DEFAULT_TERMINAL_STATUSES:
                candidate["finished_at_utc"] = candidate.get("finished_at_utc") or timestamp
            else:
                candidate["finished_at_utc"] = None
            self._state = candidate
            self._record_observation(self._state, timestamp)
            self._write()
            return self.state
        except ProgressStateError as exc:
            self._reject(exc)

    def set_stage(
        self,
        stage: str,
        *,
        total: Any = _UNSET,
        estimated_seconds: Any = _UNSET,
        completed: int = 0,
    ) -> Dict[str, Any]:
        """Move forward to a stage and reset its counter to zero or ``completed``."""

        estimate = _UNSET
        if estimated_seconds is not _UNSET:
            estimate = None if estimated_seconds is None else _nonnegative_number(
                estimated_seconds, "stage estimate"
            )
        return self.update(
            stage=stage,
            total=total,
            completed=completed,
            stage_estimate_seconds=estimate,
        )

    def checkpoint(self, value: Any) -> Dict[str, Any]:
        return self.update(checkpoint=value)

    def complete(self, *, status: str = "completed", checkpoint: Any = _UNSET) -> Dict[str, Any]:
        """Mark the run terminal while preserving count evidence."""

        success_statuses = {"completed", "success", "succeeded"}
        if str(status).strip().lower() in success_statuses:
            order = self._state.get("stage_order", [])
            if isinstance(order, list) and order and self._state.get("stage") != order[-1]:
                self._reject(ProgressStateError("cannot complete before the final stage"))
            if self._state.get("total") is not None and self._state.get("completed") != self._state.get("total"):
                self._reject(ProgressStateError("cannot complete before completed reaches total"))
        kwargs: Dict[str, Any] = {"status": status}
        if checkpoint is not _UNSET:
            kwargs["checkpoint"] = checkpoint
        return self.update(**kwargs)

    def fail(self, error: Any, *, traceback_text: Optional[str] = None) -> Dict[str, Any]:
        """Persist a failed terminal state with the complete error trace."""

        return self.update(
            status="failed",
            last_error=_error_payload(error, traceback_text),
            active=False,
        )


def _observation_rate(metrics: Mapping[str, Any]) -> Optional[float]:
    observations = metrics.get("observations")
    if not isinstance(observations, list) or len(observations) < 3:
        return None
    valid: List[Tuple[float, int]] = []
    for item in observations:
        if not isinstance(item, Mapping):
            return None
        try:
            active = _nonnegative_number(item.get("active_seconds"), "observation active_seconds")
            completed = _unit_count(item.get("completed"), "observation completed")
        except ProgressStateError:
            return None
        if valid and (active < valid[-1][0] or completed < valid[-1][1]):
            return None
        valid.append((active, completed))
    first_active, first_completed = valid[0]
    last_active, last_completed = valid[-1]
    delta_active = last_active - first_active
    delta_completed = last_completed - first_completed
    positive_intervals = sum(
        1
        for (left_active, left_completed), (right_active, right_completed) in zip(valid, valid[1:])
        if right_active > left_active and right_completed > left_completed
    )
    if delta_active <= 0 or delta_completed <= 0 or positive_intervals < 2:
        return None
    return float(delta_completed) / delta_active


def _state_errors(state: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    try:
        _unit_count(state.get("completed"), "completed")
        total = state.get("total")
        if total is not None:
            _validate_counts(state.get("completed"), total)
    except ProgressStateError as exc:
        errors.append(str(exc))
    stage = state.get("stage")
    order = state.get("stage_order", [])
    if not isinstance(stage, str) or not stage:
        errors.append("stage must be non-empty")
    if not isinstance(order, list) or stage not in order:
        errors.append("current stage is missing from stage_order")
    metrics = state.get("stage_metrics", {})
    if not isinstance(metrics, Mapping) or stage not in metrics:
        errors.append("current stage metrics are missing")
    else:
        observations = metrics[stage].get("observations") if isinstance(metrics[stage], Mapping) else None
        if not isinstance(observations, list) or not observations:
            errors.append("current stage observations are missing")
        else:
            previous_active = -1.0
            previous_completed = -1
            for item in observations:
                if not isinstance(item, Mapping):
                    errors.append("stage observation is not an object")
                    break
                try:
                    active = _nonnegative_number(item.get("active_seconds"), "observation active_seconds")
                    completed = _unit_count(item.get("completed"), "observation completed")
                except ProgressStateError as exc:
                    errors.append(str(exc))
                    break
                if active < previous_active or completed < previous_completed:
                    errors.append("stage observations regress")
                    break
                previous_active, previous_completed = active, completed
    return errors


def derive_progress(
    state: Mapping[str, Any],
    *,
    now_utc: Optional[Any] = None,
    stale_after_seconds: float = _DEFAULT_STALE_AFTER_SECONDS,
    pid_checker: Callable[[Any], bool] = is_pid_alive,
    terminal_statuses: Iterable[str] = DEFAULT_TERMINAL_STATUSES,
) -> Dict[str, Any]:
    """Derive display values without inventing progress or ETA evidence."""

    terminal: Set[str] = {str(value).lower() for value in terminal_statuses}
    status = str(state.get("status", "unknown")).lower()
    current_stage = str(state.get("stage", "unknown"))
    completed = state.get("completed")
    total = state.get("total")
    errors = _state_errors(state)
    done_value: Optional[int]
    total_value: Optional[int]
    try:
        done_value, total_value = _validate_counts(completed, total)
    except ProgressStateError:
        done_value, total_value = None, None
    pid = state.get("pid")
    pid_alive: Optional[bool] = None
    if pid is not None:
        try:
            pid_alive = bool(pid_checker(pid))
        except Exception as exc:
            errors.append("pid check failed: %s" % exc)
            pid_alive = None
    heartbeat_age: Optional[float] = None
    stale_reasons: List[str] = []
    if status not in terminal:
        if pid_alive is False:
            stale_reasons.append("pid_not_alive")
        elif pid_alive is None:
            stale_reasons.append("pid_unknown")
        try:
            now_value = now_utc() if callable(now_utc) else (now_utc if now_utc is not None else utc_now())
            now = _coerce_utc_timestamp(now_value)
            updated = _coerce_utc_timestamp(state.get("updated_at_utc"))
            heartbeat_age = max(0.0, (now - updated).total_seconds())
            if heartbeat_age > float(stale_after_seconds):
                stale_reasons.append("heartbeat_expired")
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append("updated_at_utc is invalid: %s" % exc)
            stale_reasons.append("timestamp_unknown")
    stale = bool(stale_reasons) and status not in terminal
    stage_metrics = state.get("stage_metrics", {})
    current_metrics = stage_metrics.get(current_stage, {}) if isinstance(stage_metrics, Mapping) else {}
    active_seconds: Optional[float]
    try:
        active_seconds = _nonnegative_number(current_metrics.get("active_seconds"), "active_seconds")
    except ProgressStateError:
        active_seconds = None
    rate = _observation_rate(current_metrics) if isinstance(current_metrics, Mapping) else None
    remaining_units: Optional[int] = None
    if done_value is not None and total_value is not None:
        remaining_units = max(0, total_value - done_value)
    stage_eta: Optional[float] = None
    if remaining_units == 0:
        stage_eta = 0.0
    elif remaining_units is not None and rate is not None and rate > 0 and not stale and not errors:
        stage_eta = float(remaining_units) / rate
    order = state.get("stage_order", [])
    estimates = state.get("stage_estimates_seconds", {})
    total_eta: Optional[float] = None
    if status in terminal and status in {"completed", "success", "succeeded"} and not errors:
        total_eta = 0.0
    elif (
        stage_eta is not None
        and bool(state.get("stage_plan_complete"))
        and isinstance(order, list)
        and isinstance(estimates, Mapping)
        and current_stage in order
        and not stale
        and not errors
    ):
        index = order.index(current_stage)
        future = order[index + 1 :]
        future_values: List[float] = []
        for future_stage in future:
            value = estimates.get(future_stage)
            if value is None:
                future_values = []
                break
            try:
                future_values.append(_nonnegative_number(value, "stage estimate"))
            except ProgressStateError:
                future_values = []
                break
        if len(future_values) == len(future):
            total_eta = stage_eta + sum(future_values)
    fraction: Optional[float] = None
    if done_value is not None and total_value is not None and total_value > 0:
        fraction = min(1.0, max(0.0, float(done_value) / float(total_value)))
    try:
        elapsed = _nonnegative_number(state.get("elapsed_seconds"), "elapsed_seconds")
    except ProgressStateError:
        elapsed = None
        errors.append("elapsed_seconds is invalid")
    return {
        "run_id": state.get("run_id"),
        "stage": current_stage,
        "status": status,
        "pid": pid,
        "pid_alive": pid_alive,
        "stale": stale,
        "stale_reasons": stale_reasons,
        "heartbeat_age_seconds": heartbeat_age,
        "completed": done_value,
        "total": total_value,
        "remaining_units": remaining_units,
        "fraction": fraction,
        "elapsed_seconds": elapsed,
        "active_seconds": active_seconds,
        "rate_units_per_second": rate,
        "stage_eta_seconds": stage_eta,
        "total_eta_seconds": total_eta,
        "last_error": copy.deepcopy(state.get("last_error")),
        "checkpoint": copy.deepcopy(state.get("checkpoint")),
        "validation_errors": errors,
        "terminal": status in terminal,
    }


def _duration(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unknown"
    seconds = max(0, int(round(float(value))))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, seconds)
    if minutes:
        return "%dm%02ds" % (minutes, seconds)
    return "%ds" % seconds


def render_progress(snapshot: Mapping[str, Any], *, width: int = 28) -> str:
    """Render one ASCII line plus an error trace when a run failed."""

    width = max(5, int(width))
    fraction = snapshot.get("fraction")
    if isinstance(fraction, (int, float)) and math.isfinite(float(fraction)):
        filled = min(width, max(0, int(round(float(fraction) * width))))
        bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
        percent = "%5.1f%%" % (100.0 * float(fraction))
    else:
        bar = "?" * width
        percent = "  ??%"
    status = str(snapshot.get("status", "unknown"))
    if snapshot.get("stale"):
        display_status = "stale"
    elif snapshot.get("validation_errors"):
        display_status = "invalid"
    else:
        display_status = status
    completed = "?" if snapshot.get("completed") is None else str(snapshot.get("completed"))
    total = "?" if snapshot.get("total") is None else str(snapshot.get("total"))
    remaining = "?" if snapshot.get("remaining_units") is None else str(snapshot.get("remaining_units"))
    line = (
        "[%s] %s stage=%s units=%s/%s remaining=%s status=%s stale=%s "
        "elapsed=%s eta=%s total_eta=%s"
        % (
            bar,
            percent,
            snapshot.get("stage", "unknown"),
            completed,
            total,
            remaining,
            display_status,
            "yes" if snapshot.get("stale") else "no",
            _duration(snapshot.get("elapsed_seconds")),
            _duration(snapshot.get("stage_eta_seconds")),
            _duration(snapshot.get("total_eta_seconds")),
        )
    )
    errors = snapshot.get("validation_errors") or []
    if errors:
        line += "\nERROR validation: " + "; ".join(str(item) for item in errors)
    last_error = snapshot.get("last_error")
    if status == "failed" and isinstance(last_error, Mapping):
        line += "\nERROR %s: %s" % (last_error.get("type", "RuntimeError"), last_error.get("message", ""))
        trace = str(last_error.get("traceback") or "")
        if trace:
            line += "\n" + trace.rstrip("\n")
    return line


def watch_state(
    state_path: Union[Path, str],
    *,
    once: bool = False,
    interval: float = 2.0,
    stop_on_terminal: bool = True,
    stale_after_seconds: float = _DEFAULT_STALE_AFTER_SECONDS,
    terminal_statuses: Iterable[str] = DEFAULT_TERMINAL_STATUSES,
    stream: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Watch one state path and return a useful process exit code."""

    if interval < 0:
        raise ValueError("interval must be non-negative")
    output = sys.stdout if stream is None else stream
    terminal = {str(value).lower() for value in terminal_statuses}
    interactive = bool(getattr(output, "isatty", lambda: False)())
    while True:
        try:
            state = read_state(state_path)
        except StateReadError as exc:
            output.write("ERROR %s\n" % exc)
            output.flush()
            return 1
        snapshot = derive_progress(
            state,
            stale_after_seconds=stale_after_seconds,
            terminal_statuses=terminal,
        )
        rendered = render_progress(snapshot)
        if interactive:
            output.write("\r" + rendered.split("\n", 1)[0])
            if "\n" in rendered:
                output.write("\n" + rendered.split("\n", 1)[1])
        else:
            output.write(rendered + "\n")
        output.flush()
        status = snapshot.get("status")
        if snapshot.get("validation_errors"):
            return 2
        if status == "failed":
            return 1
        if snapshot.get("stale"):
            return 2
        if once or (stop_on_terminal and status in terminal):
            return 0
        sleep(float(interval))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="watch an atomic execution progress JSON state")
    subparsers = parser.add_subparsers(dest="command", required=True)
    watch = subparsers.add_parser("watch", help="render a progress state until terminal")
    watch.add_argument("--state", required=True, type=Path)
    watch.add_argument("--once", action="store_true", help="render one observation and exit")
    watch.add_argument("--interval", type=float, default=2.0)
    watch.add_argument("--stale-after", type=float, default=_DEFAULT_STALE_AFTER_SECONDS)
    terminal_behavior = watch.add_mutually_exclusive_group()
    terminal_behavior.add_argument(
        "--stop-on-terminal",
        dest="stop_on_terminal",
        action="store_true",
        help="stop at terminal status (default)",
    )
    terminal_behavior.add_argument(
        "--keep-running",
        dest="stop_on_terminal",
        action="store_false",
        help="do not stop at terminal status",
    )
    watch.set_defaults(stop_on_terminal=True)
    watch.add_argument(
        "--terminal-status",
        action="append",
        dest="terminal_statuses",
        help="terminal status to stop on; may be repeated",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "watch":
        statuses = args.terminal_statuses or sorted(DEFAULT_TERMINAL_STATUSES)
        try:
            return watch_state(
                args.state,
                once=args.once,
                interval=args.interval,
                stop_on_terminal=args.stop_on_terminal,
                stale_after_seconds=args.stale_after,
                terminal_statuses=statuses,
            )
        except (ValueError, OSError) as exc:
            print("ERROR %s" % exc, file=sys.stderr)
            return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TERMINAL_STATUSES",
    "ProgressStateError",
    "ProgressStateWriter",
    "STATE_SCHEMA",
    "StateReadError",
    "derive_progress",
    "is_pid_alive",
    "main",
    "read_state",
    "render_progress",
    "utc_now",
    "watch_state",
]
