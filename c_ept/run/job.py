"""Windows Job Object ownership of a child process tree, for the supervisor of the run kernel (S4b-prep K5b, decision D11: "the supervisor reuses the
components proven in T009").

The wrapper waits on stdin until assigned to an unnamed, non-inheritable job. No breakaway flags are enabled. Kill-on-close also covers supervisor
death: when the supervisor dies, the last handle of the job closes and Windows kills every process in it. Only launch-owned Popen handles are
assigned; PID references never grant ownership. Win32 layouts: Microsoft Learn jobapi2/winnt job-object structures and functions.

# Ref: tools/ept_training_job.py (the T002 job module, proven in the T002 and T009 launches). It is frozen: the T002 controller checks that
# ``WindowsJob`` comes from that file, and ``c_ept/run`` may not import from ``tools`` (Q8). So the code is copied here without a change, and a test
# (``test_the_job_module_is_the_code_of_the_t002_job_module`` in ``c_ept/tests/test_run_supervisor.py``) compares the syntax trees of the six
# definitions with the ones in ``tools/ept_training_job.py``; they cannot drift apart unnoticed.

Protocol (the cooperative gate)
-------------------------------
    1. the supervisor starts the child with ``stdin=PIPE``; the child reads ONE byte from stdin before it does any work;
    2. ``assign(child)`` puts the child's own process handle into the job (``AssignProcessToJobObject``) and checks ``IsProcessInJob``;
    3. ``release(child)`` writes ``b'G'`` and closes stdin: only now does the child go on (EOF without ``G`` makes it exit);
    4. ``stop()`` terminates the job (``TerminateJobObject``) and waits until no process is active; ``close()`` closes the last handle, which kills
       whatever is left (kill-on-close).
    So no code of the child, and no grandchild it could start, runs outside the job.

해석: 자식 프로세스 트리를 Windows Job Object가 소유하게 한다. 자식은 stdin에서 한 글자(G)를 받을 때까지 아무 일도 하지 않으므로, 작업에 넣기 전에 손자
프로세스를 만들 수 없다. 감독자가 죽으면 Job의 마지막 핸들이 닫히면서 Windows가 트리 전체를 죽인다(kill-on-close). 이 파일은 T002의 동결된 작업 모듈을
그대로 옮긴 것이고(``c_ept/run``은 ``tools``를 import할 수 없다), 시험이 두 파일의 구문 트리가 같음을 확인한다.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import os
import math
import time


class JobError(OSError):
    pass


class _BasicLimits(C.Structure):
    _fields_ = [('PerProcessUserTimeLimit', C.c_longlong), ('PerJobUserTimeLimit', C.c_longlong),
                ('LimitFlags', W.DWORD), ('MinimumWorkingSetSize', C.c_size_t),
                ('MaximumWorkingSetSize', C.c_size_t), ('ActiveProcessLimit', W.DWORD),
                ('Affinity', C.c_size_t), ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]


class _IoCounters(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _ExtendedLimits(C.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IoCounters),
                ('ProcessMemoryLimit', C.c_size_t), ('JobMemoryLimit', C.c_size_t),
                ('PeakProcessMemoryUsed', C.c_size_t), ('PeakJobMemoryUsed', C.c_size_t)]


class _Accounting(C.Structure):
    _fields_ = [(name, C.c_longlong) for name in ('TotalUserTime', 'TotalKernelTime',
                'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [
                (name, W.DWORD) for name in ('TotalPageFaultCount', 'TotalProcesses',
                                           'ActiveProcesses', 'TotalTerminatedProcesses')]


class WindowsJob:
    """A fresh OS-owned job; never reopen/adopt a named or PID-only job."""
    def __init__(self):
        if os.name != 'nt':
            raise JobError('Windows Job Object required for this approved Windows run')
        self.handle = None
        self.child = None
        self.released = False
        self.k = C.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
            'QueryInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
            'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
            'IsProcessInJob': ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
            'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            'GetProcessId': ([W.HANDLE], W.DWORD),
            'GetProcessTimes': ([W.HANDLE, C.POINTER(W.FILETIME), C.POINTER(W.FILETIME),
                                 C.POINTER(W.FILETIME), C.POINTER(W.FILETIME)], W.BOOL),
            'CloseHandle': ([W.HANDLE], W.BOOL),
        }
        for name, (args, result) in signatures.items():
            func = getattr(self.k, name)
            func.argtypes, func.restype = args, result
        self.handle = self.k.CreateJobObjectW(None, None)  # no inheritable security attributes/name
        if not self.handle:
            self._error('CreateJobObjectW')
        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE only
            self._ok(self.k.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)),
                     'SetInformationJobObject(KILL_ON_JOB_CLOSE)')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _error(operation):
        code = C.get_last_error()
        raise JobError(code, operation + ': ' + C.FormatError(code))

    def _ok(self, value, operation):
        if not value:
            self._error(operation)
        return value

    def _handle(self):
        if not self.handle:
            raise JobError('job handle already closed; containment is unavailable')
        return self.handle

    def assign(self, child):
        """Assign exact live Popen handle while its cooperative wrapper awaits G."""
        if self.child is not None or self.released or child.stdin is None or child.poll() is not None:
            raise JobError('one live, gated child required before job release')
        # CPython Windows Popen retains the CreateProcess process handle. It cannot
        # be confused with a reused numeric PID and is never taken from run metadata.
        handle = getattr(child, '_handle', None)
        if handle is None or self.k.GetProcessId(int(handle)) != child.pid:
            raise JobError('launch-owned CPython process handle unavailable/mismatched')
        self._ok(self.k.AssignProcessToJobObject(self._handle(), int(handle)), 'AssignProcessToJobObject')
        contained = W.BOOL()
        self._ok(self.k.IsProcessInJob(int(handle), self._handle(), C.byref(contained)), 'IsProcessInJob(launch)')
        if not contained.value:
            raise JobError('launch handle was not contained in this job')
        self.child = child

    def release(self, child):
        if child is not self.child or self.released or child.stdin is None:
            raise JobError('release requires the exact assigned gated Popen child')
        child.stdin.write(b'G')
        child.stdin.flush()
        child.stdin.close()
        self.released = True

    def contains(self, pid, create_time):
        """Check OS job membership AND generation using the same opened handle."""
        if type(pid) is not int or pid <= 0 or type(create_time) not in (int, float) or not math.isfinite(create_time) or create_time <= 0:
            raise JobError('invalid process generation')
        handle = self.k.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION only
        if not handle:
            code = C.get_last_error()
            if code == 87:  # ERROR_INVALID_PARAMETER: process no longer exists
                return False
            self._error('OpenProcess(query membership)')
        try:
            created, ended, kernel, user = (W.FILETIME() for _ in range(4))
            self._ok(self.k.GetProcessTimes(handle, C.byref(created), C.byref(ended),
                     C.byref(kernel), C.byref(user)), 'GetProcessTimes(membership)')
            observed = ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 10000000 - 11644473600
            if abs(observed - create_time) > .00001:
                return False
            contained = W.BOOL()
            self._ok(self.k.IsProcessInJob(handle, self._handle(), C.byref(contained)), 'IsProcessInJob(owner)')
            return bool(contained.value)
        finally:
            self.k.CloseHandle(handle)

    def active_count(self):
        info = _Accounting()
        self._ok(self.k.QueryInformationJobObject(self._handle(), 1, C.byref(info), C.sizeof(info), None),
                 'QueryInformationJobObject(accounting)')
        return int(info.ActiveProcesses)

    def gone(self):
        return self.active_count() == 0

    def stop(self, timeout=10):
        self._ok(self.k.TerminateJobObject(self._handle(), 124), 'TerminateJobObject')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.gone():
                return True
            time.sleep(.05)
        return self.gone()

    def close(self):
        if self.handle:
            # Keep the owned handle/evidence if close fails; never certify cleanup.
            self._ok(self.k.CloseHandle(self.handle), 'CloseHandle(job)')
            self.handle = None  # last successful handle close kills remaining descendants

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
