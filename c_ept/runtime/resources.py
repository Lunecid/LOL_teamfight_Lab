"""프로세스 트리 메모리 측정(RUN3-2) 및 E6 상태 검사 계측(RUN3-3)을 위한 런타임 자원 경계 모듈.

주요 기능:
- 트리 단위 상주/커밋 메모리 표본 수집 및 중첩 실행 범위(scope)별 최대치 추적.
- 외부 주입 probe를 통한 E6 유효성 검사 횟수 및 실행 시간 집계.

경계 조건 및 설계 원칙:
- 40 GiB 메모리 및 24시간 한계와 같은 실행 정책 판단, 장치 제어는 상위 호출자 소유이다.
- runner 모듈을 참조하지 않으며, 프로세스나 연구 파이프라인 작업을 직접 시작하지 않는다."""
from __future__ import annotations

import contextlib
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

_DEFAULT_PROCESS_API = object()

def tree_memory(processes=None, *, process_api=_DEFAULT_PROCESS_API) -> tuple:
    """현재 프로세스 및 모든 재귀 자식 프로세스의 메모리 합산값과 관측 수명 상한을 측정한다 (RUN3-2).

    매개변수:
    - processes: 측정 대상 프로세스 목록. None이면 현재 프로세스와 하위 트리를 자동 탐색한다.
    - process_api: psutil 호환 인터페이스 객체.

    반환값:
    - (private_bytes, upper_bound) 튜플:
      - private_bytes: Windows의 private 바이트 또는 POSIX의 RSS 합산(int 바이트).
      - upper_bound: 정상 조회된 프로세스들의 peak_pagefile(수명 커밋 상한) 합계. 미지원 환경이거나 조회 프로세스가 없으면 None.

    경계 조건:
    - psutil 부재 시 (0, None)을 반환한다.
    - 조회 중 오류가 발생한 자식 프로세스는 합산에서 제외된다.
    - upper_bound는 관측 시점 기준 성공적으로 읽은 프로세스 집합의 수명 상한이므로, 조기 종료된 자식 누락이나 scope 진입 이전의 과거 기록을 포함할 수 있다."""
    process_api = psutil if process_api is _DEFAULT_PROCESS_API else process_api
    if process_api is None:
        return 0, None
    if processes is None:
        try:
            me = process_api.Process()
            processes = [me, *me.children(recursive=True)]
        except process_api.Error:
            return 0, None
    # 식 (RUN3-2)의 관측 합: 읽을 수 없는 프로세스는 합에서 빠지므로 전체 과거 트리를 보장하지 않는다(S3 R79).
    private, upper, read, complete = 0, 0, 0, True
    for p in processes:
        try:
            mi = p.memory_info()
        except process_api.Error:
            continue
        read += 1
        private += getattr(mi, 'private', None) or mi.rss
        peak = getattr(mi, 'peak_pagefile', None)
        if peak is None:
            complete = False
        else:
            upper += peak
    return private, (upper if read and complete else None)


def larger(a, b):
    """알 수 없는 상한(None)이 기존 최대값을 지우지 않게 병합한다 (RUN3-2).

    The larger of two numbers of which either may be None (unknown): None only when both are.

    식 (RUN3-2)의 해당 항을 계산한다.
    """
    return b if a is None else a if b is None else max(a, b)


class MemoryScope:
    """한 중첩 범위의 표본 최대값·관측 상한·초를 보관한다 (RUN3-2).

    What one ``ProcessTreeSampler.scope()`` saw: the sampled private-or-RSS peak (``peak``), the largest observed lifetime-peak sum
    (``peak_upper``, None where unknown) and the scope's seconds. The lifetime bound can include pre-scope history and omit children
    not successfully read; a later unavailable upper sample does not erase an earlier non-None maximum.

    식 (RUN3-2)의 해당 항을 계산한다.
    """

    def __init__(self):
        self.peak, self.peak_upper, self.wall, self._t0 = 0, None, 0.0, time.perf_counter()


class ProcessTreeSampler:
    """주입된 측정 함수의 트리 메모리 표본 최대값을 추적한다 (RUN3-2).

    interval은 데몬 스레드의 표본 간격(초)이다. measure는 무인자 함수로
    바이트 수 또는 (바이트 수, 수명 상한 또는 None)을 반환하며, 기본값은 tree_memory다.
    시작/종료와 scope 경계에서도 표본을 읽고, 락 안에서 표본 수와 활성 범위의 최대값을 갱신한다.
    peak는 관측 표본의 최대값이며 연속 시간의 실제 최대값을 보장하지 않는다.
    peak_upper는 읽힌 프로세스의 수명 상한 합계의 최대값으로, 범위 이전 기록을 포함하고
    누락된 자식은 제외될 수 있다. 새 None 표본은 이전의 알려진 상한을 지우지 않는다.
    기본 측정에서 psutil이 없으면 (0, None)을 얻는다. 이는 메모리 사용량 0의 증거가 아니다.
    40 GiB/24시간 제한 판정은 호출자의 책임이다."""

    def __init__(self, interval: float = 0.1, measure=None):
        self.interval = interval
        self._measure = tree_memory if measure is None else measure
        self.peak, self.peak_upper, self.samples = 0, None, 0
        self._scopes, self._lock = [], threading.Lock()
        self._stop, self._thread = threading.Event(), None

    def sample(self) -> None:
        """식 (RUN3-2)의 해당 항을 계산한다."""
        got = self._measure()
        used, upper = got if isinstance(got, tuple) else (got, None)
        used, upper = int(used), None if upper is None else int(upper)
        # 식 (RUN3-2)의 범위 최대값: None 표본은 이전 상한을 지우지 않는다(S3 R79).
        with self._lock:
            self.samples += 1
            self.peak = max(self.peak, used)
            self.peak_upper = larger(self.peak_upper, upper)
            for scope in self._scopes:
                scope.peak = max(scope.peak, used)
                scope.peak_upper = larger(scope.peak_upper, upper)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    @contextlib.contextmanager
    def scope(self):
        """중첩 구간의 메모리 표본 최대값과 경과 시간을 기록한다 (RUN3-2).

        MemoryScope를 활성 목록에 먼저 등록하고 진입 표본을 수집한 뒤 yield한다.
        정상 종료 및 본문 예외 시 finally에서 종료 표본, wall(초), 등록 해제를 순서대로 처리한다.
        목록 및 최대값 변경은 락으로 보호하지만 측정 함수 호출과 wall 갱신 전체가 잠기는 것은 아니다.
        측정 함수 자체의 예외는 전파되며 종료 표본이 실패하면 뒤의 정리도 완료되지 않을 수 있다.
        peak/peak_upper는 바이트 단위이고, peak_upper는 알 수 없을 때 None이다."""
        scope = MemoryScope()
        with self._lock:
            self._scopes.append(scope)
        try:
            self.sample()
            yield scope
        finally:
            self.sample()
            scope.wall = time.perf_counter() - scope._t0
            with self._lock:
                self._scopes.remove(scope)

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.sample()
        return False


class GuardWindow:
    """닫힌 범위의 E6 횟수·검사 초를 고정한다 (RUN3-3).

    What the E6 guard did while a ``E6Guard.window()`` was open: the number of checks, their seconds and the longest one. Read
    ``counters()`` after the window closed and they are those of the window, whatever the guard did since.

    식 (RUN3-3)의 해당 항을 계산한다.
    """

    def __init__(self, guard):
        self._guard, self.longest = guard, 0.0
        self._calls0, self._seconds0 = guard.calls, guard.seconds
        self._calls1 = self._seconds1 = None                      # set when the window closes

    def close(self) -> None:
        if self._calls1 is None:
            self._calls1, self._seconds1 = self._guard.calls, self._guard.seconds

    def counters(self, prefix: str = '') -> dict:
        """The keys ``{prefix}check_calls``, ``..._s_total``, ``..._s_mean`` (None without a check) and ``..._s_longest``.

        식 (RUN3-3)의 해당 항을 계산한다.
        """
        end_calls = self._guard.calls if self._calls1 is None else self._calls1
        end_seconds = self._guard.seconds if self._seconds1 is None else self._seconds1
        calls, seconds = end_calls - self._calls0, end_seconds - self._seconds0
        return {f'{prefix}check_calls': calls, f'{prefix}check_s_total': round(seconds, 4),
                f'{prefix}check_s_mean': round(seconds / calls, 4) if calls else None, f'{prefix}check_s_longest': round(self.longest, 4)}


class E6Guard:
    """주입된 E6 probe의 호출 횟수와 소요 시간을 계수한다 (RUN3-3).

    probe는 무인자 함수다. 선택적 keyword-only validate는 probe 결과를 받아 한 번 검사한다.
    probe 또는 validate의 예외(예: StrataRunning)는 호출자에게 전파되며,
    finally에서 calls, seconds, longest와 활성 윈도우의 최장 시간을 갱신한다.
    이 계수기는 스레드 안전성이나 원자적 갱신을 보장하지 않는다.
    snapshot은 전체 계수, window는 중첩 가능한 구간 계수를 제공한다.
    검사 정책과 거부 결정은 probe/validate 소유다."""


    def __init__(self, probe, *, validate=None):
        self._probe, self._validate = probe, validate
        self.calls, self.seconds, self.longest = 0, 0.0, 0.0
        self._windows = []

    def __call__(self):
        """식 (RUN3-3)의 해당 항을 계산한다."""
        t0 = time.perf_counter()
        try:
            result = self._probe()
            if self._validate is not None:
                self._validate(result)
        finally:
            spent = time.perf_counter() - t0
            self.calls += 1
            self.seconds += spent
            self.longest = max(self.longest, spent)
            for window in self._windows:
                window.longest = max(window.longest, spent)

    @contextlib.contextmanager
    def window(self):
        """특정 실행 구간 동안 발생한 E6 검사 계측 지표를 격리 수집하는 컨텍스트 매니저.

        반환값:
        - GuardWindow: 구간 내 check_calls, check_s_total, check_s_mean, check_s_longest를 계산하는 윈도우 객체.

        경계 조건:
        - 중첩 사용이 가능하며, finally 블록에서 윈도우를 닫고 가드 등록을 해제하여 이후 호출의 영향을 배제한다."""
        window = GuardWindow(self)
        self._windows.append(window)
        try:
            yield window
        finally:
            window.close()
            self._windows.remove(window)

    def snapshot(self) -> tuple:
        return self.calls, self.seconds

    def since(self, before: tuple) -> dict:
        """식 (RUN3-3)의 해당 항을 계산한다."""
        calls, seconds = self.calls - before[0], self.seconds - before[1]
        return {'check_calls': calls, 'check_s_total': round(seconds, 4), 'check_s_mean': round(seconds / calls, 4) if calls else None}
