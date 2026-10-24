"""합성 프로세스/시계로 공용 측정 경계를 검사한다. 연구 실행은 없다."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from c_ept import runtime
from c_ept.runtime import resources


class Gone(Exception):
    pass


def proc(private=8, rss=10, peak=20):
    fields = {'rss': rss}
    if private is not None:
        fields['private'] = private
    if peak is not None:
        fields['peak_pagefile'] = peak
    return SimpleNamespace(memory_info=lambda: SimpleNamespace(**fields))


def gone():
    raise Gone()


BACKEND = SimpleNamespace(Error=Gone)


@pytest.mark.parametrize('processes,expected', [
    ([proc(), proc(5, 6, 30)], (13, 50)),
    ([proc(private=None)], (10, 20)),
    ([proc(private=0)], (10, 20)),
    ([proc(), proc(peak=None)], (16, None)),
    ([proc(), SimpleNamespace(memory_info=gone)], (8, 20)),
    ([SimpleNamespace(memory_info=gone)], (0, None)),
    ([], (0, None)),
])
def test_tree_metrics_preserve_partial_and_unknown_semantics(processes, expected):
    assert runtime.tree_memory(processes, process_api=BACKEND) == expected


def test_tree_backend_unavailable_and_disappearing_parent():
    assert runtime.tree_memory([proc()], process_api=None) == (0, None)
    assert runtime.tree_memory(process_api=SimpleNamespace(Error=Gone, Process=gone)) == (0, None)


def test_tree_includes_recursive_children():
    parent = proc()
    parent.children = lambda recursive: [proc(3, 4, 7)] if recursive else []
    backend = SimpleNamespace(Error=Gone, Process=lambda: parent)
    assert runtime.tree_memory(process_api=backend) == (11, 27)


def test_nested_scopes_keep_prior_lifetime_maximum():
    values = iter([(10, 40), (20, None), (30, 70), (5, None)])
    sampler = runtime.ProcessTreeSampler(measure=lambda: next(values))
    with sampler.scope() as outer:
        with sampler.scope() as inner:
            pass
    assert (sampler.peak, sampler.peak_upper, sampler.samples) == (30, 70, 4)
    assert (outer.peak, outer.peak_upper) == (30, 70)
    assert (inner.peak, inner.peak_upper) == (30, 70)
    assert outer.wall >= 0 and inner.wall >= 0 and not sampler._scopes


def test_scalar_samples_and_scope_exception_cleanup():
    sampler = runtime.ProcessTreeSampler(measure=lambda: 7)
    with pytest.raises(ValueError), sampler.scope() as scope:
        raise ValueError('synthetic')
    assert (scope.peak, scope.peak_upper, sampler.samples) == (7, None, 2)
    assert not sampler._scopes


def test_sampler_thread_stops_and_takes_edge_samples():
    with runtime.ProcessTreeSampler(interval=60., measure=lambda: (9, 10)) as sampler:
        assert sampler.samples == 1
    assert sampler.samples == 2 and not sampler._thread.is_alive()


def test_probe_validation_failure_is_counted_and_nested_windows_freeze(monkeypatch):
    ticks = iter([0., 1., 10., 12., 20., 23.])
    monkeypatch.setattr(resources, 'time', SimpleNamespace(perf_counter=lambda: next(ticks)))
    probe = {'synthetic': True}
    seen = []
    guard = runtime.E6Guard(lambda: probe, validate=seen.append)
    before = guard.snapshot()
    with guard.window() as outer:
        guard()
        with guard.window() as inner:
            guard()
    saved = outer.counters()
    guard()
    assert seen == [probe, probe, probe]
    assert (guard.calls, guard.seconds, guard.longest) == (3, 6., 3.)
    assert saved == outer.counters() == {'check_calls': 2, 'check_s_total': 3., 'check_s_mean': 1.5, 'check_s_longest': 2.}
    assert inner.counters()['check_calls'] == 1 and inner.counters()['check_s_longest'] == 2.
    assert guard.since(before)['check_calls'] == 3
    guard._validate = lambda result: (_ for _ in ()).throw(ValueError('synthetic invalid probe'))
    ticks = iter([30., 34.])
    with pytest.raises(ValueError), guard.window() as failed:
        guard()
    assert failed.counters()['check_calls'] == 1 and failed.longest == 4.
    assert guard.calls == 4 and not guard._windows


def test_probe_refusal_propagates_before_validation():
    def refuse():
        raise Gone('synthetic refusal')
    seen = []
    guard = runtime.E6Guard(refuse, validate=seen.append)
    with pytest.raises(Gone):
        guard()
    assert guard.calls == 1 and not seen


def test_s3_compatibility_adapters_keep_dynamic_monkeypatch_points(monkeypatch):
    # Runner 전역 의존성은 로드하지 않고 실제 어댑터 정의만 실행한다.
    tree = ast.parse((Path(__file__).parents[1] / 'run_s3.py').read_text(encoding='utf-8'))
    wanted = {'_tree_memory', '_Sampler', '_require_no_strata', '_Guard'}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    ns = {'runtime': runtime, 'psutil': None}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<S3 compatibility adapters>', 'exec'), ns)
    assert ns['_tree_memory']() == (0, None)
    ns['_tree_memory'] = lambda: (123, 456)
    sampler = ns['_Sampler']()
    sampler.sample()
    assert (sampler.peak, sampler.peak_upper) == (123, 456)
    import c_ept
    seen = []
    fake_s1 = SimpleNamespace(require_no_strata=lambda: seen.append(1))
    monkeypatch.setattr(c_ept, 'run_s1', fake_s1, raising=False)
    guard = ns['_Guard']()
    guard()
    fake_s1.require_no_strata = lambda: seen.append(2)
    guard()
    assert seen == [1, 2] and guard.calls == 2


def test_training_and_runtime_have_no_runner_import_and_s4_uses_public_apis():
    root = Path(__file__).parents[1]
    for path in [root / 'ept/train.py', root / 'ept/checkpoints.py', root / 'runtime/resources.py']:
        source = path.read_text(encoding='utf-8')
        assert 'run_s3' not in source
        tree = ast.parse(source)
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(n.module and '.run_s' in n.module for n in imports)
    s4 = (root / 'run_s4a.py').read_text(encoding='utf-8')
    assert 'checkpoints.latest_checkpoint(folder)' in s4
    assert 'train._latest_checkpoint' not in s4 and 'run_s3._Guard' not in s4
    assert 'from c_ept.run_s3 import _Sampler' not in s4


@pytest.mark.parametrize('used,seconds,blind,metric,refused', [
    (40 * 2**30 - 1, 24 * 3600, False, 'private_bytes_windows', False),
    (40 * 2**30, 24 * 3600, False, 'private_bytes_windows', True),
    (1024, 24 * 3600 + .01, False, 'rss_bytes_fallback', True),
    (0, 0., True, 'private_bytes_windows', True),
    (1024, 0., False, 'unavailable', True),
])
def test_train_guard_keeps_strict_memory_inclusive_time_and_blind_refusal(used, seconds, blind, metric, refused):
    import threading
    tree = ast.parse((Path(__file__).parents[1] / 'ept/train.py').read_text(encoding='utf-8'))
    names = {'_Resources', 'ResourceBoundError', '_violation'}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    clock = [0.]
    ns = {'Path': Path, 'threading': threading, 'runtime': runtime, 'MEMORY_INTERVAL_S': .1,
          'MEMORY_CAP_BYTES': 40 * 2**30, 'WALL_CAP_SECONDS': 24 * 3600,
          '_clock': lambda: clock[0], '_memory_sample': lambda: (used, None, used, metric)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<train pure guard>', 'exec'), ns)
    resources = ns['_Resources']()
    resources.sampler.sample()
    clock[0] = seconds
    assert resources.track['blind'] == blind
    if refused:
        with pytest.raises(ns['ResourceBoundError']):
            resources.guard()
    else:
        resources.guard()
    assert resources.snapshot()['seconds'] == seconds
