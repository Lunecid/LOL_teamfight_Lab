"""Synthetic migration checks: no research inputs, training, or external fixtures."""
import ast
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

GROUPS = {
    'data': 'casts reader store_core store_build event_rows frame_rows head_rows'.split(),
    'preprocessing': 'features terrain tokens traits'.split(),
    'domain': 'units targets state symmetry'.split(),
    'valuation': 'flowprice rulepay towertier value compose'.split(),
    'models': 'baselines_b0 baselines_b1 heads nll'.split(),
}
PAIRS = [(f'c_ept.{name}', f'c_ept.{group}.{name}')
         for group, names in GROUPS.items() for name in names]
REPO = Path(__file__).resolve().parents[2]


def run_python(code, *args, input=None):
    done = subprocess.run([sys.executable, '-B', '-c', code, *args], cwd=REPO,
                          input=input, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert done.returncode == 0, done.stderr
    return done


@pytest.mark.parametrize('old,new', PAIRS)
@pytest.mark.parametrize('new_first', [False, True], ids=['legacy-first', 'group-first'])
def test_cold_import_identity_attributes_globals_and_type_pickles(old, new, new_first):
    run_python(r"""
import importlib, inspect, pickle, sys
from pathlib import Path
old, new, new_first = sys.argv[1:]
first, second = (new, old) if new_first == 'True' else (old, new)
a, b = importlib.import_module(first), importlib.import_module(second)
assert a is b is sys.modules[old] is sys.modules[new]
for path in (old, new):
    parent, name = path.rsplit('.', 1)
    assert getattr(importlib.import_module(parent), name) is a
assert Path(a.__file__).resolve() == Path(*new.split('.')).with_suffix('.py').resolve()
assert a.__name__ == a.__spec__.name == new
assert a.__package__ == new.rsplit('.', 1)[0]
marker = object()
a._migration_probe = marker
assert b._migration_probe is marker
for name, obj in list(vars(a).items()):
    if getattr(obj, '__module__', None) != new:
        continue
    if hasattr(obj, '__globals__'):
        function = inspect.unwrap(obj)
        assert function.__globals__ is a.__dict__
        assert function.__globals__['_migration_probe'] is marker
    if isinstance(obj, type) and obj.__qualname__ == name:
        legacy_pickle = b'c' + old.encode() + b'\n' + name.encode() + b'\n.'
        assert pickle.loads(legacy_pickle) is obj
        assert pickle.loads(pickle.dumps(obj)) is obj
        assert getattr(b, name) is obj
del a._migration_probe
""", old, new, str(new_first))


def test_group_packages_have_no_import_side_effects_or_path_hooks():
    run_python(r"""
import importlib, sys
paths, hooks, modules = list(sys.path), list(sys.meta_path), set(sys.modules)
for group in sys.argv[1:]:
    importlib.import_module('c_ept.' + group)
assert sys.path == paths and sys.meta_path == hooks
assert {n for n in set(sys.modules) - modules if n.startswith('c_ept')} == {
    'c_ept', *('c_ept.' + group for group in sys.argv[1:])}
""", *GROUPS)
    for group in GROUPS:
        tree = ast.parse((REPO / 'c_ept' / group / '__init__.py').read_bytes())
        assert all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                   and isinstance(n.value.value, str) for n in tree.body)


def test_monkeypatch_of_legacy_name_changes_implementation_execution(monkeypatch):
    old = importlib.import_module('c_ept.casts')
    new = importlib.import_module('c_ept.data.casts')

    class SyntheticFailure(Exception):
        pass

    class FakeNumpy:
        @staticmethod
        def asarray(value):
            raise SyntheticFailure('synthetic conversion')

    with monkeypatch.context() as patch:
        patch.setattr(old, 'np', FakeNumpy)
        with pytest.raises(SyntheticFailure, match='synthetic conversion'):
            new.narrow('synthetic', [], 'i4')
    with monkeypatch.context() as patch:
        patch.setattr(new, 'np', FakeNumpy)
        with pytest.raises(SyntheticFailure, match='synthetic conversion'):
            old.narrow('synthetic', [], 'i4')


def test_old_and_new_instance_pickles_load_in_fresh_interpreter(monkeypatch):
    from c_ept import reader, state
    instances = [state.BuildingKill(7, 2, 1), reader.MatchFormatError('SYN', '15.18', 'synthetic')]
    payloads = []
    for value in instances:
        cls = type(value)
        new = cls.__module__
        old = 'c_ept.' + new.rsplit('.', 1)[1]
        with monkeypatch.context() as patch:
            patch.setattr(cls, '__module__', old)
            payloads.append(pickle.dumps(value).hex())
        payloads.append(pickle.dumps(value).hex())
    run_python(r"""
import json, pickle, sys
values = [pickle.loads(bytes.fromhex(s)) for s in json.loads(sys.stdin.read())]
from c_ept.domain.state import BuildingKill
from c_ept.data.reader import MatchFormatError
assert type(values[0]) is type(values[1]) is BuildingKill
assert values[0] == values[1] == BuildingKill(7, 2, 1)
assert type(values[2]) is type(values[3]) is MatchFormatError
assert values[2].args == values[3].args
""", input=json.dumps(payloads))


def test_legacy_flowprice_module_dispatches_to_moved_main():
    run_python(r"""
import importlib, runpy
module = importlib.import_module('c_ept.valuation.flowprice')
called = []
module.main = lambda: called.append(True) or 17
try:
    runpy.run_module('c_ept.flowprice', run_name='__main__')
except SystemExit as error:
    assert error.code == 17
else:
    raise AssertionError('legacy CLI did not exit')
assert called == [True]
""")
