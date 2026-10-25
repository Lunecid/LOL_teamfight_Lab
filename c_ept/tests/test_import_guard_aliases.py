"""Synthetic import-guard regressions; never import a mapped scientific module."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from c_ept.tests.conftest import (IMPORT_GUARD_ALIASES, PROTECTED_IMPORT_PRELUDE,
                                  all_imports, import_watch_names, imported_names,
                                  legacy_import_aliases, module_level_imports)


ROOT = Path(__file__).resolve().parents[2]
PAIRS = tuple(sorted(IMPORT_GUARD_ALIASES.items()))


def names(source, package='c_ept'):
    return imported_names(ast.parse(source).body[0], package)


def test_all_34_aliases_agree_with_layout_registry_and_static_compat_targets():
    # Read one layout registry and compatibility shims, never the mapped target
    # module bodies (in particular never protected cohort/split sources).
    layout = json.loads((ROOT / 'docs/module-layout-20261024.json').read_text(encoding='utf-8'))
    assert layout['schema'] == 'module-layout-v1'
    moved = {a: b for a, b in layout['moved_implementations'].items() if a.startswith('c_ept/')}
    moved.update(layout['ept_visible_root_cleanup']['additional_implementations'])
    dotted = lambda path: path.removesuffix('.py').replace('/', '.')
    expected = {dotted(old): dotted(new) for old, new in moved.items()}
    assert len(expected) == 34 and len(set(expected.values())) == 34
    assert IMPORT_GUARD_ALIASES == expected
    assert 'c_ept.cohort' not in expected
    assert all('research_pipeline' not in value for value in expected.values())
    for old, canonical in expected.items():
        shim = ROOT / 'c_ept/_compat' / (old.rsplit('.', 1)[1] + '.py')
        tree = ast.parse(shim.read_text(encoding='utf-8'))
        targets = [call.args[0].value for call in ast.walk(tree)
                   if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                   and call.func.id == '_import_module' and call.args
                   and isinstance(call.args[0], ast.Constant)]
        assert targets == [canonical], old


@pytest.mark.parametrize('legacy,canonical', PAIRS)
def test_canonical_imports_keep_exact_names_and_add_legacy_equivalents(legacy, canonical):
    assert names(f'import {canonical} as alias') == {canonical, legacy}
    assert names(f'from {canonical} import Member as M') == {
        canonical, canonical + '.Member', legacy, legacy + '.Member'}
    assert names(f'from {canonical} import *') == {canonical, canonical + '.*', legacy, legacy + '.*'}
    parent, leaf = canonical.rsplit('.', 1)
    assert names(f'from {parent} import {leaf} as local') == {parent, canonical, legacy}


@pytest.mark.parametrize('legacy,canonical', PAIRS)
def test_legacy_only_inputs_preserve_exact_historical_sets(legacy, canonical):
    leaf = legacy.rsplit('.', 1)[1]
    assert names(f'import {legacy} as alias') == {legacy}
    assert names(f'from c_ept import {leaf}') == {'c_ept', legacy}
    assert names(f'from {legacy} import Member') == {legacy, legacy + '.Member'}
    assert names(f'from .{leaf} import Member') == {legacy, legacy + '.Member'}
    assert names(f'from .. import {leaf}', 'c_ept.checks') == {'c_ept', legacy}
    assert canonical not in legacy_import_aliases({legacy})


@pytest.mark.parametrize('legacy,canonical', PAIRS)
def test_fresh_module_watchers_catch_either_route_without_requiring_shims(legacy, canonical):
    expected = {legacy, canonical}
    assert set(import_watch_names([legacy])) == expected
    assert set(import_watch_names([canonical])) == expected
    for loaded in ({legacy: object()}, {canonical: object()}):
        assert set(import_watch_names([legacy])) & loaded.keys() == set(loaded)
    assert set(import_watch_names([legacy + '.Member'])) == {legacy + '.Member', canonical + '.Member'}


def test_unmapped_modules_and_similar_prefixes_do_not_gain_fake_aliases():
    unchanged = {'numpy', 'c_ept.config', 'c_ept.cohort', 'c_ept.ept.heads',
                 'c_ept.data.store_core_extra', 'c_ept.workflows.run_s10',
                 'other.c_ept.domain.targets', 'c_ept.domain', 'c_ept.data'}
    assert legacy_import_aliases(unchanged) == unchanged
    assert set(import_watch_names(unchanged)) == unchanged
    assert names('import c_ept.ept.heads') == {'c_ept.ept.heads'}
    assert names('import c_ept.models.heads') == {'c_ept.models.heads', 'c_ept.heads'}


@pytest.mark.parametrize('source,package,expected', [
    ('from . import targets', 'c_ept.domain', {'c_ept.domain', 'c_ept.domain.targets', 'c_ept.targets'}),
    ('from .store_core import Store', 'c_ept.data', {'c_ept.data.store_core', 'c_ept.data.store_core.Store', 'c_ept.store_core', 'c_ept.store_core.Store'}),
    ('from ..workflows import run_s1', 'c_ept.preprocessing', {'c_ept.workflows', 'c_ept.workflows.run_s1', 'c_ept.run_s1'}),
    ('from ..data.reader import MatchRecord', 'c_ept.domain', {'c_ept.data.reader', 'c_ept.data.reader.MatchRecord', 'c_ept.reader', 'c_ept.reader.MatchRecord'}),
])
def test_relative_canonical_imports_resolve_before_alias_normalization(source, package, expected):
    assert names(source, package) == expected


def test_invalid_relative_import_still_refused():
    with pytest.raises(ValueError, match='above'):
        names('from ... import run_s1', 'c_ept.domain')
    with pytest.raises(ValueError, match='above'):
        names('from . import run_s1', '')


def test_canonical_prohibited_names_cannot_bypass_legacy_scanner_guards(tmp_path):
    path = tmp_path / 'synthetic.py'
    path.write_text('''import numpy
if True:
    from c_ept.domain import targets
try:
    from c_ept.data import store_core
except ImportError:
    from c_ept.data.reader import MatchRecord
with context():
    import c_ept.workflows.run_s1
def delayed():
    from c_ept.data import event_rows
class Delayed:
    import c_ept.data.frame_rows
''', encoding='utf-8')
    top = module_level_imports(path, 'c_ept.preprocessing')
    full = all_imports(path, 'c_ept.preprocessing')
    assert {'c_ept.targets', 'c_ept.store_core', 'c_ept.reader', 'c_ept.run_s1'} <= top
    assert {'c_ept.domain.targets', 'c_ept.data.store_core', 'c_ept.data.reader', 'c_ept.workflows.run_s1'} <= top
    assert not {'c_ept.event_rows', 'c_ept.data.event_rows', 'c_ept.frame_rows', 'c_ept.data.frame_rows'} & top
    assert {'c_ept.event_rows', 'c_ept.data.event_rows', 'c_ept.frame_rows', 'c_ept.data.frame_rows'} <= full


def test_legacy_exact_scanner_contract_remains_unchanged(tmp_path):
    path = tmp_path / 'legacy.py'
    path.write_text('''import numpy
from . import run_s1
from .targets import Targets
def delayed():
    from .store_core import Store
    from . import event_rows as rows
''', encoding='utf-8')
    top = {'numpy', 'c_ept', 'c_ept.run_s1', 'c_ept.targets', 'c_ept.targets.Targets'}
    assert module_level_imports(path, 'c_ept') == top
    assert all_imports(path, 'c_ept') == top | {'c_ept.store_core', 'c_ept.store_core.Store', 'c_ept.event_rows'}


def test_fresh_interpreter_prelude_refuses_protected_imports_without_loading_sources():
    # Extract only the pure finder class; do not install or run an import hook in
    # the test interpreter. Both failure branches precede any source loader.
    finder = next(node for node in ast.parse(PROTECTED_IMPORT_PRELUDE).body if isinstance(node, ast.ClassDef))
    namespace = {}
    exec(compile(ast.Module(body=[finder], type_ignores=[]), '<synthetic finder>', 'exec'), namespace)
    deny = namespace['_DenyProtected']()
    for name in ('c_ept.cohort', 'research_pipeline.cohort_policy', 'research_pipeline.holdout_r3', 'research_pipeline.holdout_r3.child'):
        with pytest.raises(AssertionError, match='protected'):
            deny.find_spec(name)
    assert deny.find_spec('c_ept.domain.state') is None


def test_s4a_fresh_process_probe_detects_canonical_and_legacy_runner_names():
    # Evaluate only the probe's boolean expression with a fake sys.modules;
    # never execute the S4a fixture, native environment setup or operator.
    tree = ast.parse((ROOT / 'c_ept/tests/test_s4a_operator.py').read_text(encoding='utf-8'))
    code = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and 'PHASEA_RESULT=' in n.value and 'runner_loaded' in n.value)
    report = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Dict)
                  and any(isinstance(k, ast.Constant) and k.value == 'runner_loaded' for k in n.keys))
    expression = next(v for k, v in zip(report.keys, report.values)
                      if isinstance(k, ast.Constant) and k.value == 'runner_loaded')
    compiled = compile(ast.Expression(body=expression), '<runner-loaded probe>', 'eval')
    for name in ('c_ept.run_s4a', 'c_ept.workflows.run_s4a'):
        assert eval(compiled, {'sys': SimpleNamespace(modules={name: object()})}) is True
    assert eval(compiled, {'sys': SimpleNamespace(modules={'c_ept.config': object()})}) is False
