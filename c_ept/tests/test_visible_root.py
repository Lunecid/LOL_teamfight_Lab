"""Bounded root-layout checks: synthetic imports/CLI, no protected source or inputs."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    'workflows': 'run_s1 run_s2 run_s3 run_s4a s4a_operator s4a_runtime'.split(),
    'reporting': 'report_s1 report_s2 report_s3'.split(),
    'checks': ['gate_s3'],
}
PAIRS = [(f'c_ept.{n}', f'c_ept.{g}.{n}') for g, names in GROUPS.items() for n in names]
# No interpreter may open these protected modules, even during a transitive import.
PRELUDE = r"""
import sys, types
class DenyProtected:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'research_pipeline.cohort_policy' or fullname.startswith('research_pipeline.holdout_r3'):
            raise AssertionError('protected import attempted: ' + fullname)
sys.meta_path.insert(0, DenyProtected())
fake = types.ModuleType('c_ept.cohort')
def refuse(*args, **kwargs):
    raise AssertionError('no real cohort operation is permitted')
fake.role_of = fake.match_list = refuse
sys.modules['c_ept.cohort'] = fake
"""


def run(code, *args, cwd=ROOT):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_THREADING_LAYER='SEQUENTIAL', KMP_DUPLICATE_LIB_OK='FALSE')
    result = subprocess.run([sys.executable, '-B', '-c', PRELUDE + code, *args],
                            cwd=cwd, env=env, capture_output=True, text=True,
                            encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize('old,new', PAIRS)
@pytest.mark.parametrize('canonical_first', [False, True])
def test_workflow_report_guard_runtime_identity_and_old_type_pickles(old, new, canonical_first):
    run(r"""
import importlib, inspect, pickle
old, new, flag = sys.argv[1:]
first, second = (new, old) if flag == 'True' else (old, new)
a, b = importlib.import_module(first), importlib.import_module(second)
assert a is b is sys.modules[old] is sys.modules[new]
for name in (old, new):
    parent, leaf = name.rsplit('.', 1)
    assert getattr(importlib.import_module(parent), leaf) is a
assert a.__spec__.name == a.__name__ == new
marker = object()
a._root_probe = marker
assert b._root_probe is marker
for name, value in list(vars(a).items()):
    if getattr(value, '__module__', None) != new:
        continue
    if inspect.isfunction(value):
        assert inspect.unwrap(value).__globals__ is a.__dict__
    if isinstance(value, type) and value.__qualname__ == name:
        assert pickle.loads(b'c'+old.encode()+b'\n'+name.encode()+b'\n.') is value
        assert pickle.loads(pickle.dumps(value)) is value
""", old, new, str(canonical_first))


CLI = [(old, new) for old, new in PAIRS if old.rsplit('.', 1)[1].startswith(('run_s', 'report_s')) or old.endswith('s4a_operator')]


@pytest.mark.parametrize('old,new', CLI)
def test_legacy_main_delegates_to_canonical_globals(old, new):
    run(r"""
import importlib, runpy
old, new = sys.argv[1:]
a = importlib.import_module(new)
called = []
a.main = lambda: called.append(True) or 17
try:
    runpy.run_module(old, run_name='__main__')
except SystemExit as error:
    assert error.code == 17
else:
    raise AssertionError('missing main dispatch')
assert called == [True]
""", old, new)


@pytest.mark.parametrize('old,new', CLI)
@pytest.mark.parametrize('canonical', [False, True])
def test_both_module_help_entrypoints_need_no_real_inputs(old, new, canonical):
    result = run(r"""
import runpy
module = sys.argv[1]
sys.argv = [module, '--help']
runpy.run_module(module, run_name='__main__')
""", new if canonical else old)
    assert 'usage:' in result.stdout.lower()


def test_root_discovery_is_unique_and_compat_storage_not_a_second_namespace():
    run(r"""
import c_ept, importlib, pkgutil
from pathlib import Path
root = Path(c_ept.__file__).parent
assert list(c_ept.__path__) == [str(root), str(root / '_compat')]
paths, hooks = list(sys.path), list(sys.meta_path)
found = list(pkgutil.iter_modules(c_ept.__path__, 'c_ept.'))
names = [item.name for item in found]
assert len(names) == len(set(names))
assert {'c_ept.reader', 'c_ept.run_s4a', 'c_ept.reporting', 'c_ept.workflows'} <= set(names)
compat = importlib.import_module('c_ept._compat')
assert list(pkgutil.iter_modules(compat.__path__)) == []
assert sys.path == paths and sys.meta_path == hooks
assert not {'numpy','torch','scipy','c_ept.workflows.run_s4a'} & sys.modules.keys()
""")
    assert {p.name for p in (ROOT/'c_ept').glob('*.py')} == {'__init__.py','config.py','legacy.py','runlog.py','cohort.py'}


def test_package_and_repository_resource_anchors_stay_at_root():
    run(r"""
from pathlib import Path
from c_ept import config, legacy, runlog, s4a_operator, run_s4a
root = Path.cwd()
assert Path(config.__file__).parent == root/'c_ept'
assert legacy.PINS_PATH == root/'c_ept/legacy_pins.json'
assert runlog.HUB == s4a_operator.ROOT == run_s4a.REPO == root
assert legacy.PINS_PATH.is_file()
""")


def test_fixed_extraction_inventory_covers_canonical_sources_and_compat_glue():
    run(r"""
from pathlib import Path
from c_ept import run_s1, runlog
paths = list(run_s1.EXTRACTION_PATHS)
assert len(paths) == len(set(paths)) and all((Path.cwd()/p).is_file() for p in paths)
assert {'c_ept/cohort.py','c_ept/config.py','c_ept/legacy.py','c_ept/__init__.py','c_ept/_compat/__init__.py'} <= set(paths)
for name, group in [('run_s1','workflows'),('reader','data'),('units','domain'),('casts','data'),('targets','domain'),('frame_rows','data'),('event_rows','data'),('store_core','data')]:
    assert f'c_ept/{group}/{name}.py' in paths
    assert f'c_ept/_compat/{name}.py' in paths
for dirty in ('c_ept/data/reader.py','c_ept/_compat/reader.py','c_ept/__init__.py'):
    seen = []
    def git(*args):
        seen.append(args)
        return 'a'*40 if args[0]=='rev-parse' else ' M '+dirty+'\n'
    runlog._git = git
    result = run_s1.extraction_state()
    assert result['extraction_clean'] is False and result['dirty_paths'] == [dirty]
    assert seen[1] == ('status','--porcelain','--',*paths)
runlog._git = lambda *args: None
assert run_s1.extraction_state()['extraction_clean'] is None
""")


def test_startup_helper_is_stdlib_only_in_both_import_orders():
    for first, second in [('c_ept.s4a_runtime','c_ept.workflows.s4a_runtime'),
                          ('c_ept.workflows.s4a_runtime','c_ept.s4a_runtime')]:
        run(r"""
import importlib
assert importlib.import_module(sys.argv[1]) is importlib.import_module(sys.argv[2])
assert not {'numpy','torch','scipy','psutil','c_ept.runtime.resources'} & sys.modules.keys()
""", first, second)


def test_packaged_wheel_contains_aliases_and_installed_cli_imports(tmp_path):
    # Build an offline wheel from allowlisted source. The protected cohort body is
    # neither copied nor opened: use a raising sentinel in this disposable stage.
    import shutil
    import zipfile
    from setuptools import find_packages
    packages = find_packages(str(ROOT), include=['c_ept','c_ept.*'],
                             exclude=['c_ept.tests','c_ept.tests.*'])
    assert 'c_ept._compat' in packages
    stage = tmp_path/'stage'
    stage.mkdir()
    shutil.copyfile(ROOT/'pyproject.toml', stage/'pyproject.toml')
    for source in (ROOT/'c_ept').rglob('*.py'):
        relative = source.relative_to(ROOT)
        if 'tests' in relative.parts or relative.as_posix() == 'c_ept/cohort.py':
            continue
        target = stage/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (stage/'c_ept/cohort.py').write_text("raise AssertionError('protected sentinel')\n", encoding='utf-8')
    result = run("from setuptools.build_meta import build_wheel; build_wheel('dist')", cwd=stage)
    wheel, = (stage/'dist').glob('*.whl')
    installed = tmp_path/'installed'
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for shim in (ROOT/'c_ept/_compat').glob('*.py'):
            assert 'c_ept/_compat/'+shim.name in names
        for old, new in PAIRS:
            assert new.replace('.', '/')+'.py' in names
            assert old.replace('.', '/')+'.py' not in names
        assert 'c_ept/_compat/__init__.py' in names
        entry, = [n for n in names if n.endswith('.dist-info/entry_points.txt')]
        assert 'c_ept.s4a_operator:main' in archive.read(entry).decode()
        archive.extractall(installed)
    env = dict(os.environ, PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1')
    code = r"""
import importlib, sys, pkgutil
from pathlib import Path
import c_ept
assert Path(c_ept.__file__).is_relative_to(Path(sys.argv[1]))
a = importlib.import_module('c_ept.s4a_operator')
b = importlib.import_module('c_ept.workflows.s4a_operator')
assert a is b
assert a.ROOT == Path(sys.argv[1])
r = importlib.import_module('c_ept.s4a_runtime')
assert r is importlib.import_module('c_ept.workflows.s4a_runtime')
assert not {'numpy','torch','psutil'} & sys.modules.keys()
from c_ept import legacy
assert legacy.PINS_PATH == Path(c_ept.__file__).with_name('legacy_pins.json')
names = [m.name for m in pkgutil.iter_modules(c_ept.__path__)]
assert len(names) == len(set(names)) and 'reader' in names and 'run_s4a' in names
"""
    done = subprocess.run([sys.executable,'-B','-c',code,str(installed)], cwd=tmp_path,
                          env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert done.returncode == 0, done.stdout+done.stderr
    for module in ('c_ept.s4a_operator','c_ept.workflows.s4a_operator','c_ept.run_s4a','c_ept.workflows.run_s4a'):
        done = subprocess.run([sys.executable,'-B','-m',module,'--help'], cwd=tmp_path,
                              env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
        assert done.returncode == 0 and 'usage:' in done.stdout.lower(), done.stdout+done.stderr


def test_new_group_initializers_are_docstring_only():
    for group in ('workflows', 'reporting'):
        tree = ast.parse((ROOT/'c_ept'/group/'__init__.py').read_text(encoding='utf-8'))
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
        assert isinstance(tree.body[0].value, ast.Constant)
        assert isinstance(tree.body[0].value.value, str)
