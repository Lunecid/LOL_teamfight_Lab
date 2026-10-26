"""Tests of the gate set of the run kernel (``c_ept.run.guards``, S4b-prep K4, decisions D11 and A5-F3, Q7).

The earlier stacks (S3, S4a, T002, T003, T008/T009) each implement the start-of-run gates in their own way (review A5-F3): the native-thread
and MKL policy (``workflows.s4a_runtime.prepare_native_runtime``; the comparison workers never call it), E6 (``run_s1.require_no_strata`` with
``s4a_runtime.validate_e6_probe`` and ``run_s4a._e6``; the comparison stack has no E6 check), the child environment (three copies:
``tools/ept_training_controller.child_environment``, ``comparison/operator.child_environment``, ``tools/ept_dev_evaluate.child_env``), the decision
record (``config.check_spec``) and the budgets (``evaluation.runtime.Budget``, ``comparison.infrastructure``). The kernel has one function
or class for each. These tests pin each gate's refusal and pass case, and they pin the pieces that reproduce the T002 stack against that
stack's own source: the native policy by calling the frozen function in a fresh process, the child environment by running the frozen
function body on the same inputs, the E6 record check by calling ``validate_e6_probe`` on the same records.

Nothing here starts or stops a process, uses the GPU or reads a research file. The process listing, the probe, the clock and the memory reader
are injected; the decision records are synthetic files in ``tmp_path``.

해석: 관문은 실행을 시작하기 전에 한 번씩 하는 검사이다. 지금까지는 스택마다 같은 검사를 따로 구현했다. 커널은 그 가운데 T002 스택의 동작을 기본값으로
삼아, 기본 설정으로 T002와 같은 결과를 낼 수 있게 한다. 그래서 시험은 관문마다 거부와 통과를 따로 확인하고, T002와 같다는 점은 값을 시험 파일에 다시
적어 비교하지 않고 T002 쪽 코드를 직접 실행해서 비교한다(새 프로세스에서 얼린 함수 호출, 얼린 함수 본문 실행, 같은 기록에 얼린 검사 함수 호출).
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from c_ept.run import guards
from c_ept.run.guards import Budget, GateRefused

REPO = Path(__file__).resolve().parents[2]
S4A_RUNTIME = REPO / 'c_ept/workflows/s4a_runtime.py'
CONTROLLER = REPO / 'tools/ept_training_controller.py'
GUARDS_SOURCE = Path(guards.__file__)
MARK = 'LEAKED_VALUE_MARKER'               # text that must never appear in a message
STDLIB = set(sys.stdlib_module_names)


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'))


def load_frozen_s4a_runtime():
    """The frozen T002 module, loaded from its file (it imports only the standard library) under a private name."""
    spec = importlib.util.spec_from_file_location('frozen_s4a_runtime', S4A_RUNTIME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_controller_child_environment(interpreter: Path):
    """The body of ``tools/ept_training_controller.child_environment`` (T002), compiled alone with ``PYTHON`` set to ``interpreter``.

    The controller module itself is not imported (it is a command module); only the function node is taken from its syntax tree.
    """
    node = next(n for n in parse(CONTROLLER).body if isinstance(n, ast.FunctionDef) and n.name == 'child_environment')
    namespace = {'os': os, 'PYTHON': interpreter}
    exec(compile(ast.Module([node], []), str(CONTROLLER), 'exec'), namespace)
    return namespace['child_environment']


def controller_function_node() -> ast.FunctionDef:
    return next(n for n in parse(CONTROLLER).body if isinstance(n, ast.FunctionDef) and n.name == 'child_environment')


# ---- the interface and the module's own discipline ------------------------------------------------------------------------------------
def test_the_interface_has_the_planned_names_and_signatures():
    assert list(inspect.signature(guards.native_policy).parameters) == []
    e6 = inspect.signature(guards.e6)
    assert list(e6.parameters) == ['processes', 'probe']
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None for p in e6.parameters.values())
    child = inspect.signature(guards.child_environment)
    assert list(child.parameters) == ['base', 'cuda_visible_devices', 'threads']
    assert child.parameters['base'].default is inspect.Parameter.empty
    assert child.parameters['cuda_visible_devices'].kind is child.parameters['threads'].kind is inspect.Parameter.KEYWORD_ONLY
    assert (child.parameters['cuda_visible_devices'].default, child.parameters['threads'].default) == ('0', 8)     # the T002 defaults
    spec = inspect.signature(guards.spec_check)
    assert list(spec.parameters) == ['path', 'expected_sha256']
    assert spec.parameters['path'].default is None and spec.parameters['path'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert spec.parameters['expected_sha256'].default is None and spec.parameters['expected_sha256'].kind is inspect.Parameter.KEYWORD_ONLY
    budget = inspect.signature(Budget)
    assert list(budget.parameters) == ['wall_s', 'memory_bytes', 'clock', 'measure']
    assert budget.parameters['clock'].kind is budget.parameters['measure'].kind is inspect.Parameter.KEYWORD_ONLY
    assert budget.parameters['measure'].default is None
    assert issubclass(GateRefused, RuntimeError)


def test_the_module_imports_only_the_standard_library_at_import_time_and_only_public_kernel_names_lazily():
    tree = parse(GUARDS_SOURCE)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            top.add((node.module or '').split('.')[0])
    assert top <= STDLIB, sorted(top - STDLIB)
    lazy = set()
    for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(function):
            if isinstance(node, ast.ImportFrom):
                lazy |= {f'{node.module}.{a.name}' for a in node.names}
            elif isinstance(node, ast.Import):
                lazy |= {a.name for a in node.names}
    assert lazy <= {'c_ept.config', 'c_ept.runlog', 'c_ept.runtime.resources'}, sorted(lazy)


def test_the_module_follows_the_kernel_boundary_rule_q8():
    """Q8/K7 for ``c_ept/run/``: no private (underscore) name or module from another package, and no command module."""
    commands = ('c_ept.workflows.run_', 'c_ept.workflows.s4a_operator', 'c_ept.workflows.training_operator', 'c_ept.comparison.operator', 'tools')
    for node in ast.walk(parse(GUARDS_SOURCE)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            assert not any(part.startswith('_') and part != '__future__' for part in module.split('.')), module
            assert not any(a.name.startswith('_') for a in node.names if module != '__future__'), module
            assert not module.startswith(commands), module
            assert not any(f'{module}.{a.name}'.startswith(commands) for a in node.names), module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(part.startswith('_') for part in alias.name.split('.')), alias.name
                assert not alias.name.startswith(commands), alias.name


def test_the_module_writes_no_file_and_starts_or_stops_no_process():
    """A gate refuses before any reservation, so no gate may create a file or a folder; no gate starts or stops a process either."""
    tree = parse(GUARDS_SOURCE)
    forbidden_modules = {'subprocess', 'shutil', 'tempfile', 'multiprocessing', 'signal', 'psutil', 'ctypes'}
    forbidden_calls = {'mkdir', 'makedirs', 'write_text', 'write_bytes', 'touch', 'unlink', 'rename', 'rmdir', 'rmtree', 'remove', 'symlink_to',
                       'hardlink_to', 'chmod', 'kill', 'terminate', 'system', 'popen', 'startfile', 'Popen', 'fsync'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {a.name.split('.')[0] for a in node.names} & forbidden_modules
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in forbidden_modules
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
            assert name not in forbidden_calls, name
            if name == 'open' or name.startswith('spawn'):                 # a plain open() is a read here only if it has no write mode
                modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)] + [k.value.value for k in node.keywords
                                                                                              if k.arg == 'mode' and isinstance(k.value, ast.Constant)]
                assert not name.startswith('spawn') and not any(set('wax+') & set(str(m)) for m in modes)


def test_importing_the_module_loads_no_native_or_model_code():
    """The native policy must run before numpy, torch or scipy are imported, so the module that holds it may not import them (or the heavy gates)."""
    code = ('import sys; import c_ept.run.guards; '
            'bad = sorted({"numpy", "torch", "scipy", "psutil", "c_ept.runlog", "c_ept.config", "c_ept.legacy", "c_ept.runtime"} & sys.modules.keys()); print(bad)')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '[]', done.stdout


def test_a_gate_refusal_names_its_gate_is_a_runtime_error_and_survives_pickling():
    error = GateRefused('e6', 'a model server runs')
    assert isinstance(error, RuntimeError)
    assert (error.gate, error.reason) == ('e6', 'a model server runs')
    assert str(error) == 'e6 gate: a model server runs'
    again = pickle.loads(pickle.dumps(error))
    assert type(again) is GateRefused and (again.gate, again.reason) == ('e6', 'a model server runs') and str(again) == str(error)


def test_a_gate_refusal_accepts_only_a_known_gate_name():
    """K4 review m7: a misspelt gate name in K5 must not pass silently."""
    assert guards.GATES == ('native', 'e6', 'environment', 'spec', 'budget')
    for gate in guards.GATES:
        assert GateRefused(gate, 'why').gate == gate
    for gate in ('nativ', 'E6', '', 'budgets', None, 3, ['e6']):
        with pytest.raises(ValueError):
            GateRefused(gate, 'why')


# ---- native_policy ---------------------------------------------------------------------------------------------------------------------
MATRIX_CHILD = r'''
import importlib.util, json, os, sys, types
spec = importlib.util.spec_from_file_location('frozen_s4a_runtime', sys.argv[1])
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
from c_ept.run import guards
assert not {'numpy', 'torch', 'scipy'} & sys.modules.keys(), 'a native module was imported before the policy'
KEYS = ('KMP_DUPLICATE_LIB_OK', 'MKL_THREADING_LAYER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
NATIVE = ('numpy', 'torch', 'scipy')

def prepare(kmp, mkl, imported, threads):
    for key in KEYS:
        os.environ.pop(key, None)
    if kmp is not None:
        os.environ['KMP_DUPLICATE_LIB_OK'] = kmp
    if mkl is not None:
        os.environ['MKL_THREADING_LAYER'] = mkl
    if threads is not None:
        for key in KEYS[2:]:
            os.environ[key] = threads
    for name in NATIVE:
        sys.modules.pop(name, None)
    for name in imported:
        sys.modules[name] = types.ModuleType(name)

def run(function, refusal):
    try:
        record = function()
    except refusal:
        record = 'refused'
    return record, {key: os.environ.get(key) for key in KEYS}

counts = {'scenarios': 0, 'accepted': 0, 'refused': 0}
mismatches = []
for kmp in (None, '', '0', 'false', 'FALSE', 'No', 'off', 'OFF', ' false ', 'TRUE', '1', 'yes', 'on', 'x'):
    for mkl in (None, 'SEQUENTIAL', 'sequential', 'INTEL', 'GNU'):
        for imported in ((), ('numpy',), ('torch',), ('scipy',), ('numpy', 'torch', 'scipy')):
            for threads in (None, '3'):
                prepare(kmp, mkl, imported, threads)
                old = run(frozen.prepare_native_runtime, frozen.RuntimeRefused)
                prepare(kmp, mkl, imported, threads)
                new = run(guards.native_policy, guards.GateRefused)
                counts['scenarios'] += 1
                counts['refused' if old[0] == 'refused' else 'accepted'] += 1
                if old != new:
                    mismatches.append([kmp, mkl, list(imported), threads])
print(json.dumps({**counts, 'mismatches': mismatches[:5], 'n_mismatches': len(mismatches)}))
'''


def test_native_policy_agrees_with_the_frozen_t002_function_over_a_scenario_matrix():
    """Both functions run in a fresh process (numpy not imported) over every combination of the override variable, the MKL layer, the
    native modules that count as imported and the thread variables: the same refusal or the same record, and the same environment afterwards."""
    done = subprocess.run([sys.executable, '-B', '-c', MATRIX_CHILD, str(S4A_RUNTIME)], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result['n_mismatches'] == 0, result
    assert result['scenarios'] == 14 * 5 * 5 * 2
    assert result['accepted'] > 0 and result['refused'] > 0                  # the matrix reaches both outcomes


def test_native_policy_values_are_pinned_to_the_t002_source_by_syntax_tree():
    tree = parse(S4A_RUNTIME)
    version = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'RUNTIME_POLICY_VERSION' for t in n.targets))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_native_runtime')
    assigned = {}
    tuples = []
    record_keys = None
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript) and isinstance(node.value, ast.Constant):
            target = node.targets[0]
            if isinstance(target.value, ast.Attribute) and target.value.attr == 'environ':
                assigned[target.slice.value] = node.value.value
        elif isinstance(node, ast.Tuple) and all(isinstance(e, ast.Constant) for e in node.elts):
            tuples.append(tuple(e.value for e in node.elts))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            record_keys = tuple(k.value for k in node.value.keys)
    assert guards.NATIVE_POLICY_VERSION == version
    assert assigned == {'KMP_DUPLICATE_LIB_OK': guards.CHILD_SETTINGS['KMP_DUPLICATE_LIB_OK'], 'MKL_THREADING_LAYER': guards.CHILD_SETTINGS['MKL_THREADING_LAYER']}
    assert assigned == {'KMP_DUPLICATE_LIB_OK': 'FALSE', 'MKL_THREADING_LAYER': 'SEQUENTIAL'}
    assert guards.OVERRIDE_OFF in tuples and guards.NATIVE_MODULES in tuples and guards.THREAD_KEYS in tuples
    assert record_keys == ('policy_version', 'mkl_threading_layer', 'numpy_mkl_parallelism', 'kmp_duplicate_lib_ok', 'python_executable', 'thread_environment')


def test_native_policy_returns_the_t002_record_and_is_idempotent(monkeypatch):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK', 'FALSE')                         # set (not deleted) so that the teardown restores what the policy writes
    monkeypatch.setenv('MKL_THREADING_LAYER', 'SEQUENTIAL')
    first = guards.native_policy()
    assert first == {'policy_version': 1, 'mkl_threading_layer': 'SEQUENTIAL', 'numpy_mkl_parallelism': 'sequential', 'kmp_duplicate_lib_ok': 'disabled',
                     'python_executable': str(Path(sys.executable).resolve()),
                     'thread_environment': {'OMP_NUM_THREADS': None, 'MKL_NUM_THREADS': None, 'OPENBLAS_NUM_THREADS': None}}
    assert os.environ['KMP_DUPLICATE_LIB_OK'] == 'FALSE' and os.environ['MKL_THREADING_LAYER'] == 'SEQUENTIAL'
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    assert guards.native_policy()['thread_environment']['OMP_NUM_THREADS'] == '8'      # the record reads the environment, it does not set the counts


def test_native_policy_refuses_the_error_hiding_override_without_echoing_its_value(monkeypatch):
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK', MARK)
    monkeypatch.setenv('MKL_THREADING_LAYER', 'SEQUENTIAL')
    with pytest.raises(GateRefused) as caught:
        guards.native_policy()
    assert caught.value.gate == 'native'
    assert 'KMP_DUPLICATE_LIB_OK' in str(caught.value) and MARK not in str(caught.value)
    assert os.environ['KMP_DUPLICATE_LIB_OK'] == MARK                           # a refusal changes nothing


@pytest.mark.parametrize('layer', ['INTEL', 'GNU', 'sequential', MARK])
def test_native_policy_refuses_an_imported_native_module_without_the_sequential_layer(monkeypatch, layer):
    monkeypatch.setitem(sys.modules, 'numpy', sys.modules.get('numpy') or type(sys)('numpy'))
    monkeypatch.delenv('KMP_DUPLICATE_LIB_OK', raising=False)
    monkeypatch.setenv('MKL_THREADING_LAYER', layer)
    with pytest.raises(GateRefused) as caught:
        guards.native_policy()
    assert caught.value.gate == 'native' and 'MKL_THREADING_LAYER' in str(caught.value) and MARK not in str(caught.value)
    assert os.environ['MKL_THREADING_LAYER'] == layer


# ---- child_environment -----------------------------------------------------------------------------------------------------------------
def make_interpreter(tmp_path: Path, monkeypatch) -> Path:
    """A stand-in interpreter path under ``tmp_path/anaconda3``; ``sys.executable`` is set to it (the kernel reads its folder)."""
    interpreter = (tmp_path / 'anaconda3' / 'python.exe').resolve()                 # resolved, as the kernel resolves sys.executable (K4 review m6)
    monkeypatch.setattr(sys, 'executable', str(interpreter))
    return interpreter


def sample_path(interpreter: Path) -> str:
    root = str(interpreter.parent)
    return os.pathsep.join([root, os.path.join(root, 'Scripts'), root.upper() + '\\Library\\bin', 'C:\\Windows\\system32', 'C:\\Tools\\Codex\\bin',
                            'C:\\Program Files\\Git\\cmd', '', root + '-other\\bin', 'D:\\CODEX-tools', 'E:\\keep\\me'])


def test_child_environment_removes_the_listed_prefixes_and_keys_in_any_case_and_keeps_the_rest(tmp_path, monkeypatch):
    make_interpreter(tmp_path, monkeypatch)
    kept = {'HOME': 'h', 'USERPROFILE': 'u', 'SystemRoot': 's', 'TEMP': 't', 'GITHUB_TOKEN': 'g', 'GITLAB': 'x', 'CONDAX': 'c', 'CODEX': 'c', 'OPENAI': 'o',
            'MYGIT_DIR': 'm', 'PYTHONPATHS': 'p', 'PYTHONSTARTUP': 'p', 'LANG': 'l', 'XDG_CACHE_HOME': 'x'}
    # one spelling per name in each base (a base with two spellings of one name is refused, see below); the two bases together give 14 spellings
    for removed in (['CODEX_HOME', 'codex_thread_id', 'Codex_Sandbox', 'OPENAI_API_KEY', 'openai_base_url', 'CONDA_PREFIX', 'Conda_Default_Env',
                     'GIT_DIR', 'git_work_tree', 'GIT_CONFIG_COUNT', 'PYTHONPATH', 'PythonHome'], ['pythonpath', 'PYTHONHOME']):
        base = {**{name: MARK for name in removed}, **kept, 'PATH': 'a'}
        result = guards.child_environment(base)
        assert not {name for name in result if name.upper().startswith(('CODEX_', 'OPENAI_', 'CONDA_', 'GIT_')) or name.upper() in ('PYTHONPATH', 'PYTHONHOME')}
        assert all(result[name] == value for name, value in kept.items())           # the others keep their spelling and value
        assert set(result) == set(kept) | {'PATH'} | set(guards.CHILD_SETTINGS)     # nothing else is added or lost
        assert MARK not in result.values()


def test_child_environment_does_not_change_its_input(tmp_path, monkeypatch):
    make_interpreter(tmp_path, monkeypatch)
    base = {'CODEX_HOME': 'x', 'PATH': 'C:\\Codex' + os.pathsep + 'E:\\keep', 'OMP_NUM_THREADS': '2', 'KEEP': 'k'}
    snapshot = dict(base)
    result = guards.child_environment(base)
    assert base == snapshot and result is not base
    result['KEEP'] = 'changed'
    result['NEW'] = 'n'
    assert base == snapshot                                                           # the result is independent of the input
    assert guards.child_environment(base) is not guards.child_environment(base)
    from types import MappingProxyType
    assert guards.child_environment(MappingProxyType(base))['KEEP'] == 'k'            # any mapping is accepted (os.environ is not a dict)


def test_child_environment_filters_path_by_the_t002_rule(tmp_path, monkeypatch):
    interpreter = make_interpreter(tmp_path, monkeypatch)
    root = str(interpreter.parent)
    entries = [root, os.path.join(root, 'Scripts'), root.upper() + '\\Library\\bin', 'C:\\Windows\\system32', 'C:\\Tools\\Codex\\bin',
               'C:\\Program Files\\Git\\cmd', '', root + '-other\\bin', 'D:\\CODEX-tools', 'E:\\keep\\me']
    result = guards.child_environment({'PATH': os.pathsep.join(entries)})
    # dropped: the interpreter folder and everything that starts with it (case-insensitive prefix, as T002), every entry that contains "codex" in any case
    assert result['PATH'].split(os.pathsep) == ['C:\\Windows\\system32', 'C:\\Program Files\\Git\\cmd', '', 'E:\\keep\\me']


def test_child_environment_handles_a_path_variable_of_any_case_and_a_missing_one(tmp_path, monkeypatch):
    make_interpreter(tmp_path, monkeypatch)
    mixed = guards.child_environment({'Path': 'C:\\Codex' + os.pathsep + 'E:\\keep'})
    assert mixed['Path'] == 'E:\\keep' and 'PATH' not in mixed                      # the Windows spelling is filtered in place, no second PATH appears
    assert guards.child_environment({})['PATH'] == ''                                # as T002: a missing PATH becomes an empty one


@pytest.mark.parametrize('first, second', [('Path', 'PATH'), ('omp_num_threads', 'OMP_NUM_THREADS'), ('git_dir', 'GIT_DIR'), ('Home', 'HOME'),
                                          ('PythonPath', 'pythonpath')])
def test_child_environment_refuses_two_spellings_of_one_name_without_naming_a_value(tmp_path, monkeypatch, first, second):
    """K4 review m5: Windows reads names without case, so a base that holds two spellings of one name is ambiguous. (os.environ cannot hold them there.)"""
    make_interpreter(tmp_path, monkeypatch)
    base = {first: MARK, 'KEEP': 'k', second: MARK + '2'}
    snapshot = dict(base)
    with pytest.raises(GateRefused) as caught:
        guards.child_environment(base)
    assert caught.value.gate == 'environment' and MARK not in str(caught.value)
    assert base == snapshot


def test_child_environment_sets_the_t002_child_settings_over_the_callers_values(tmp_path, monkeypatch):
    make_interpreter(tmp_path, monkeypatch)
    base = {'PYTHONUTF8': '0', 'omp_num_threads': '2', 'Cuda_Visible_Devices': '3', 'MKL_THREADING_LAYER': 'INTEL', 'PATH': 'x'}
    result = guards.child_environment(base)
    assert {name: result[name] for name in guards.CHILD_SETTINGS} == dict(guards.CHILD_SETTINGS)
    assert len({name.upper() for name in result}) == len(result)                    # one spelling per name: Windows reads names without case
    assert 'omp_num_threads' not in result and 'Cuda_Visible_Devices' not in result
    assert guards.CHILD_SETTINGS == {
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'MKL_THREADING_LAYER': 'SEQUENTIAL',
        'KMP_DUPLICATE_LIB_OK': 'FALSE', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'CUDA_VISIBLE_DEVICES': '0'}


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def test_child_environment_defaults_are_the_t002_values_and_explicit_defaults_change_nothing(tmp_path, monkeypatch):
    """K4 review I1: the override keywords default to the T002 child (device 0, 8 threads); the dict is exactly what it was before the keywords."""
    make_interpreter(tmp_path, monkeypatch)
    bases = [{}, {'PATH': 'a' + os.pathsep + 'b', 'KEEP': 'k'}, {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'PATH': 'x'},
             {'CODEX_HOME': 'x', 'PYTHONUTF8': '0', 'MKL_NUM_THREADS': '2', 'TEMP': 't'}, dict(os.environ)]
    for base in bases:
        plain = guards.child_environment(base)
        assert plain == guards.child_environment(base, cuda_visible_devices='0', threads=8)
        assert plain['CUDA_VISIBLE_DEVICES'] == '0' and all(plain[name] == '8' for name in THREAD_VARIABLES)
    assert guards.child_environment({'KEEP': 'k'}) == {
        'KEEP': 'k', 'PATH': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
        'MKL_THREADING_LAYER': 'SEQUENTIAL', 'KMP_DUPLICATE_LIB_OK': 'FALSE', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': '0'}
    assert guards.THREAD_KEYS == THREAD_VARIABLES and guards.DEFAULT_THREADS == 8
    assert {name for name in guards.CHILD_SETTINGS if name.endswith('_NUM_THREADS')} == set(guards.THREAD_KEYS)   # every thread variable of the settings takes ``threads``


def test_child_environment_override_gives_a_cpu_child_no_device_and_one_thread(tmp_path, monkeypatch):
    """K4 review I1: a CPU or test child (the harness sets CUDA_VISIBLE_DEVICES='' and 1 thread) must not be given GPU 0 and 8 threads back."""
    make_interpreter(tmp_path, monkeypatch)
    base = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PATH': 'x', 'KEEP': 'k'}
    snapshot = dict(base)
    cpu = guards.child_environment(base, cuda_visible_devices='', threads=1)
    assert cpu['CUDA_VISIBLE_DEVICES'] == '' and all(cpu[name] == '1' for name in THREAD_VARIABLES)
    assert base == snapshot                                                           # the input is not mutated
    default = guards.child_environment(base)
    assert {k: v for k, v in cpu.items() if k not in ('CUDA_VISIBLE_DEVICES', *THREAD_VARIABLES)} == \
           {k: v for k, v in default.items() if k not in ('CUDA_VISIBLE_DEVICES', *THREAD_VARIABLES)}       # nothing else moves
    other = guards.child_environment(base, cuda_visible_devices='0,1', threads=16)
    assert other['CUDA_VISIBLE_DEVICES'] == '0,1' and all(other[name] == '16' for name in THREAD_VARIABLES)
    assert guards.child_environment(base, threads=guards.MAX_THREADS)['OMP_NUM_THREADS'] == str(guards.MAX_THREADS)
    assert guards.child_environment(base, cuda_visible_devices='')['OMP_NUM_THREADS'] == '8'          # each keyword on its own


@pytest.mark.parametrize('devices', ['a', '0 ', ' 0', '0;1', '-1', '0\n', 'GPU-' + MARK, '٣', 0, None, b'0', ['0'],
                                     ',', ',,', '0,', ',0', '0,,1', '0,1,', '00,1,2,'])             # K4 re-check R-n5: an empty item is refused
def test_child_environment_refuses_a_device_list_that_is_not_digits_and_commas(tmp_path, monkeypatch, devices):
    make_interpreter(tmp_path, monkeypatch)
    with pytest.raises(GateRefused) as caught:
        guards.child_environment({'KEEP': 'k'}, cuda_visible_devices=devices)
    assert caught.value.gate == 'environment' and MARK not in str(caught.value)


@pytest.mark.parametrize('devices', ['', '0', '1', '0,1', '0,1,2', '10', '3,0', '00'])
def test_child_environment_accepts_the_empty_text_and_comma_separated_numbers(tmp_path, monkeypatch, devices):
    """The narrowed format (R-n5): no device at all (the empty text) or one or more numbers joined by single commas."""
    make_interpreter(tmp_path, monkeypatch)
    assert guards.child_environment({'KEEP': 'k'}, cuda_visible_devices=devices)['CUDA_VISIBLE_DEVICES'] == devices


@pytest.mark.parametrize('threads', [0, -1, True, False, 1.5, 8.0, '8', None, 1025, 10 ** 9, 10 ** 400])
def test_child_environment_refuses_a_thread_count_that_is_not_a_whole_number_in_range(tmp_path, monkeypatch, threads):
    make_interpreter(tmp_path, monkeypatch)
    with pytest.raises(GateRefused) as caught:
        guards.child_environment({'KEEP': 'k'}, threads=threads)
    assert caught.value.gate == 'environment'


def test_child_environment_takes_the_interpreter_folder_from_the_resolved_executable(tmp_path, monkeypatch):
    """K4 review m6: ``Path(sys.executable).resolve().parent``, as ``native_policy`` resolves it (an alias such as a junction or an 8.3 name is not the folder)."""
    real, alias = tmp_path / 'real_py', tmp_path / 'alias_py'
    real.mkdir()
    try:
        import _winapi
        _winapi.CreateJunction(str(real), str(alias))                                # a Windows junction: no process, no privilege needed
    except (ImportError, AttributeError, OSError):
        pytest.skip('a junction could not be made here')
    monkeypatch.setattr(sys, 'executable', str(alias / 'python.exe'))
    result = guards.child_environment({'PATH': os.pathsep.join([str(real), str(real / 'Scripts'), 'E:\\keep\\me'])})
    assert result['PATH'] == 'E:\\keep\\me'                                           # the entries under the real folder are dropped


def test_child_environment_agrees_with_the_frozen_t002_controller_on_the_same_inputs(tmp_path, monkeypatch):
    interpreter = make_interpreter(tmp_path, monkeypatch)
    frozen = frozen_controller_child_environment(interpreter)
    path = sample_path(interpreter)
    battery = [
        {},
        {'PATH': path},
        {'HOME': 'x', 'GITHUB_TOKEN': 'x', 'PATH': 'a' + os.pathsep + 'b'},
        {'PATH': path, 'CODEX_HOME': 'x', 'OPENAI_API_KEY': 'x', 'CONDA_PREFIX': 'x', 'GIT_DIR': 'x', 'GIT_CONFIG_COUNT': '1', 'PYTHONPATH': 'x', 'PYTHONHOME': 'x',
         'PYTHONUTF8': '0', 'OMP_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '1', 'MKL_THREADING_LAYER': 'INTEL', 'KEEP': 'k', 'TEMP': 't'},
        dict(os.environ),
    ]
    for base in battery:
        assert guards.child_environment(base) == frozen(dict(base)), sorted(base)


def test_child_environment_constants_are_pinned_to_the_t002_controller_source_by_syntax_tree():
    function = controller_function_node()
    tuples = [tuple(e.value for e in n.elts) for n in ast.walk(function) if isinstance(n, ast.Tuple) and all(isinstance(e, ast.Constant) for e in n.elts)]
    assert guards.REMOVED_PREFIXES in tuples
    assert guards.REMOVED_KEYS in tuples
    updates = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'update']
    assert len(updates) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in updates[0].keywords} == dict(guards.CHILD_SETTINGS)
    assert 'codex' in {n.value for n in ast.walk(function) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert guards.PATH_BANNED_TEXT == 'codex'


def test_child_environment_settings_cover_the_other_two_copies_except_the_t009_extras():
    """The T009 copy sets two more thread variables and no KMP guard; the DEV evaluation copy equals the T002 settings but for PYTHONIOENCODING."""
    sources = {'comparison': REPO / 'c_ept/comparison/operator.py', 'dev': REPO / 'tools/ept_dev_evaluate.py'}
    found = {}
    for name, path in sources.items():
        function = next(n for n in parse(path).body if isinstance(n, ast.FunctionDef) and n.name in ('child_environment', 'child_env'))
        update = next(n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'update')
        found[name] = {k.arg: ast.literal_eval(k.value) for k in update.keywords}
    kernel = dict(guards.CHILD_SETTINGS)
    assert {k: v for k, v in found['dev'].items()} == {k: v for k, v in kernel.items() if k != 'PYTHONIOENCODING'}
    assert set(found['comparison']) - set(kernel) == {'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'}        # recorded, not adopted (K4 report)
    assert {k: v for k, v in found['comparison'].items() if k in kernel} == {k: kernel[k] for k in found['comparison'] if k in kernel}
    assert set(kernel) - set(found['comparison']) == {'PYTHONUNBUFFERED', 'PYTHONIOENCODING', 'KMP_DUPLICATE_LIB_OK'}


def test_child_environment_refuses_non_text_entries_and_an_unknown_interpreter(tmp_path, monkeypatch):
    make_interpreter(tmp_path, monkeypatch)
    for base in ({'A': 1}, {1: 'a'}, {'A': None}, {'A': b'x'}):
        with pytest.raises(TypeError) as caught:
            guards.child_environment(base)
        assert MARK not in str(caught.value)
    monkeypatch.setattr(sys, 'executable', '')
    with pytest.raises(GateRefused) as refused:
        guards.child_environment({'PATH': 'x'})
    assert refused.value.gate == 'environment'


# ---- spec_check ------------------------------------------------------------------------------------------------------------------------
def write_record(path: Path, **changes) -> str:
    """A synthetic decision record whose operating point equals the constants of ``c_ept.config``; returns the SHA-256 of its bytes."""
    from c_ept import config
    point = {'G_ms': config.G_MS, 'L_u': config.L_U, 'tau_lead_ms': config.TAU_LEAD_MS, 'presence_gate': None,
             'size_classes': {'T': 1, 'S': 2, 'ASYM': 3, 'P': 4}}
    point.update(changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({'operating_point': point, 'note': 'synthetic'}, indent=2).encode('utf-8'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listing(root: Path):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob('*'))


def test_spec_check_returns_the_hash_of_the_record_bytes_when_the_record_matches_the_code(tmp_path):
    expected = write_record(tmp_path / 'SPEC.json')
    assert guards.spec_check(tmp_path / 'SPEC.json') == expected
    assert guards.spec_check(str(tmp_path / 'SPEC.json'), expected_sha256=expected) == expected


@pytest.mark.parametrize('change', [{'G_ms': 1}, {'L_u': 0.5}, {'tau_lead_ms': 7}, {'presence_gate': 3}, {'size_classes': {'T': 1}}])
def test_spec_check_refuses_a_record_that_differs_from_the_code_before_anything_is_created(tmp_path, change):
    record = tmp_path / 'in' / 'SPEC.json'
    write_record(record, **change)
    run_dir, lock = tmp_path / 'runs' / '20261101_k4_1', tmp_path / 'runs' / 'k4.lock'
    before = listing(tmp_path)

    def start_a_run():                                                                  # what the supervisor does: the gate first, the reservation after
        guards.spec_check(record)
        run_dir.mkdir(parents=True)
        lock.write_text('lock')

    with pytest.raises(GateRefused) as caught:
        start_a_run()
    assert caught.value.gate == 'spec'
    assert not run_dir.exists() and not lock.exists() and not run_dir.parent.exists()
    assert listing(tmp_path) == before                                                  # not even a temporary file


def test_spec_check_refuses_an_unreadable_or_malformed_record(tmp_path):
    cases = {'missing.json': None, 'empty.json': b'', 'text.json': b'not json ' + MARK.encode(), 'bytes.json': b'\xff\xfe\x00',
             'list.json': b'[1, 2]', 'no_point.json': b'{"x": 1}', 'point_list.json': b'{"operating_point": [1]}',
             'deep.json': b'[' * 200_000 + b']' * 200_000}                               # nested too deeply for the parser: RecursionError, not a crash
    for name, content in cases.items():
        if content is not None:
            (tmp_path / name).write_bytes(content)
        with pytest.raises(GateRefused) as caught:
            guards.spec_check(tmp_path / name)
        assert caught.value.gate == 'spec' and MARK not in str(caught.value), name
        assert caught.value.__cause__ is None and caught.value.__context__ is None, name      # no chain: the parser error with the document is not kept


PATH_MARK, DIR_MARK = 'LEAKED_RECORD_NAME_MARKER', 'LEAKED_RECORD_DIR_MARKER'


def refusal_text(error: BaseException) -> str:
    return str(error) + '|' + ''.join(traceback.format_exception(type(error), error, error.__traceback__))


def test_spec_refusal_carries_neither_the_record_path_nor_a_record_value_and_names_the_kind_of_failure(tmp_path):
    """K4 review m1: ``check_spec`` puts the record path and ``repr`` of record values into its message; the kernel's refusal is fixed text, with no chain."""
    record = tmp_path / DIR_MARK / (PATH_MARK + '.json')
    expected = write_record(record, G_ms=MARK, L_u=[MARK], presence_gate=MARK)           # a constant mismatch whose values are markers
    cases = {}
    with pytest.raises(GateRefused) as mismatch:
        guards.spec_check(record)
    cases['constant mismatch'] = mismatch.value
    with pytest.raises(GateRefused) as unreadable:
        guards.spec_check(tmp_path / DIR_MARK / (PATH_MARK + '_absent.json'))
    cases['unreadable'] = unreadable.value
    good = tmp_path / DIR_MARK / (PATH_MARK + '_good.json')
    good_hash = write_record(good)
    with pytest.raises(GateRefused) as wrong_hash:
        guards.spec_check(good, expected_sha256='0' * 64 if good_hash != '0' * 64 else '1' * 64)
    cases['hash mismatch'] = wrong_hash.value
    assert expected != good_hash
    for kind, error in cases.items():
        assert error.gate == 'spec'
        text = refusal_text(error)
        for leak in (PATH_MARK, DIR_MARK, MARK, str(tmp_path), 'SPEC_C_DEFINITION'):          # the record path, the folder, a record value, the configured name
            assert leak not in text, (kind, leak)
        assert error.__cause__ is None and error.__context__ is None
        assert kind in str(error), (kind, str(error))                                  # the fixed text names the kind of failure
    assert len({str(e) for e in cases.values()}) == 3


def test_spec_check_documents_that_it_reads_through_check_spec_and_not_through_run_io():
    """K4 review m2: the one exception to "every kernel module reads through ``run.io``" is written down where a reader will look."""
    for text in (guards.spec_check.__doc__, guards.__doc__):
        assert 'the one exception to "every kernel module reads through ``run.io``"' in text and 'check_spec' in text
    assert 'D, L' not in guards.__doc__                                                  # K4 review m9: the symbol table has no unused lock


def test_spec_check_compares_the_hash_with_the_expected_one_when_one_is_given(tmp_path):
    expected = write_record(tmp_path / 'SPEC.json')
    other = '0' * 64 if expected != '0' * 64 else '1' * 64
    with pytest.raises(GateRefused) as caught:
        guards.spec_check(tmp_path / 'SPEC.json', expected_sha256=other)
    assert caught.value.gate == 'spec' and other not in str(caught.value)
    for bad in ('abc', expected.upper(), 5, expected + '0'):
        with pytest.raises(ValueError):
            guards.spec_check(tmp_path / 'SPEC.json', expected_sha256=bad)


def test_spec_check_without_a_path_reads_the_configured_decision_record_at_call_time(tmp_path, monkeypatch):
    from c_ept import config
    expected = write_record(tmp_path / 'configured.json')
    monkeypatch.setattr(config, 'SPEC_PATH', tmp_path / 'configured.json')
    assert guards.spec_check() == expected
    monkeypatch.setattr(config, 'SPEC_PATH', tmp_path / 'absent.json')
    with pytest.raises(GateRefused):
        guards.spec_check()


# ---- e6 --------------------------------------------------------------------------------------------------------------------------------
CLEAN = [(4, 'System', None), (410, 'python.exe', ['python.exe', 'train.py']), (512, 'chrome.exe', None),
         (777, 'node.exe', ['node', 'D:/x/strata_worker_mcp/index.js'])]            # the last is an idle MCP bridge: no model, never counted
SERVER = (4242, 'python.exe', ['python.exe', 'D:\\Strata\\serve\\server.py', '--port', '8000', MARK])
LLAMA = (4243, 'llama-server.exe', ['llama-server.exe', '-m', 'x.gguf'])
OLLAMA = (4244, 'ollama.exe', None)


def test_e6_passes_on_a_listing_without_a_model_server_and_returns_the_check_record():
    record = guards.e6(processes=CLEAN)
    assert record['running'] is False and record['matched'] == []
    assert (record['scanned'], record['bridges_ignored'], record['cmdline_unreadable']) == (4, 1, 2)
    assert isinstance(record['rule'], str) and record['rule']


@pytest.mark.parametrize('server', [SERVER, LLAMA, OLLAMA], ids=['strata-command-line', 'llama-name', 'ollama-name'])
def test_e6_refuses_while_a_model_server_runs_and_names_it_by_name_and_pid_only(server):
    with pytest.raises(GateRefused) as caught:
        guards.e6(processes=[*CLEAN, server])
    assert caught.value.gate == 'e6'
    assert f'{server[1]} (pid {server[0]})' in str(caught.value)
    assert MARK not in str(caught.value) and 'server.py' not in str(caught.value)        # a command line is never shown


def test_e6_refuses_when_the_check_cannot_be_made_as_t002_does(monkeypatch):
    with pytest.raises(GateRefused) as unavailable:                                     # the probe has no answer (no psutil)
        guards.e6(probe=lambda: None)
    assert unavailable.value.gate == 'e6'
    from c_ept import runlog
    monkeypatch.setattr(runlog, 'psutil', None)
    with pytest.raises(GateRefused):                                                    # the default probe without psutil
        guards.e6()

    def broken():
        raise PermissionError(MARK)

    with pytest.raises(GateRefused) as failed:                                          # any probe error: refused, with the error's type only
        guards.e6(probe=broken)
    assert 'PermissionError' in str(failed.value) and MARK not in str(failed.value) and failed.value.__cause__ is not None


def test_e6_does_not_list_processes_when_a_listing_is_injected(monkeypatch):
    from c_ept import runlog

    class NoListing:
        @staticmethod
        def process_iter(*args, **kwargs):
            raise AssertionError('the real process table was read')

    monkeypatch.setattr(runlog, 'psutil', NoListing)
    assert guards.e6(processes=CLEAN)['running'] is False
    with pytest.raises(GateRefused) as caught:                                          # and the default probe does read it (its error is a refusal)
        guards.e6()
    assert 'AssertionError' in str(caught.value)


def test_e6_default_probe_is_the_runlog_detection_rule(monkeypatch):
    from c_ept import runlog
    calls = []
    record = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 0, 'cmdline_unreadable': 0, 'rule': 'r'}
    monkeypatch.setattr(runlog, 'strata_check', lambda processes=None: calls.append(processes) or dict(record))
    assert guards.e6() == record
    assert guards.e6(processes=CLEAN) == record
    assert calls == [None, CLEAN]
    with pytest.raises(ValueError):
        guards.e6(processes=CLEAN, probe=lambda: record)


VALID_CLEAN = {'running': False, 'matched': [], 'bridges_ignored': 1, 'scanned': 4, 'cmdline_unreadable': 1, 'rule': 'a rule'}
VALID_RUNNING = {'running': True, 'matched': [{'pid': 7, 'name': 'llama'}, {'pid': 9, 'name': 'strata'}], 'bridges_ignored': 0, 'scanned': 3,
                 'cmdline_unreadable': 0, 'rule': 'a rule'}


def mutated(base, **changes):
    result = json.loads(json.dumps(base))
    result.update(changes)
    return result


RECORDS = {
    'valid-clean': VALID_CLEAN, 'valid-running': VALID_RUNNING, 'none': None, 'list': [], 'string': 'running',
    'extra-key': mutated(VALID_CLEAN, extra=1), 'missing-key': {k: v for k, v in VALID_CLEAN.items() if k != 'scanned'},
    'running-int': mutated(VALID_CLEAN, running=0), 'matched-tuple': mutated(VALID_CLEAN, matched=()),
    'rule-empty': mutated(VALID_CLEAN, rule=''), 'rule-blank': mutated(VALID_CLEAN, rule='  '), 'rule-int': mutated(VALID_CLEAN, rule=5),
    'scanned-negative': mutated(VALID_CLEAN, scanned=-1), 'scanned-bool': mutated(VALID_CLEAN, scanned=True),
    'scanned-float': mutated(VALID_CLEAN, scanned=4.0), 'bridges-negative': mutated(VALID_CLEAN, bridges_ignored=-1),
    'unreadable-str': mutated(VALID_CLEAN, cmdline_unreadable='1'),
    'entry-extra-key': mutated(VALID_RUNNING, matched=[{'pid': 7, 'name': 'a', 'cmd': 'x'}]),
    'entry-pid-negative': mutated(VALID_RUNNING, matched=[{'pid': -7, 'name': 'a'}]),
    'entry-pid-bool': mutated(VALID_RUNNING, matched=[{'pid': True, 'name': 'a'}]),
    'entry-name-int': mutated(VALID_RUNNING, matched=[{'pid': 7, 'name': 5}]),
    'entry-not-dict': mutated(VALID_RUNNING, matched=[[7, 'a']]),
    'duplicate-pids': mutated(VALID_RUNNING, matched=[{'pid': 7, 'name': 'a'}, {'pid': 7, 'name': 'b'}]),
    'unsorted': mutated(VALID_RUNNING, matched=[{'pid': 9, 'name': 'a'}, {'pid': 7, 'name': 'b'}]),
    'running-without-match': mutated(VALID_CLEAN, running=True), 'match-without-running': mutated(VALID_RUNNING, running=False),
    'matched-over-scanned': mutated(VALID_RUNNING, scanned=1), 'bridges-over-scanned': mutated(VALID_CLEAN, bridges_ignored=5),
    'unreadable-over-scanned': mutated(VALID_CLEAN, cmdline_unreadable=4, bridges_ignored=1),
}


def test_e6_record_check_agrees_with_the_frozen_validate_e6_probe_on_the_same_records():
    frozen = load_frozen_s4a_runtime()
    outcomes = {}
    for name, record in RECORDS.items():
        try:
            frozen.validate_e6_probe(copy.deepcopy(record))
            old = 'accepted'
        except frozen.RuntimeRefused as error:
            # the frozen messages: "E6 확인 불가" (the record cannot be confirmed) before "E6 거부" (a server runs); the order is part of the rule
            old = 'unconfirmed' if '확인 불가' in str(error) else 'running' if 'E6 거부' in str(error) else 'other'
        try:
            guards.e6(probe=lambda r=record: copy.deepcopy(r))
            new = 'accepted'
        except GateRefused as error:
            assert error.gate == 'e6'
            new = 'unconfirmed' if 'cannot be confirmed' in str(error) else 'running' if 'is running' in str(error) else 'other'
        assert new == old, (name, old, new)
        outcomes[name] = new
    assert 'other' not in outcomes.values()
    assert [n for n, o in outcomes.items() if o == 'accepted'] == ['valid-clean']          # only the one valid clean record passes
    assert [n for n, o in outcomes.items() if o == 'running'] == ['valid-running']          # the only record that is well formed and says a server runs
    assert sum(o == 'unconfirmed' for o in outcomes.values()) == len(RECORDS) - 2


def test_e6_fits_the_e6guard_counter_of_the_kernel_runtime():
    from c_ept.runtime import E6Guard
    clean = E6Guard(lambda: guards.e6(processes=CLEAN))
    clean()
    clean()
    assert clean.calls == 2
    busy = E6Guard(lambda: guards.e6(processes=[*CLEAN, LLAMA]))
    with pytest.raises(GateRefused):
        busy()
    assert busy.calls == 1


# ---- Budget ----------------------------------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_budget_passes_at_the_wall_bound_and_refuses_past_it():
    clock = Clock()
    budget = Budget(10, 100, clock=clock, measure=lambda: 1)
    for elapsed in (0.0, 4.5, 10.0):
        clock.now = 1000.0 + elapsed
        budget.check()
        assert budget.elapsed_s() == elapsed
    clock.now = 1000.0 + 10.001
    with pytest.raises(GateRefused) as caught:
        budget.check()
    assert caught.value.gate == 'budget' and 'wall' in str(caught.value)


def test_budget_passes_at_the_memory_bound_and_refuses_past_it():
    clock, used = Clock(), [100]
    budget = Budget(10, 100, clock=clock, measure=lambda: used[0])
    budget.check()                                                                      # at the bound: passes
    used[0] = 101
    with pytest.raises(GateRefused) as caught:
        budget.check()
    assert caught.value.gate == 'budget' and 'memory' in str(caught.value)
    assert budget.peak_bytes == 101 and budget.samples == 2                             # the refused sample is recorded


def test_budget_reads_the_memory_reader_as_a_number_or_as_the_pair_of_the_runtime_sampler():
    budget = Budget(10, 100, clock=Clock(), measure=lambda: (60, 90))                   # (private bytes, lifetime upper bound), as tree_memory
    budget.check()
    assert budget.peak_bytes == 60
    over = Budget(10, 100, clock=Clock(), measure=lambda: (101, None))
    with pytest.raises(GateRefused):
        over.check()


def test_budget_keeps_the_largest_upper_bound_of_the_pairs_as_peak_upper():
    """K4 review m4: ``evaluation.runtime.Budget`` kept ``peak_upper`` (the lifetime bound of the process tree); an unknown bound (None) erases nothing."""
    clock, samples = Clock(), iter([(60, None), (50, 90), (70, 80), (40, None), 30])
    budget = Budget(10, 100, clock=clock, measure=lambda: next(samples))
    seen = []
    for _ in range(5):
        budget.check()
        seen.append(budget.peak_upper)
    assert seen == [None, 90, 90, 90, 90] and budget.peak_bytes == 70 and budget.samples == 5
    refused = Budget(10, 100, clock=clock, measure=lambda: (101, 500))
    with pytest.raises(GateRefused):
        refused.check()
    assert refused.peak_upper == 500                                                    # the refused sample is recorded too


@pytest.mark.parametrize('sample', [0, -5, None, 1.5, True, 'x', (0, None), (None, 5), (), (5,), (1, 2, 3), (5, True), (5, -1), (5, 1.5), (5, '9'), math.nan])
def test_budget_refuses_when_the_memory_cannot_be_read(sample):
    budget = Budget(10, 100, clock=Clock(), measure=lambda: sample)
    with pytest.raises(GateRefused) as caught:
        budget.check()
    assert caught.value.gate == 'budget' and 'memory' in str(caught.value)
    assert budget.samples == 0 and budget.peak_upper is None                            # nothing unreadable is recorded


def test_budget_tracks_the_peak_of_the_samples():
    clock, used = Clock(), iter([10, 70, 30])
    budget = Budget(10, 100, clock=clock, measure=lambda: next(used))
    for _ in range(3):
        budget.check()
    assert (budget.peak_bytes, budget.samples) == (70, 3)


def test_budget_refuses_a_clock_that_went_backwards_and_does_not_read_memory_after_a_wall_refusal():
    clock, reads = Clock(), []
    budget = Budget(10, 100, clock=clock, measure=lambda: reads.append(1) or 1)
    clock.now = 999.0
    with pytest.raises(GateRefused) as backwards:
        budget.check()
    assert backwards.value.gate == 'budget' and 'clock' in str(backwards.value) and not reads
    clock.now = 2000.0
    with pytest.raises(GateRefused):
        budget.check()
    assert not reads                                                                    # the wall refusal came first


def test_budget_default_clock_and_memory_reader_are_the_monotonic_clock_and_the_process_tree_sampler(monkeypatch):
    from c_ept.runtime import resources
    monkeypatch.setattr(resources, 'tree_memory', lambda: (123, None))
    budget = Budget(3600, 1 << 30)
    budget.check()
    assert budget.peak_bytes == 123 and 0 <= budget.elapsed_s() < 60
    monkeypatch.setattr(resources, 'tree_memory', lambda: (0, None))                    # no psutil: tree_memory says (0, None), which is no evidence
    with pytest.raises(GateRefused):
        budget.check()


@pytest.mark.parametrize('wall', [0, -1, math.nan, math.inf, True, '10', None])
def test_budget_refuses_a_wall_bound_that_is_not_a_positive_finite_number(wall):
    with pytest.raises(ValueError):
        Budget(wall, 100, clock=Clock(), measure=lambda: 1)


@pytest.mark.parametrize('memory', [0, -1, 1.5, True, '100', None])
def test_budget_refuses_a_memory_bound_that_is_not_a_positive_whole_number(memory):
    with pytest.raises(ValueError):
        Budget(10, memory, clock=Clock(), measure=lambda: 1)


def test_budget_refuses_a_clock_that_does_not_give_a_finite_number():
    for bad in (lambda: math.nan, lambda: 'now', lambda: None):
        with pytest.raises(ValueError):
            Budget(10, 100, clock=bad, measure=lambda: 1)


def test_budget_refuses_numbers_beyond_the_float_range_without_an_overflow_error():
    """K4 review m4: ``math.isfinite(10 ** 400)`` raises ``OverflowError``; the bounds and the clock are refused with ``ValueError`` instead."""
    huge = 10 ** 400
    for wall in (huge, -huge):
        with pytest.raises(ValueError):
            Budget(wall, 100, clock=Clock(), measure=lambda: 1)
    for memory in (huge, -huge, (1 << 63) + 1, 1 << 64):
        with pytest.raises(ValueError):
            Budget(10, memory, clock=Clock(), measure=lambda: 1)
    Budget(10, 1 << 62, clock=Clock(), measure=lambda: 1)                               # a byte count that a host could have is accepted
    Budget(10, 1 << 63, clock=Clock(), measure=lambda: 1)                               # the largest byte count that is accepted (2^63, inclusive)
    with pytest.raises(ValueError):
        Budget(10, 100, clock=lambda: huge, measure=lambda: 1)
    clock = Clock()
    budget = Budget(10, 100, clock=clock, measure=lambda: 1)
    clock.now = huge
    with pytest.raises(ValueError):                                                     # a reading beyond the range, at a check
        budget.check()
    clock.now = 1000.0
    big = Budget(10 ** 300, 100, clock=clock, measure=lambda: 10 ** 400)                # a huge sample is only a sample: it is above the bound
    with pytest.raises(GateRefused):
        big.check()
