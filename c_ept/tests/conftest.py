"""Shared fixtures and helpers of the c_ept tests.

Match keys are computed at run time from the cache listing; no match key is written into the repository. The helpers are the key-free
JSON reader (``read_json``), the golden-lock lookup (``golden_folder``), the E6 process list (``empty_process_list``,
``fake_processes``) and the import scans of the structure tests (``module_level_imports``, ``all_imports``).
"""
import ast
import contextlib
import importlib.util
import json
import os
import sys
import types
import warnings
from pathlib import Path

import pytest

PROTO = Path(r'D:\LOL_Data\runs\20261019_link_rule_fit_1')       # the prototype of the link rule (read only)


class UnreadableTestFile(Exception):
    """A data file that a test reads cannot be read. The message names the kind of file and what is wrong, never the path: the name of
    a cache or raw match file is the match key."""


def read_json(path, kind: str):
    """The parsed JSON of the UTF-8 file ``path`` for a test; ``kind`` names the file in words, for example ``'cache events.json'``.

    ``Path.read_text`` raises FileNotFoundError with the whole path in it, and a short traceback prints it: for a cache or raw match file
    the path holds the match key. So an OSError (missing, a folder, no permission) or a ValueError (a JSONDecodeError or a
    UnicodeDecodeError is one) becomes an ``UnreadableTestFile`` that names ``kind`` and the problem only. It is raised after the
    ``except`` clauses and ``from None``, so it has neither a cause nor a context: nothing in the exception chain holds the path.
    """
    problem = None
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        problem = 'missing'
    except OSError as err:
        problem = f'unreadable ({type(err).__name__})'
    except ValueError:
        problem = 'is not valid UTF-8 JSON'
    if problem is not None:
        raise UnreadableTestFile(f'{kind} {problem}') from None
    return data


# ---- the import scans of the structure tests ----------------------------------------------------------------------------------------
# Pure test metadata, independently checked against module-layout-v1 by
# test_import_guard_aliases. No production module or protected source is loaded.
IMPORT_GUARD_GROUPS = {
    'data': 'casts reader store_core store_build event_rows frame_rows head_rows'.split(),
    'preprocessing': 'features terrain tokens traits'.split(),
    'domain': 'units targets state symmetry'.split(),
    'valuation': 'flowprice rulepay towertier value compose'.split(),
    'models': 'baselines_b0 baselines_b1 heads nll'.split(),
    'workflows': 'run_s1 run_s2 run_s3 run_s4a s4a_operator s4a_runtime'.split(),
    'reporting': 'report_s1 report_s2 report_s3'.split(),
    'checks': ['gate_s3'],
}
IMPORT_GUARD_ALIASES = {f'c_ept.{name}': f'c_ept.{group}.{name}'
                        for group, names in IMPORT_GUARD_GROUPS.items() for name in names}


def _equivalent_names(names, pairs):
    result = set(names)
    for name in tuple(result):
        for source, target in pairs:
            if name == source or name.startswith(source + '.'):
                result.add(target + name[len(source):])
    return result


def legacy_import_aliases(names) -> set:
    """Retain canonical names and add equivalent legacy names, never the reverse.

    Legacy-only scanner inputs keep their exact historical result sets. Prefix
    matches require a dot boundary, including imported members such as Store.
    """
    return _equivalent_names(names, tuple((new, old) for old, new in IMPORT_GUARD_ALIASES.items()))


def import_watch_names(names) -> tuple:
    """Both spellings for sys.modules watchers; unlike scanner normalization."""
    pairs = tuple(IMPORT_GUARD_ALIASES.items())
    return tuple(sorted(_equivalent_names(names, pairs + tuple((b, a) for a, b in pairs))))


# Fresh-interpreter guards fail before reading protected module bodies. Unlike
# the outer synthetic test runner's cohort stub, this does not hide imports of a
# forbidden chain member: attempting to load it must fail the subprocess test.
PROTECTED_IMPORT_PRELUDE = '''
import sys
class _DenyProtected:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('c_ept.cohort', 'research_pipeline.cohort_policy') or fullname.startswith('research_pipeline.holdout_r3'):
            raise AssertionError('protected import forbidden in guard test: ' + fullname)
sys.meta_path.insert(0, _DenyProtected())
'''


def package_of(path) -> str:
    """The package that a module file of the repository belongs to, dotted: ``c_ept`` for ``c_ept/tokens.py`` (and for
    ``c_ept/__init__.py``), ``c_ept.checks`` for ``c_ept/checks/common.py``. ValueError for a file outside the repository: the caller
    gives the package of such a file (a test's temporary module) itself."""
    root = Path(__file__).resolve().parents[2]
    try:
        folder = Path(path).resolve().parent.relative_to(root)
    except ValueError:
        raise ValueError(f'{path} is not in the repository: give the package of the module') from None
    return '.'.join(folder.parts)


def imported_names(node, package: str) -> set:
    """The absolute names that one ``import`` or ``from ... import`` statement (an ``ast.Import`` or ``ast.ImportFrom`` node) names in a
    module of ``package``: ``a.b`` for ``import a.b``; ``a``, ``a.b`` and ``a.c`` for ``from a import b, c``. A relative import is
    resolved the way the interpreter does it: in package ``p``, ``from . import x`` names ``p`` and ``p.x``; in package ``p.q``,
    ``from ..m import y`` names ``p.m`` and ``p.m.y``. Read as an absolute name, ``from . import run_s1`` was ``None.run_s1`` and a scan
    for an S1 module missed it (final review, Task 12). Canonical grouped names
    additionally expose equivalent legacy names so existing prohibitions cannot
    be bypassed; legacy-only input results are unchanged. ValueError for a
    relative import that climbs above the top-level package."""
    if isinstance(node, ast.Import):
        return legacy_import_aliases({a.name for a in node.names})
    base = node.module or ''
    if node.level:
        parts = package.split('.') if package else []
        if node.level > len(parts):
            raise ValueError(f'a relative import of level {node.level} in the package {package!r} climbs above the top-level package')
        base = '.'.join(parts[:len(parts) - node.level + 1] + ([node.module] if node.module else []))
    return legacy_import_aliases({base} | {f'{base}.{a.name}' for a in node.names})


def module_level_imports(path, package: str | None = None) -> set:
    """Every module that a statement at module level imports (also inside ``if``/``try``/``with``, not inside a function or a class), by
    absolute name (``imported_names``). ``package``: that of the file, ``package_of(path)`` unless given."""
    package = package_of(path) if package is None else package
    found = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found.update(imported_names(node, package))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for part in ('body', 'orelse', 'finalbody'):
                    visit(getattr(node, part, []))
                for handler in getattr(node, 'handlers', []):
                    visit(handler.body)
    visit(ast.parse(Path(path).read_text(encoding='utf-8')).body)
    return found


def all_imports(path, package: str | None = None) -> set:
    """Every module that any Import or ImportFrom node of the file names, wherever it stands (also inside a function or a class), by
    absolute name (``imported_names``). ``package``: that of the file, ``package_of(path)`` unless given."""
    package = package_of(path) if package is None else package
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8'))):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update(imported_names(node, package))
    return found


def pytest_configure(config):
    """Keep match keys out of the test output (Rulings 14 and 33).

    The default traceback (``--tb=auto``) and ``--tb=long`` print the arguments of the failing test, and ``--showlocals`` its locals: a
    fixture such as ``raw_fixture`` is a match key. So ``auto`` and ``long`` become ``short`` (the other styles print no values) and
    ``--showlocals`` / ``-l`` is refused. What an assert itself prints is the test's own business: asserts that compare keys or records
    compare booleans, counts or digests (and ``MatchRecord.__repr__`` hides the key).
    """
    if config.getoption('showlocals'):
        raise pytest.UsageError('--showlocals (-l) is refused in c_ept: the locals of a failing test can hold match keys. Use --tb=short.')
    if config.getoption('fulltrace'):
        raise pytest.UsageError('--full-trace is refused in c_ept: full tracebacks can print match keys. Drop the flag.')
    if config.option.tbstyle in ('auto', 'long'):
        config.option.tbstyle = 'short'


def _install_process_table(monkeypatch, table) -> None:
    """Make ``psutil.process_iter`` list ``table`` [(pid, name, command line: a list, a string or None), ...], not the machine's processes.

    The fake honours ``attrs`` like the real one: ``.info`` holds exactly the requested attributes (``ad_value`` for those it lacks).
    """
    from c_ept import runlog      # imported here: most test files never need it
    if runlog.psutil is None:
        return
    rows = [{'pid': pid, 'name': name, 'cmdline': cmd} for pid, name, cmd in table]

    def process_iter(attrs=None, ad_value=None):
        names = list(attrs) if attrs else ['pid', 'name', 'cmdline']
        return iter([types.SimpleNamespace(info={a: r.get(a, ad_value) for a in names}) for r in rows])
    monkeypatch.setattr(runlog.psutil, 'process_iter', process_iter)


@pytest.fixture(autouse=True)
def _empty_process_list(monkeypatch):
    """No test depends on what runs on the machine: the E6 check sees an empty process list. (A real scan costs about 0.4 s and is made
    several times per S1 run; a running Strata server would stop every run_s1 test.) A test that needs the real list calls
    ``monkeypatch.undo()`` first; one that needs a particular list uses ``fake_processes``; a module- or session-scoped fixture, which is
    set up before this one, builds inside ``with empty_process_list():`` (below).
    """
    _install_process_table(monkeypatch, [])


@contextlib.contextmanager
def empty_process_list():
    """The empty process list, for the length of a ``with`` block: for a fixture that is wider than a test (module or session scope).

    The autouse fixture above works per test, and a module- or session-scoped fixture is set up before it, so such a fixture sees the
    machine's real processes. The E6 check (``run_s1.require_no_strata``, reached by ``build_store``, ``run_s1`` and ``golden``) then
    refuses while a model server runs, and the fixture errors at setup (Ruling 50: the author keeps Strata on while coding). A wide fixture
    that builds a store does the building inside ``with empty_process_list():``. The patch is undone at the end of the block, so the tests
    that need the real list (``monkeypatch.undo()``) or a particular one (``fake_processes``) are not affected.
    """
    with pytest.MonkeyPatch.context() as patch:
        _install_process_table(patch, [])
        yield


@pytest.fixture
def fake_processes(monkeypatch):
    """``fake_processes([(pid, name, command line), ...])`` replaces the process list of ``psutil.process_iter`` for the rest of a test."""
    def install(table):
        _install_process_table(monkeypatch, table)
    return install


@pytest.fixture(scope='session')
def raw_fixture():
    """``(key, patch)`` of the first FIT match of patch 15.18 (cache keys in sorted order): the arguments of ``read_match``."""
    from c_ept import cohort      # imported here so that a broken cohort module does not stop the collection of other test files
    patch = '15.18'
    key = next(k for k in cohort.cache_keys(patch) if cohort.role_of(k, patch)[1] == 'FIT')
    return key, patch


def golden_folder(run) -> Path:
    """The golden folder of the S1 run ``run`` (the run folder, or its ``s1`` store): ``C_EPT_GOLDEN_DIR`` when it is set, else
    ``<run folder>/golden``. The one place that lookup is written in the tests: the ``golden_records`` fixture, the ``golden_dir`` fixture
    of ``test_golden_s1.py`` and every test that hands the lock to a measurement script (``--golden-dir``) all call it, so that they read
    the same lock."""
    path = Path(run)
    run_dir = path.parent if (path / 'STORE.json').is_file() else path
    return Path(os.environ.get('C_EPT_GOLDEN_DIR') or run_dir / 'golden')


@pytest.fixture(scope='session')
def golden_records():
    """The ``MatchRecord`` of the 20 matches of the S1 golden lock, read from the raw files with ``read_match`` (no holdout patch).

    Skipped when ``C_EPT_S1_RUN`` is not set. The run folder (or its ``s1`` store) holds ``golden/s1/matches/key.npy`` and ``patch.npy``
    (``C_EPT_GOLDEN_DIR`` overrides the golden folder: ``golden_folder``); both are read through numpy here, so no key is written
    in the repository. The records hold the match keys: a test must neither print nor compare them in a way that shows them in a failure
    message (``MatchRecord.__repr__`` hides the key). If the lock is missing the fixture fails: a golden test never passes without it.
    """
    run = os.environ.get('C_EPT_S1_RUN')
    if not run:
        pytest.skip('C_EPT_S1_RUN is not set: the 20 golden matches are named by the golden lock of an S1 run')
    import numpy as np
    from c_ept import config as C
    from c_ept.data import reader
    matches = golden_folder(run) / 's1' / 'matches'
    if not ((matches / 'key.npy').is_file() and (matches / 'patch.npy').is_file()):
        pytest.fail('the golden lock (golden/s1/matches/key.npy, patch.npy) is missing in the S1 run folder', pytrace=False)
    keys = [k.decode('ascii') for k in np.load(matches / 'key.npy').tolist()]
    patches = [C.PATCHES[int(i)] for i in np.load(matches / 'patch.npy').tolist()]
    if len(keys) != 20 or len(patches) != 20:
        pytest.fail(f'the golden lock names {len(keys)} matches, not 20', pytrace=False)
    return [reader.read_match(key, patch) for key, patch in zip(keys, patches)]


@pytest.fixture(scope='session')
def open_table():
    """The all-pairs geodesic table of the open 75 x 75 grid, built once per session (about 4 s and a 63 MB result) and shared by the
    terrain tests and the view tests (R12-n, R13-j)."""
    import numpy as np
    from c_ept.preprocessing import terrain
    return terrain.geodesic_table(np.ones((75, 75), dtype=bool))


@pytest.fixture(scope='session')
def open_dir(open_table, tmp_path_factory):
    """The folder of the open grid's terrain, written once per session. Each test opens its own ``Terrain`` (``open_terrain``, 8 ms), so
    that no memory map is open when the folder is removed at the end: Windows does not delete a mapped file."""
    import gc
    import shutil

    import numpy as np
    from c_ept.preprocessing import terrain
    out = tmp_path_factory.mktemp('open_terrain')
    zero = np.zeros((150, 150))
    terrain.write_terrain_files(out, np.ones((75, 75), dtype=bool), zero, zero, open_table)
    yield out
    gc.collect()
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def open_terrain(open_dir):
    """A fresh ``Terrain`` of the open grid (every cell walkable); the test owns it and it is dropped afterwards."""
    from c_ept.preprocessing import terrain
    t = terrain.Terrain(open_dir)
    yield t
    del t


def load_proto():
    """The prototype module ``PROTO/link_units.py``, or None when it cannot be loaded (another machine, other packages, ...).

    Loaded with bytecode writing off, so that no ``__pycache__`` appears in the read-only prototype folders. The prototype puts a
    second prototype folder on ``sys.path`` and imports ``g2_window_check`` from it: both are undone afterwards.
    """
    flag, path = sys.dont_write_bytecode, list(sys.path)
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('proto_link_units', PROTO / 'link_units.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:           # the comparison is a check on top of the brute force: say so, do not stop the collection
        warnings.warn(f'prototype {PROTO} not loaded, comparison with it skipped: {exc!r}')
        return None
    finally:
        sys.dont_write_bytecode = flag
        sys.path[:] = path
        sys.modules.pop('g2_window_check', None)


@pytest.fixture(scope='session')
def proto():
    """The prototype module, loaded when a test first asks for it (collecting does not import it); None if it cannot be.

    The unit tests compare with it only when it is there. The golden tests of the S1 run (test_golden_s1.py) fail explicitly when it is
    not, so that a missing prototype can never make them pass.
    """
    return load_proto()
