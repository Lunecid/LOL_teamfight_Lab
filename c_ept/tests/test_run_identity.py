"""Tests of the import-file hash identity of the run kernel (``c_ept.run.identity``, S4b-prep K2, decisions D11 and A5-F5).

The earlier stacks bind a run to the *whole* working tree: ``run_s4a.repository_state`` (``git status --porcelain --untracked-files=all``,
30 s), ``s4a_operator.code_check`` (``--untracked-files=normal``, no timeout), ``runlog.code_state`` (``-- c_ept``, 60 s) and
``comparison.protocol.clean_source``/``verify_source`` (``--untracked-files=all``, 60 s, plus a reviewed file manifest). A documentation edit
or one slow Git call ended a multi-hour run (review A5-F5). The identity of this module is the HEAD commit plus the SHA-256 of every
``c_ept`` module file that was actually imported; Git is asked once at the start, and the per-epoch check is the hashes only.

These tests use a temporary Git repository with a tiny fake package (``c_ept_k2fake``) and never the real repository. Git itself is real; the
timeouts are real too (a child process that sleeps stands in for a hung Git). Protected modules are stand-in files under the temporary
folder. The tests pin: documentation and unimported-file changes do not matter; an imported-file change does, in the working tree, in
the index, by deletion and by a commit; a slow or missing Git pauses instead of failing; a protected module is named, never opened or hashed;
and the record round trip through ``c_ept.run.io``.

해석: 정체성은 "어느 커밋에서, 어떤 파일을 실제로 불러와 돌았는가"이다. 문서나 하네스 파일을 고쳐도 실행은 같은 코드이므로 정체성이 깨지지
않아야 하고, 불러온 모듈 하나라도 바뀌면 깨져야 한다. Git이 느리거나 시작되지 않으면 정체성이 바뀐 것이 아니므로 실패가 아니라 멈춤(다시 할
수 있는 상태)이다. 분할 정책 모듈은 이름만 남기고 열지도 해시하지도 않는다.
"""
import ast
import builtins
import dataclasses
import hashlib
import importlib
import inspect
import io as stdlib_io
import os
import subprocess
import sys
import sysconfig
import time
import types
from pathlib import Path

import pytest

from c_ept.run import identity as run_identity
from c_ept.run import io as run_io
from c_ept.run.identity import Identity, IdentityChanged, IdentityUnconfirmed, Paused, capture, start_gate, verify

FAKE = 'c_ept_k2fake'                       # starts with "c_ept", so the capture rule applies; the real c_ept package is not touched
ENTRY = [f'{FAKE}.main']
MARK = 'LEAKED_CONTENT_MARKER'              # text that must never appear in a message
HELPER = f'{FAKE}/helper.py'
SLEEPER = (sys.executable, '-c', 'import time; time.sleep(60)')          # a Git that never answers
TOOLS = 'k2tools/k2_runner.py'               # a repository module whose name does not start with "c_ept" (like tools.ept_training_job); no __init__.py
FILES = {
    f'{FAKE}/__init__.py': 'ROOT = True\n',
    f'{FAKE}/main.py': f'from {FAKE} import helper\nfrom k2tools import k2_runner\nVALUE = helper.VALUE + k2_runner.VALUE\n',
    f'{FAKE}/helper.py': 'VALUE = 1\n',
    f'{FAKE}/unused.py': 'VALUE = 99\n',        # in the same package, never imported by the entry module
    TOOLS: 'VALUE = 10\n',
    'docs/NOTE.md': 'a note about the run\n',
}
IMPORTED = {f'{FAKE}/__init__.py', f'{FAKE}/main.py', HELPER, TOOLS}


def git(repo, *args):
    # core.longpaths: the temporary folder of a test run is long (about 225 characters), and Git for Windows stops at 260 without it
    done = subprocess.run(['git', '-C', str(repo), '-c', 'core.longpaths=true', '-c', 'user.name=K2 Test', '-c', 'user.email=k2@example.invalid',
                           '-c', 'commit.gpgsign=false', *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def put(repo, rel, text):
    path = Path(repo) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))
    return path


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A committed temporary repository with the fake package; its folder is on ``sys.path`` and the fake modules are removed afterwards."""
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q', '--template=')                      # no sample hooks: their long names alone pass the Windows limit in a deep temporary folder
    git(root, 'config', 'core.autocrlf', 'false')               # the global setting of this host is true; byte-exact files are wanted
    git(root, 'config', 'core.longpaths', 'true')               # also for the Git calls of the code under test
    for rel, text in FILES.items():
        put(root, rel, text)
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'base')
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    for loaded in [n for n in sys.modules if run_identity._is_protected_module(n)]:
        monkeypatch.delitem(sys.modules, loaded)                # the identity names protected modules that are loaded in the process: start from none
    importlib.invalidate_caches()
    yield root
    for name in [n for n in sys.modules if n.startswith(FAKE) or n.startswith('k2tools')]:
        sys.modules.pop(name, None)


def the_identity(repo):
    return capture(ENTRY, repo=repo)


# ---- the interface and the module layout ---------------------------------------------------------------------------------------------
def test_the_interface_has_the_planned_names_and_signatures():
    assert [f.name for f in dataclasses.fields(Identity)] == ['commit', 'files', 'protected']
    assert Identity.__dataclass_params__.frozen
    cap = inspect.signature(capture)
    assert list(cap.parameters) == ['modules', 'repo', 'timeout_s']
    assert cap.parameters['repo'].kind is inspect.Parameter.KEYWORD_ONLY and cap.parameters['repo'].default is None
    assert cap.parameters['timeout_s'].kind is inspect.Parameter.KEYWORD_ONLY and cap.parameters['timeout_s'].default == 30
    ver = inspect.signature(verify)
    assert list(ver.parameters) == ['saved', 'repo'] and ver.parameters['repo'].kind is inspect.Parameter.KEYWORD_ONLY
    gate = inspect.signature(start_gate)
    assert list(gate.parameters) == ['repo', 'identity', 'timeout_s']
    assert gate.parameters['timeout_s'].kind is inspect.Parameter.KEYWORD_ONLY and gate.parameters['timeout_s'].default == 30
    assert issubclass(IdentityChanged, Exception) and issubclass(Paused, Exception)
    assert not issubclass(Paused, IdentityChanged) and not issubclass(IdentityChanged, Paused)    # K5 maps them to different exit paths
    assert issubclass(IdentityUnconfirmed, IdentityChanged) and not issubclass(IdentityUnconfirmed, Paused)    # a refusal that K5 can tell apart
    assert IdentityChanged.proven is True and IdentityUnconfirmed.proven is False
    assert callable(Identity.to_record) and callable(Identity.from_record)


def test_the_module_imports_only_the_standard_library_and_run_io():
    """So that the identity can be captured before any model code is loaded, and the capture does not depend on what the module itself pulls in."""
    tree = ast.parse(Path(run_identity.__file__).read_text(encoding='utf-8'))
    external = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            external += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'a relative import'
            if node.module == 'c_ept.run':
                assert [a.name for a in node.names] == ['io']
            else:
                external.append(node.module)
    assert {name.split('.')[0] for name in external} <= set(sys.stdlib_module_names), external


def test_the_protected_lists_agree_with_the_comparison_protocol():
    """The protected paths are the ones of ``comparison.protocol.PROTECTED``; the module names follow from them (read as text, nothing imported)."""
    source = Path(run_identity.__file__).resolve().parents[1] / 'comparison' / 'protocol.py'
    expected = None
    for node in ast.parse(source.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROTECTED' for t in node.targets):
            expected = ast.literal_eval(node.value)
    assert expected is not None
    assert run_identity.PROTECTED_PATHS == expected
    assert run_identity.PROTECTED_MODULES == ('c_ept.cohort', 'research_pipeline.cohort_policy', 'research_pipeline.holdout_r3')


def test_the_default_repository_is_the_one_that_holds_the_c_ept_package():
    """Only the derivation of the default is checked: no Git is run against the real repository in a test."""
    root = run_identity._default_repo()
    assert (root / 'c_ept' / 'run' / 'identity.py').samefile(run_identity.__file__)


# ---- capture ---------------------------------------------------------------------------------------------------------------------------
def test_capture_names_the_commit_and_hashes_the_imported_files_only(repo):
    assert FAKE + '.main' not in sys.modules                       # the entry module is imported by capture itself
    identity = capture(ENTRY, repo=repo)
    assert FAKE + '.main' in sys.modules
    assert identity.commit == git(repo, 'rev-parse', 'HEAD')
    assert dict(identity.files) == {rel: sha(FILES[rel]) for rel in IMPORTED}    # repository-relative POSIX paths -> SHA-256
    assert list(identity.files) == sorted(identity.files)
    assert f'{FAKE}/unused.py' not in identity.files and 'docs/NOTE.md' not in identity.files
    assert identity.protected == ()
    assert TOOLS in identity.files                                              # a repository module is covered whatever its name (I3)
    assert 'c_ept/run/identity.py' not in identity.files                        # the real c_ept modules lie outside the temporary repository


def test_capture_refuses_a_bare_string_a_protected_entry_and_an_empty_identity(repo, monkeypatch):
    with pytest.raises(TypeError):
        capture(FAKE + '.main', repo=repo)                         # a bare string would be iterated letter by letter
    monkeypatch.delitem(sys.modules, 'c_ept.cohort', raising=False)
    with pytest.raises(ValueError, match='protected'):
        capture(['c_ept.cohort'], repo=repo)
    assert 'c_ept.cohort' not in sys.modules                      # refused before the import: its body is never executed
    with pytest.raises(ValueError, match='protected'):
        capture(['holdout_r3.split'], repo=repo)                   # a name with the part holdout_r3 is protected wherever it comes from
    with pytest.raises(ValueError, match='no imported'):
        capture(['json'], repo=repo)                               # nothing of the repository was imported: a hollow identity is refused
    with pytest.raises(ValueError, match='positive'):
        capture(ENTRY, repo=repo, timeout_s=0)


def test_capture_lets_an_import_error_through(repo):
    with pytest.raises(ModuleNotFoundError):
        capture([FAKE + '.missing'], repo=repo)


# ---- the identity does not move for documentation, and moves for an imported module ---------------------------------------------------
def test_doc_change_does_not_break_identity(repo):
    identity = the_identity(repo)
    put(repo, 'docs/NOTE.md', f'a changed note {MARK}\n')                       # tracked, not imported
    put(repo, f'{FAKE}/unused.py', f'VALUE = 100  # {MARK}\n')                  # tracked, same package, not imported
    put(repo, 'docs/NEW.md', 'an untracked document\n')                         # untracked
    put(repo, f'{FAKE}/untracked_new.py', 'VALUE = 5\n')                        # untracked module of the package
    put(repo, 'harness/task.md', 'a harness note\n')
    assert git(repo, 'status', '--porcelain', '--untracked-files=all')          # the whole-tree check of the old stacks would refuse here
    verify(identity, repo=repo)
    start_gate(repo, identity)
    verify(identity, repo=repo)                                                 # and again: the gate does not change anything


@pytest.mark.parametrize('how', ['edit', 'staged', 'delete', 'committed'])
def test_imported_module_change_breaks_identity(repo, how):
    identity = the_identity(repo)
    helper = Path(repo) / HELPER
    if how == 'delete':
        helper.unlink()
    else:
        helper.write_bytes(f'VALUE = 2  # {MARK}\n'.encode('utf-8'))
        if how in ('staged', 'committed'):
            git(repo, 'add', HELPER)
        if how == 'committed':
            git(repo, 'commit', '-q', '-m', 'change the helper')
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)                                             # the hash differs, or the file is missing
    assert changed.value.paths == (HELPER,)
    assert HELPER in str(changed.value)
    assert MARK not in str(changed.value) and 'main.py' not in str(changed.value) and '__init__' not in str(changed.value)
    with pytest.raises(IdentityChanged) as gate:
        start_gate(repo, identity)                                              # a new HEAD, or a tracked modification of an imported file
    assert MARK not in str(gate.value)
    assert changed.value.proven and gate.value.proven                           # seen to differ: not the "cannot confirm" refusal
    assert not isinstance(changed.value, IdentityUnconfirmed) and not isinstance(gate.value, IdentityUnconfirmed)
    if how == 'committed':
        assert 'commit' in str(gate.value) and gate.value.paths == ()
    else:
        assert gate.value.paths == (HELPER,) and HELPER in str(gate.value)


def test_the_gate_refuses_a_new_head_even_when_only_documentation_changed(repo):
    identity = the_identity(repo)
    put(repo, 'docs/NOTE.md', 'changed\n')
    git(repo, 'add', 'docs/NOTE.md')
    git(repo, 'commit', '-q', '-m', 'documentation only')
    verify(identity, repo=repo)                                                 # the imported files are the same bytes ...
    with pytest.raises(IdentityChanged, match='commit'):
        start_gate(repo, identity)                                              # ... but the commit that names the code is another


def test_verify_names_every_changed_path_and_reads_only_the_identity_files(repo):
    identity = the_identity(repo)
    put(repo, f'{FAKE}/main.py', f'VALUE = 0  # {MARK}\n')
    (Path(repo) / HELPER).unlink()
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)
    assert changed.value.paths == (HELPER, f'{FAKE}/main.py')


def test_a_path_that_became_a_directory_counts_as_missing(repo):
    identity = the_identity(repo)
    (Path(repo) / HELPER).unlink()
    (Path(repo) / HELPER).mkdir()                                               # on Windows opening a directory raises PermissionError
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)
    assert changed.value.paths == (HELPER,)


def test_an_unreadable_file_pauses_and_a_changed_file_wins(repo, monkeypatch):
    identity = the_identity(repo)
    real = run_io.sha_file
    def flaky(path, **kwargs):
        if Path(path).name == 'main.py':
            raise PermissionError(13, 'locked by another process')
        return real(path, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_RETRY_PAUSE_S', 0)                            # the retries are tested below; here only the classification
        m.setattr(run_identity.run_io, 'sha_file', flaky)
        with pytest.raises(Paused, match='main.py') as paused:                  # present but unreadable: not evidence of a change
            verify(identity, repo=repo)
        assert not isinstance(paused.value, IdentityChanged)
        put(repo, HELPER, 'VALUE = 3\n')
        with pytest.raises(IdentityChanged) as changed:                         # a proven change is reported even when another file is unreadable
            verify(identity, repo=repo)
        assert changed.value.paths == (HELPER,)


def test_a_long_list_of_changed_paths_is_cut_in_the_message_but_not_in_the_attribute():
    paths = [f'{FAKE}/m{i:03d}.py' for i in range(50)]
    error = IdentityChanged('imported code changed', paths)
    assert error.paths == tuple(sorted(paths))
    assert paths[0] in str(error) and paths[19] in str(error) and paths[20] not in str(error) and '30 more' in str(error)


# ---- git is asked at the start only; a slow or missing git pauses ----------------------------------------------------------------------
def test_start_gate_runs_git_a_fixed_number_of_times_and_verify_never(repo, monkeypatch):
    identity = the_identity(repo)
    seen = []
    real = subprocess.run
    def spy(command, *args, **kwargs):
        seen.append((list(command), kwargs.get('timeout'), kwargs.get('env', {}).get('GIT_OPTIONAL_LOCKS')))
        return real(command, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(run_identity.subprocess, 'run', spy)
        verify(identity, repo=repo)
        assert seen == []                                                       # the per-epoch check is hashes only (A5-F5)
        start_gate(repo, identity)
    assert len(seen) == 2                                                       # one rev-parse, one status
    (first, t1, lock1), (second, t2, lock2) = seen
    assert 'rev-parse' in first and 'status' in second
    assert lock1 == lock2 == '0'                                                # a read-only check must not rewrite the index of the checkout
    assert t2 < t1 <= 30                                                        # one deadline for both calls: the second gets what the first left
    assert '--literal-pathspecs' in second and '--porcelain' in second
    assert '--untracked-files=all' in second and '--ignored=matching' in second  # an imported file that is untracked or ignored is seen (I2)
    assert second[second.index('--') + 1:] == list(identity.files)             # exactly the imported files, nothing else


def test_many_files_are_checked_in_chunks_so_the_command_line_stays_short(repo, monkeypatch):
    names = [f'{FAKE}/bulk/m{i:02d}.py' for i in range(20)]
    for name in names:
        put(repo, name, f'VALUE = "{name}"\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'bulk')
    identity = Identity(git(repo, 'rev-parse', 'HEAD'), {name: sha(f'VALUE = "{name}"\n') for name in names})
    calls = []
    real = subprocess.run
    def spy(command, *args, **kwargs):
        calls.append(list(command))
        return real(command, *args, **kwargs)
    monkeypatch.setattr(run_identity, '_STATUS_CHUNK', 7)
    with monkeypatch.context() as m:
        m.setattr(run_identity.subprocess, 'run', spy)
        start_gate(repo, identity)
        assert len(calls) == 1 + 3                                              # rev-parse + ceil(20 / 7) statuses
        calls.clear()
        put(repo, names[13], 'VALUE = 0\n')                                     # in the middle chunk
        with pytest.raises(IdentityChanged) as changed:
            start_gate(repo, identity)
    assert changed.value.paths == (names[13],)


def test_an_identity_without_files_checks_the_commit_only(repo):
    identity = Identity(git(repo, 'rev-parse', 'HEAD'), {})
    put(repo, 'docs/NOTE.md', 'dirty\n')                                        # with no pathspec Git would list the whole tree: it must not be asked
    start_gate(repo, identity)


def test_git_timeout_pauses_instead_of_failing(repo, monkeypatch):
    identity = the_identity(repo)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_GIT', SLEEPER)
        started = time.monotonic()
        with pytest.raises(Paused) as paused:
            start_gate(repo, identity, timeout_s=0.5)
        assert time.monotonic() - started < 30                                  # the hung child was stopped, not waited for
        assert not isinstance(paused.value, IdentityChanged) and '0.5' in str(paused.value)
        with pytest.raises(Paused):
            capture(ENTRY, repo=repo, timeout_s=0.5)                            # the same at the start of a run
    start_gate(repo, identity)                                                  # Paused leaves nothing behind: the run can be started again


def test_git_that_cannot_start_pauses_and_git_that_cannot_answer_refuses(repo, monkeypatch):
    identity = the_identity(repo)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_GIT', ('k2-no-such-git-executable-4821',))
        with pytest.raises(Paused, match='started'):
            start_gate(repo, identity)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_GIT', (sys.executable, '-c', 'import sys; sys.exit(3)'))
        with pytest.raises(IdentityChanged, match='exit 3') as refused:         # Git answered "no": the identity cannot be confirmed
            start_gate(repo, identity)
        assert not isinstance(refused.value, Paused)
        assert isinstance(refused.value, IdentityUnconfirmed) and refused.value.proven is False      # distinguishable from a proven change (M1)


def test_the_time_budget_is_one_deadline_for_all_git_calls(repo, monkeypatch):
    with monkeypatch.context() as m:
        reached = []
        m.setattr(run_identity.subprocess, 'run', lambda *a, **k: reached.append(1))
        with pytest.raises(Paused):
            run_identity._run_git(Path(repo), ('status',), time.monotonic() - 1.0, 30)      # the deadline has passed: Git is not started
        assert reached == []


def test_a_folder_that_is_not_the_repository_root_is_refused(repo):
    identity = the_identity(repo)
    with pytest.raises(IdentityUnconfirmed, match='root'):
        start_gate(Path(repo) / FAKE, identity)                                  # a pathspec relative to a subfolder would match nothing: a false "clean"
    start_gate(str(repo), identity)                                              # a str works as well as a Path


def test_git_that_answers_in_an_unexpected_form_is_refused(repo, monkeypatch):
    identity = the_identity(repo)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_GIT', (sys.executable, '-c', 'print("not a path"); print("not a commit")'))
        with pytest.raises(IdentityUnconfirmed, match='unexpected form') as refused:
            start_gate(repo, identity)
        assert refused.value.proven is False


# ---- protected modules: named, never opened, never hashed ------------------------------------------------------------------------------
def test_protected_modules_are_named_not_hashed(repo, monkeypatch):
    body = 'SPLIT = "a stand-in for a protected body, never to be read"\n'
    files = {
        'c_ept/cohort.py': put(repo, 'c_ept/cohort.py', body),
        'research_pipeline/cohort_policy.py': put(repo, 'research_pipeline/cohort_policy.py', body),
        'research_pipeline/holdout_r3/rules.py': put(repo, 'research_pipeline/holdout_r3/rules.py', body),
        'anywhere/holdout_r3/deep.py': put(repo, 'anywhere/holdout_r3/deep.py', body),         # any folder named holdout_r3 is protected
    }
    stubs = {                                                                   # name -> where the module says it was loaded from
        'c_ept.cohort': files['c_ept/cohort.py'],
        'research_pipeline.cohort_policy': files['research_pipeline/cohort_policy.py'],
        'research_pipeline.holdout_r3.rules': files['research_pipeline/holdout_r3/rules.py'],
        f'{FAKE}.aliased': files['c_ept/cohort.py'],                            # a harmless name for a protected file: caught by its path
        f'{FAKE}.ghost': 'Z:/nowhere/c_ept/cohort.py',                          # a path that does not exist: caught by its text, no file system call
        'split_k2': 'Z:/elsewhere/holdout_r3/split.py',                         # the external split module: outside the repository, in a holdout_r3 folder
        f'{FAKE}.deep': files['anywhere/holdout_r3/deep.py'],                   # inside the repository, outside research_pipeline/: caught by the part
        'holdout_r3.shadow': 'Z:/nowhere/shadow.py',                            # caught by its dotted name
    }
    opened, hashed = watch_files(monkeypatch)
    for name, file in stubs.items():
        stub = types.ModuleType(name)
        stub.__file__ = str(file)
        monkeypatch.setitem(sys.modules, name, stub)
    identity = capture(ENTRY, repo=repo)
    verify(identity, repo=repo)
    start_gate(repo, identity)
    assert identity.protected == tuple(sorted(stubs))                           # every protected module that was loaded is named ...
    assert set(identity.files) == IMPORTED                                      # ... and none is among the hashed files
    assert sha(body) not in identity.files.values()
    protected_files = {real_path(p) for p in files.values()}
    assert {real_path(p) for p in opened + hashed if isinstance(p, (str, os.PathLike))}.isdisjoint(protected_files)    # never opened, never hashed
    assert sorted(Path(p).name for p in hashed) == sorted([Path(rel).name for rel in IMPORTED] * 2)   # once by capture, once by verify


def link_directory(link, target):
    """A directory link without special rights: a symbolic link, or a junction on Windows. ``pytest.skip`` when neither can be made."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        done = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True, text=True) if os.name == 'nt' else None
        if done is None or done.returncode != 0:
            pytest.skip('no directory link can be made on this host')


def test_a_protected_file_behind_a_link_is_named_not_hashed(repo, monkeypatch):
    """The text of the path says nothing (``.../linked/cohort.py``); only the resolved path shows that it is ``c_ept/cohort.py``."""
    body = 'SPLIT = "a stand-in for a protected body, never to be read"\n'
    protected_file = put(repo, 'c_ept/cohort.py', body)
    link_directory(Path(repo) / 'linked', Path(repo) / 'c_ept')
    opened, hashed = watch_files(monkeypatch)
    stub = types.ModuleType(f'{FAKE}.linked')
    stub.__file__ = str(Path(repo) / 'linked' / 'cohort.py')
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    identity = capture(ENTRY, repo=repo)
    assert identity.protected == (f'{FAKE}.linked',)
    assert set(identity.files) == IMPORTED
    assert real_path(protected_file) not in {real_path(p) for p in opened + hashed if isinstance(p, (str, os.PathLike))}


def real_path(path):
    return os.path.normcase(os.path.realpath(os.fspath(path)))                  # a link resolves to its target: an alias counts as the protected file


def watch_files(monkeypatch):
    """Record every file that Python opens (``open``) and every path that ``run.io.sha_file`` is asked for; the lists are filled as the test runs."""
    opened, hashed = [], []
    real_open, real_sha = builtins.open, run_io.sha_file
    def spy_open(file, *args, **kwargs):
        opened.append(file)
        return real_open(file, *args, **kwargs)
    def spy_sha(path, **kwargs):
        hashed.append(path)
        return real_sha(path, **kwargs)
    monkeypatch.setattr(builtins, 'open', spy_open)
    monkeypatch.setattr(stdlib_io, 'open', spy_open)
    monkeypatch.setattr(run_identity.run_io, 'sha_file', spy_sha)
    return opened, hashed


# ---- the record ------------------------------------------------------------------------------------------------------------------------
def test_record_round_trip(repo, tmp_path):
    identity = the_identity(repo)
    with_protected = Identity(identity.commit, identity.files, ('c_ept.cohort', 'research_pipeline.cohort_policy'))
    for original in (identity, with_protected):
        record = original.to_record()
        assert set(record) == {'schema', 'commit', 'files', 'protected'} and record['schema'] == run_identity.SCHEMA
        assert type(record['files']) is dict and type(record['protected']) is list     # plain JSON types
        path = tmp_path / 'run' / 'IDENTITY.json'
        digest = run_io.write_json_atomic(path, record)
        back = Identity.from_record(run_io.read_json(path))
        assert back == original and hash(back) == hash(original)
        assert back.to_record() == record
        assert run_io.write_json_atomic(tmp_path / 'again.json', back.to_record()) == digest      # the same bytes
        assert (tmp_path / 'again.json').read_bytes() == path.read_bytes()
        verify(back, repo=repo)                                                  # a reloaded identity verifies like the captured one
        start_gate(repo, back)


def test_the_identity_is_immutable_and_owns_its_data():
    source = {HELPER: sha('x')}
    identity = Identity('a' * 40, source, ['b.c', 'a.b'])
    source['c_ept_k2fake/other.py'] = sha('y')                                  # the caller's dict is a copy for the identity
    assert set(identity.files) == {HELPER} and identity.protected == ('a.b', 'b.c')
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.commit = 'b' * 40
    with pytest.raises(TypeError):
        identity.files[HELPER] = sha('z')
    with pytest.raises(TypeError):
        del identity.files[HELPER]


GOOD = {'schema': run_identity.SCHEMA, 'commit': 'a' * 40, 'files': {HELPER: 'b' * 64}, 'protected': []}


def variant(**changes):
    record = dict(GOOD, files=dict(GOOD['files']))
    record.update(changes)
    return record


@pytest.mark.parametrize('record', [
    variant(extra=1),                                                           # unknown field
    {k: v for k, v in GOOD.items() if k != 'protected'},                        # missing field
    variant(schema='other-v9'),
    variant(commit='a' * 39), variant(commit='A' * 40), variant(commit='g' * 40), variant(commit=None),
    variant(files=[HELPER]), variant(files={HELPER: 'b' * 63}), variant(files={HELPER: 'B' * 64}), variant(files={HELPER: 5}),
    variant(files={'../outside.py': 'b' * 64}), variant(files={'/abs/path.py': 'b' * 64}), variant(files={'C:/x/y.py': 'b' * 64}),
    variant(files={'c_ept\\x.py': 'b' * 64}), variant(files={'a//b.py': 'b' * 64}), variant(files={'a/./b.py': 'b' * 64}),
    variant(files={'': 'b' * 64}), variant(files={'.': 'b' * 64}), variant(files={'a/b.py/': 'b' * 64}), variant(files={'a\nb.py': 'b' * 64}),
    variant(files={'c_ept/cohort.py': 'b' * 64}), variant(files={'C_EPT/Cohort.py': 'b' * 64}),                 # protected, also by case
    variant(files={'research_pipeline/cohort_policy.py': 'b' * 64}),
    variant(files={'research_pipeline/holdout_r3/x/y.py': 'b' * 64}),
    variant(files={'anywhere/holdout_r3/x.py': 'b' * 64}), variant(files={'HOLDOUT_R3/x.py': 'b' * 64}),       # any part holdout_r3 (I3)
    variant(files={'c_ept/cohort.py.': 'b' * 64}), variant(files={'c_ept./cohort.py': 'b' * 64}), variant(files={'c_ept/cohort.py ': 'b' * 64}),
    variant(files={'c_ept/cohort.py. ': 'b' * 64}), variant(files={'c_ept /cohort.py': 'b' * 64}),                 # Windows drops a trailing dot or space (I1)
    variant(files={'RESEAR~1/COHORT~1.PY': 'b' * 64}), variant(files={'c_ept/cohort~1.py': 'b' * 64}),            # 8.3 short names (I1)
    variant(files={'c_ept/cohort.py::$DATA': 'b' * 64}),                                                          # an alternate data stream
    variant(protected='c_ept.cohort'), variant(protected=[1]), variant(protected=['']), variant(protected=None),
    ['not', 'an', 'object'], None, 'text',
])
def test_a_bad_record_is_refused(record):
    with pytest.raises(run_io.IORefused) as refused:
        Identity.from_record(record)
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    assert isinstance(refused.value, ValueError)                              # K5 maps refusals of input to the gate exit


def test_a_refusal_does_not_echo_a_key_or_a_value():
    record = variant(files={'LEAK_KEY_MARKER/../x.py': 'LEAK_VALUE_MARKER'})
    with pytest.raises(run_io.IORefused) as refused:
        Identity.from_record(record)
    assert 'LEAK' not in str(refused.value)
    with pytest.raises(run_io.IORefused) as direct:
        Identity('LEAK_COMMIT_MARKER', {})
    assert 'LEAK' not in str(direct.value)


def test_a_hand_built_identity_cannot_hold_a_protected_file():
    """``verify`` opens every identity file, so the constructor is where a protected path is stopped, for a loaded record and a built object alike."""
    with pytest.raises(run_io.IORefused):
        Identity('a' * 40, {'c_ept/cohort.py': 'b' * 64})
    with pytest.raises(run_io.IORefused):
        Identity('a' * 40, {'research_pipeline/holdout_r3/rules.py': 'b' * 64})
    Identity('a' * 40, {'c_ept/cohorts.py': 'b' * 64, 'c_ept/cohort.pyc': 'b' * 64})       # similar names are not protected


# ---- fix round (review of 82054dd): aliases, links, scope, untracked files, retries, the Git environment --------------------------------
@pytest.mark.skipif(os.name != 'nt', reason='Windows path normalisation')
def test_windows_really_aliases_the_names_that_the_constructor_refuses(tmp_path):
    """The reason for the lexical rule (I1): on Windows these names open the stand-in ``c_ept/cohort.py``."""
    put(tmp_path, 'c_ept/cohort.py', 'STAND_IN = 1\n')
    aliases = ('c_ept/cohort.py.', 'c_ept./cohort.py', 'c_ept/cohort.py ')
    assert all((tmp_path / alias).is_file() for alias in aliases)
    for alias in aliases:
        with pytest.raises(run_io.IORefused):
            Identity('a' * 40, {alias: 'b' * 64})


def test_a_link_swapped_in_after_capture_is_changed_and_never_opened(repo, monkeypatch):
    """A junction put at an identity path after the capture leads to a stand-in protected file, and a decoy with the very bytes of the module is
    next to it: if ``verify`` followed the link, the hash would match and nothing would be reported."""
    identity = the_identity(repo)
    body = 'SPLIT = "a stand-in for a protected body, never to be read"\n'
    protected_file = put(repo, 'c_ept/cohort.py', body)
    decoy = put(repo, 'c_ept/k2_runner.py', FILES[TOOLS])
    os.rename(Path(repo) / 'k2tools', Path(repo) / 'k2tools_old')
    link_directory(Path(repo) / 'k2tools', Path(repo) / 'c_ept')                # k2tools/k2_runner.py now names c_ept/k2_runner.py
    opened, hashed = watch_files(monkeypatch)
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)
    assert changed.value.paths == (TOOLS,) and changed.value.proven
    touched = {real_path(p) for p in opened + hashed if isinstance(p, (str, os.PathLike))}
    assert real_path(protected_file) not in touched and real_path(decoy) not in touched      # the target was never opened
    assert sorted(Path(p).name for p in hashed) == sorted(Path(rel).name for rel in IMPORTED - {TOOLS})


def test_a_link_that_names_a_protected_file_is_changed_and_never_opened(repo, monkeypatch):
    """A hand-built identity with a harmless-looking path through a junction; the stored hash is that of the stand-in, so opening it would pass."""
    body = 'SPLIT = "a stand-in for a protected body, never to be read"\n'
    protected_file = put(repo, 'c_ept/cohort.py', body)
    link_directory(Path(repo) / 'linkdir', Path(repo) / 'c_ept')
    link_directory(Path(repo) / 'linkdocs', Path(repo) / 'docs')
    identity = Identity(git(repo, 'rev-parse', 'HEAD'), {'linkdir/cohort.py': sha(body), 'linkdocs/NOTE.md': sha(FILES['docs/NOTE.md'])})
    opened, hashed = watch_files(monkeypatch)
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)
    assert changed.value.paths == ('linkdir/cohort.py', 'linkdocs/NOTE.md')    # also the link to an ordinary file: it is another path, not the named one
    assert hashed == [] and real_path(protected_file) not in {real_path(p) for p in opened if isinstance(p, (str, os.PathLike))}


def test_a_file_symlink_swapped_in_after_capture_is_changed_and_never_opened(repo, monkeypatch):
    identity = the_identity(repo)
    protected_file = put(repo, 'c_ept/cohort.py', 'SPLIT = "a stand-in"\n')
    (Path(repo) / HELPER).unlink()
    try:
        os.symlink(protected_file, Path(repo) / HELPER)
    except (OSError, NotImplementedError):
        pytest.skip('no file symlink can be made on this host (Windows needs a privilege for it)')
    opened, hashed = watch_files(monkeypatch)
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)
    assert changed.value.paths == (HELPER,)
    assert real_path(protected_file) not in {real_path(p) for p in opened + hashed if isinstance(p, (str, os.PathLike))}


def test_a_changed_repository_module_outside_the_c_ept_name_breaks_identity(repo):
    identity = the_identity(repo)
    assert TOOLS in identity.files                                              # k2tools.k2_runner, a module that does not start with "c_ept"
    put(repo, TOOLS, f'VALUE = 11  # {MARK}\n')
    with pytest.raises(IdentityChanged) as changed:
        verify(identity, repo=repo)
    assert changed.value.paths == (TOOLS,) and MARK not in str(changed.value)
    with pytest.raises(IdentityChanged) as gate:
        start_gate(repo, identity)
    assert gate.value.paths == (TOOLS,)


def test_repository_scripts_are_hashed_and_library_code_is_not(repo, monkeypatch):
    stubs = {                                                                   # name -> file; the files exist so that a hash would be possible
        'k2_script': put(repo, 'scripts/run_k2.py', 'VALUE = 5\n'),                        # a repository script / __main__: hashed
        'k2_site': put(repo, '.venv/Lib/site-packages/pkg/mod.py', 'VALUE = 6\n'),        # a virtual environment inside the repository
        'k2_dist': put(repo, 'vendor/Dist-Packages/other.py', 'VALUE = 7\n'),
        'k2_interpreter': put(repo, 'envs/lib/os_like.py', 'VALUE = 8\n'),                 # under an interpreter root
        'k2_nofile': None,
    }
    for name, file in stubs.items():
        stub = types.ModuleType(name)
        if file is not None:
            stub.__file__ = str(file)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(run_identity, '_interpreter_roots', lambda base: [(Path(repo) / 'envs').resolve()])
    identity = capture(ENTRY, repo=repo)
    assert set(identity.files) == IMPORTED | {'scripts/run_k2.py'}
    assert identity.files['scripts/run_k2.py'] == sha('VALUE = 5\n')
    assert identity.protected == ()


def test_interpreter_roots_leave_out_a_root_that_holds_the_repository(repo, tmp_path, monkeypatch):
    base = Path(repo).resolve()
    assert sysconfig.get_paths()                                                # cached before sys.prefix is changed
    assert Path(sys.prefix).resolve() in run_identity._interpreter_roots(base)  # the real interpreter is apart from the temporary repository
    inside = base / 'envs'
    inside.mkdir()
    for attribute in ('prefix', 'base_prefix', 'exec_prefix', 'base_exec_prefix'):
        monkeypatch.setattr(sys, attribute, str(tmp_path))                      # an ancestor of the repository: leaving it out keeps the repo files
    assert tmp_path.resolve() not in run_identity._interpreter_roots(base)
    for attribute in ('prefix', 'base_prefix', 'exec_prefix', 'base_exec_prefix'):
        monkeypatch.setattr(sys, attribute, str(inside))                        # a root inside the repository (a virtual environment) is excluded
    assert inside.resolve() in run_identity._interpreter_roots(base)


@pytest.mark.parametrize('kind', ['untracked', 'ignored', 'tracked'])
def test_an_imported_file_that_the_commit_does_not_contain_refuses_the_gate(repo, kind):
    """I2: D11 names the start commit and the hash of every imported file. An imported file that is untracked or ignored cannot be reproduced
    from the commit, so the gate refuses it; the same file committed is fine. Documents and unimported files stay irrelevant (see above)."""
    late = f'{FAKE}/late.py'
    if kind == 'ignored':
        put(repo, '.gitignore', 'late.py\n')
        git(repo, 'add', '.gitignore')
        git(repo, 'commit', '-q', '-m', 'ignore')
    put(repo, late, 'VALUE = 7\n')
    if kind == 'tracked':
        git(repo, 'add', late)
        git(repo, 'commit', '-q', '-m', 'late')
    identity = capture([*ENTRY, f'{FAKE}.late'], repo=repo)
    assert late in identity.files
    if kind == 'tracked':
        start_gate(repo, identity)
        verify(identity, repo=repo)
        return
    with pytest.raises(IdentityChanged) as refused:
        start_gate(repo, identity)
    assert refused.value.paths == (late,) and refused.value.proven and not isinstance(refused.value, IdentityUnconfirmed)
    verify(identity, repo=repo)                                                 # the bytes are the captured ones: only the commit cannot reproduce them


def test_git_runs_without_the_variables_that_redirect_it(repo, tmp_path, monkeypatch):
    """M5: inside a hook ``GIT_DIR`` and friends are set; Git would then look at another repository and say "clean" or "same commit"."""
    identity = the_identity(repo)
    decoy = tmp_path / 'decoy'
    decoy.mkdir()
    git(decoy, 'init', '-q', '--template=')
    put(decoy, 'x.txt', 'x\n')
    git(decoy, 'add', '-A')
    git(decoy, 'commit', '-q', '-m', 'decoy')
    redirect = {'GIT_DIR': str(decoy / '.git'), 'GIT_WORK_TREE': str(decoy), 'GIT_INDEX_FILE': str(decoy / '.git' / 'index'),
                'GIT_COMMON_DIR': str(decoy / '.git'), 'GIT_CEILING_DIRECTORIES': str(tmp_path), 'GIT_OBJECT_DIRECTORY': str(decoy / '.git' / 'objects'),
                'GIT_NAMESPACE': 'k2', 'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.worktree', 'GIT_CONFIG_VALUE_0': str(decoy)}
    fooled = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], env={**os.environ, **redirect}, capture_output=True, text=True)
    assert fooled.stdout.strip() != identity.commit                              # control: with these variables a plain Git call looks at the decoy
    for key, value in redirect.items():
        monkeypatch.setenv(key, value)
    seen = []
    real = subprocess.run
    def spy(command, *args, **kwargs):
        seen.append(kwargs['env'])
        return real(command, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(run_identity.subprocess, 'run', spy)
        start_gate(repo, identity)                                               # the decoy HEAD would be another commit: refused
    assert len(seen) == 2
    for env in seen:
        assert not [key for key in env if key.upper() in redirect or key.upper().startswith('GIT_CONFIG_')]
        assert env['GIT_OPTIONAL_LOCKS'] == '0'


def test_git_env_drops_the_redirecting_variables_and_keeps_the_rest(monkeypatch):
    names = ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_CEILING_DIRECTORIES', 'GIT_OBJECT_DIRECTORY',
             'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE', 'GIT_PREFIX', 'GIT_CONFIG', 'GIT_CONFIG_PARAMETERS', 'GIT_CONFIG_COUNT',
             'GIT_CONFIG_GLOBAL', 'GIT_CONFIG_SYSTEM', 'GIT_CONFIG_NOSYSTEM', 'GIT_CONFIG_KEY_3', 'GIT_CONFIG_VALUE_3']
    for name in names:
        monkeypatch.setenv(name, 'x')
    monkeypatch.setenv('K2_KEEP', 'kept')
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS', '1')
    env = run_identity._git_env()
    assert not [key for key in env if key.upper() in names]
    assert env['GIT_OPTIONAL_LOCKS'] == '0' and env['K2_KEEP'] == 'kept' and 'PATH' in {key.upper() for key in env}


def test_an_unreadable_file_is_retried_before_the_run_pauses(repo, monkeypatch):
    """M2: a brief lock must not stop a multi-hour run: 3 tries with a pause between them, then ``Paused``; "missing" is not retried."""
    identity = the_identity(repo)
    real = run_io.sha_file
    attempts = {}
    sleeps = []
    def failing(times, error=PermissionError(13, 'locked by another process')):
        def sha_file(path, **kwargs):
            if Path(path).name == 'main.py':
                attempts['n'] = attempts.get('n', 0) + 1
                if attempts['n'] <= times:
                    raise error
            return real(path, **kwargs)
        return sha_file
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_RETRY_PAUSE_S', 0.25)
        m.setattr(run_identity.time, 'sleep', sleeps.append)
        m.setattr(run_identity.run_io, 'sha_file', failing(2))
        verify(identity, repo=repo)                                              # the third try reads the file
        assert attempts['n'] == 3 and sleeps == [0.25, 0.25]
        attempts.clear(), sleeps.clear()
        m.setattr(run_identity.run_io, 'sha_file', failing(99))
        with pytest.raises(Paused, match='main.py'):
            verify(identity, repo=repo)
        assert attempts['n'] == 3 and sleeps == [0.25, 0.25]                     # exactly 3 tries, 2 pauses
        attempts.clear(), sleeps.clear()
        m.setattr(run_identity.run_io, 'sha_file', failing(99, FileNotFoundError(2, 'gone')))
        with pytest.raises(IdentityChanged) as gone:
            verify(identity, repo=repo)
        assert gone.value.paths == (f'{FAKE}/main.py',) and attempts['n'] == 1 and sleeps == []    # missing is a proven change, no retry


def test_a_stat_error_is_retried_too_and_capture_pauses_instead_of_raising_oserror(repo, monkeypatch):
    identity = the_identity(repo)
    real_is_file = Path.is_file
    calls = {'n': 0}
    def flaky_is_file(self, **kwargs):
        if self.name == 'helper.py':
            calls['n'] += 1
            if calls['n'] <= 2:
                raise PermissionError(13, 'stat refused')                        # Python up to 3.12 lets such an error out of is_file
        return real_is_file(self, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_RETRY_PAUSE_S', 0)
        m.setattr(Path, 'is_file', flaky_is_file)
        verify(identity, repo=repo)
    assert calls['n'] == 3
    real = run_io.sha_file
    def locked(path, **kwargs):
        if Path(path).name == 'helper.py':
            raise PermissionError(13, 'locked by another process')
        return real(path, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(run_identity, '_RETRY_PAUSE_S', 0)
        m.setattr(run_identity.run_io, 'sha_file', locked)
        with pytest.raises(Paused, match='helper.py') as paused:
            capture(ENTRY, repo=repo)                                            # no raw OSError out of the capture
        assert not isinstance(paused.value, IdentityChanged)


def test_verify_alone_does_not_see_a_shadowing_package_but_a_fresh_capture_differs(repo):
    """M3: a new package ``helper/`` shadows ``helper.py`` on the next import. ``verify`` re-hashes the old file and passes, so a resume must
    capture again (same entry modules) and require equality with the saved identity, then run the gate. This pins that obligation."""
    saved = the_identity(repo)
    put(repo, f'{FAKE}/helper/__init__.py', 'VALUE = 1\n')
    verify(saved, repo=repo)                                                    # helper.py is unchanged: passes
    for name in [n for n in sys.modules if n.startswith(FAKE)]:
        del sys.modules[name]                                                   # a new process: nothing is loaded yet
    importlib.invalidate_caches()
    fresh = the_identity(repo)
    assert fresh != saved
    assert f'{FAKE}/helper/__init__.py' in fresh.files and HELPER not in fresh.files
    with pytest.raises(IdentityChanged) as gate:
        start_gate(repo, fresh)                                                 # and the new file is untracked besides
    assert gate.value.paths == (f'{FAKE}/helper/__init__.py',)


def test_gate_and_verify_agree_with_the_hosts_autocrlf_setting(repo):
    """M4: the identity hashes working-tree bytes. With ``core.autocrlf=true`` a CRLF file is clean for Git and has CRLF bytes: no false refusal."""
    git(repo, 'config', 'core.autocrlf', 'true')
    put(repo, HELPER, 'VALUE = 1  # crlf\r\n')                                  # a CRLF file, committed: the blob holds LF, the work tree CRLF
    git(repo, 'add', HELPER)
    git(repo, 'commit', '-q', '-m', 'crlf helper')
    assert git(repo, 'show', f'HEAD:{HELPER}') == 'VALUE = 1  # crlf'          # the committed bytes are LF (git() strips the end)
    assert git(repo, 'status', '--porcelain', '--untracked-files=no') == ''
    identity = the_identity(repo)
    assert identity.files[HELPER] == hashlib.sha256(b'VALUE = 1  # crlf\r\n').hexdigest()      # the bytes that Python runs
    verify(identity, repo=repo)
    start_gate(repo, identity)
