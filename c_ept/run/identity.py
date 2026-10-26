"""Execution identity of the run kernel: the HEAD commit plus the SHA-256 of every imported module file of the repository (S4b-prep K2, decision D11).

Design reference: review EPT-T010 A5-F5 ("execution identity is a live Git working tree, re-checked at every stage and epoch; a tree change
or a slow Git call ends the run") and its recommendation: bind the identity to the commit plus the hashes of the imported module files,
check those hashes per epoch, keep ``git status`` as a start-time gate only, and turn a slow Git call into a paused, resumable state.

What the earlier stacks check, and what this module keeps or changes (those modules are frozen and unchanged):

* ``run_s4a.repository_state``: ``rev-parse HEAD`` (40 hex digits), ``--show-toplevel`` equal to the repository, ``GIT_OPTIONAL_LOCKS=0``,
  30 s, and ``status --porcelain --untracked-files=all`` of the *whole* tree must be empty. Kept: the HEAD form, the top-level check, the
  environment variable, the 30 s default and "a Git that cannot answer is never read as clean". Changed: only the imported files are asked.
* ``s4a_operator.code_check``: the same with ``--untracked-files=normal`` and no timeout. Changed: a timeout exists.
* ``runlog.code_state``: ``status --porcelain -- c_ept``, 60 s, ``None`` (unknown, not clean) when Git cannot answer. Kept: unknown is never clean.
* ``comparison.protocol.clean_source`` / ``verify_source``: HEAD equals the reviewed commit, the tree is clean, every file of a reviewed manifest
  has the reviewed SHA-256, a protected path is refused ("protected source body must not be read"), a relative path must not climb or be absolute.
  Kept: the commit comparison, the hash per file, the protected-path rule (same list, same case-insensitive match), the safe relative path.
  Changed: the file list is not a manifest written by hand but the modules that were really imported.
* ``run_s3.check_resume_code``: a resume must run the code the run began on. Here that is NOT ``verify`` alone (see "Resume" below).

Symbols
-------
    R          the repository root (the folder that holds the ``c_ept`` package, or the ``repo`` argument)
    c          the commit, 40 lowercase hex digits (``git rev-parse HEAD``)
    L          the module files of the identity: repository-relative POSIX path -> file under R
    F          the file hashes, ``F(p) = SHA-256(bytes of R/p)`` for p in L (64 lowercase hex digits)
    P          the protected modules that were imported, by module name (their files are never opened)
    I          the identity ``(c, F, P)``
    m          a module in ``sys.modules``; ``name(m)`` its dotted name and ``file(m)`` its ``__file__``
    T          the time budget of Git in seconds (``timeout_s``, 30 by default); one deadline for all Git calls of one gate
    H          the commit that ``HEAD`` names now
    X          the interpreter roots: ``sys.prefix`` and ``sys.exec_prefix`` (base ones too), the ``sysconfig`` install paths, the site folders

Equations
---------
    L = { resolved file(m) : m in sys.modules,  the file lies under R,  not under an X that is inside R or apart from R,
                             no part of its path below R is "site-packages" or "dist-packages",  m is not protected }
        whatever the module name is: c_ept.*, tools.*, research_pipeline.* (not protected), a repository __main__, a top-level script module
    P = { name(m) : m is protected }
        protected = a module name in PROTECTED_MODULES (or below one, or with a part "holdout_r3"), or a file whose path ends in or runs through a
        path of PROTECTED_PATHS, or has a path part "holdout_r3" (the external split module lives in such a folder outside R)
    capture(entry modules) = (c, F, P)  after importing the entry modules               F(p) = SHA-256(R/p)
    verify(I)    :  changed = { p in L : resolve(R/p) is not R/p (links, aliases),  or  R/p is not a file,  or  SHA-256(R/p) != F(p) }
                    changed != {}  =>  IdentityChanged(changed);  a file that exists but cannot be read after 3 tries in about 1 s  =>  Paused
    start_gate   :  H != c  or  { p in L : p is modified (staged or not), deleted, untracked or ignored } != {}  =>  IdentityChanged
                    Git that exits non-zero, answers in another form, or is asked in a folder that is not a root  =>  IdentityUnconfirmed
                    Git slower than T, or Git that cannot start  =>  Paused
    independence :  a change to a file outside L (documentation, harness, an unimported module, an untracked file) changes no verdict
    cost         :  verify runs no Git at all; start_gate runs Git 1 + ceil(|L| / 100) times, all inside one deadline T
    resume       :  capture(same entry modules) == saved,  then start_gate(R, saved)     (verify alone does not prove that a resume runs the same code)

해석
----
정체성은 "어느 커밋에서 어떤 파일을 실제로 불러와 돌았는가"이다. 이전 방식은 작업 트리 전체의 청결을 정체성으로 삼아, 문서 한 줄을 고치거나
Git 호출 하나가 30 s를 넘기는 것만으로 여러 시간짜리 실행이 끝났다(A5-F5). 여기서는 실행이 실제로 읽은 코드, 곧 불러온 모듈 가운데 해석된 파일이
저장소 안에 있는 것(c_ept뿐 아니라 tools, 보호되지 않은 research_pipeline, 저장소의 __main__)의 해시만 정체성이다. 인터프리터와 site-packages
파일은 환경 잠금의 몫이므로 뺀다. 문서·하네스·불러오지 않은 모듈은 실행 코드와 무관하므로 어느 것도 판정을 바꾸지 않는다. 불러온 파일 하나가
바뀌면(작업 트리, 색인, 삭제, 다른 커밋 어느 쪽이든) ``IdentityChanged``이다. 불러온 파일이 추적되지 않거나 무시 목록에 있으면 시작 커밋이 그
실행을 다시 만들 수 없으므로 역시 거부한다(D11은 "시작 커밋과 불러온 모든 모듈 파일의 해시"를 정체성으로 정의한다).

Git은 시작 시점에 묻는다(``start_gate``: HEAD가 정체성의 커밋과 같은지, 불러온 파일이 수정·추적 안 됨·무시됨이 아닌지). 에폭마다 하는 검사는
해시뿐이다(``verify``, Git 없음). 그래서 느린 Git이 중간에 실행을 끝내지 못한다. 시작 시점에 Git이 시간을 넘기거나 시작조차 되지 않으면 정체성이
바뀐 것이 아니라 "아직 모른다"이므로 실패가 아닌 ``Paused``(다시 할 수 있는 상태)이다. 반대로 Git이 답했는데 "아니오"인 경우(저장소가 아님, 종료 코드가
0이 아님)는 정체성을 확인할 수 없으므로 이전 검사들처럼 거부하되, 증명된 변경(``IdentityChanged``)과 구별되도록 하위 형
``IdentityUnconfirmed``(``proven`` 이 False)로 올린다. 증명된 변경은 그 실행을 다시 하면 안 되고, 확인 불가는 환경을 고친 뒤 다시 시작하면 된다.
읽을 수는 있으나 지금 읽지 못하는 파일(다른 프로세스가 잠금)은 바뀐 증거가 아니므로 짧게 다시 시도한 뒤(약 1 s 동안 3번) 그래도 안 되면
``Paused``이고, 바뀐 증거가 이미 있으면 ``IdentityChanged``가 이긴다.

분할 정책 모듈(``c_ept.cohort``, ``research_pipeline.cohort_policy``, ``research_pipeline.holdout_r3``)은 이름만 ``protected``에 남기고 파일을 열지도,
읽지도, 해시하지도 않는다. 외부 분할 모듈(``holdout_r3`` 폴더에 있는 ``split``)도 같다. 이름으로, 파일 경로의 글자로(파일 시스템을 부르지 않고),
마지막으로 링크를 푼 경로로 세 번 거른다. 정체성이 보호 파일을 담지 못하도록 생성자가 글자 수준에서 막는다. Windows는 경로 조각 끝의 ``.``와
공백을 지우고 8.3 짧은 이름도 같은 파일로 풀기 때문에 그런 조각(``cohort.py.``, ``c_ept./``, ``COHORT~1.PY``)은 정체성 경로로 받지 않는다. 그래도
``verify``는 파일을 열기 전에 경로를 풀어(``resolve``는 읽기 접근 없이 이름만 푼다) 저장소 안의 같은 상대 경로로 돌아오는지, 보호 경로가 아닌지
다시 확인한다. capture 뒤에 정체성 경로에 놓인 링크나 정션이 보호 파일이나 저장소 밖 파일로 이어져도 열지 않고 "바뀜"으로 센다.

Git을 부를 때는 저장소의 위치를 바꿀 수 있는 환경 변수(``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``, ``GIT_COMMON_DIR``,
``GIT_CEILING_DIRECTORIES``, ``GIT_OBJECT_DIRECTORY``, ``GIT_ALTERNATE_OBJECT_DIRECTORIES``, ``GIT_NAMESPACE``, ``GIT_PREFIX``, ``GIT_CONFIG*``)를 지운다.
훅 안에서처럼 이 값들이 남아 있으면 ``git -C``가 다른 저장소나 색인을 보고 거짓 "깨끗함"을 낼 수 있다.

Resume (K5 requirement). ``verify`` 하나로는 이어하기가 같은 코드를 돌린다고 증명되지 않는다. (1) 나중에 처음 불러오는 모듈(지연 import), (2) 다른
import 집합, (3) 정체성 파일을 가리는 새 패키지(예: ``io.py`` 옆에 ``io/__init__.py`` 를 만들면 다음 import는 패키지를 쓰는데 ``verify`` 는 옛
``io.py`` 만 다시 해시해 통과한다)는 ``verify`` 를 빠져나간다. 한 프로세스 안에서 에폭마다 하는 검사는 ``verify`` 로 충분하다. 이어하기에서
K5는 같은 진입 모듈로 ``capture`` 를 다시 해서 저장된 정체성과 같은지(파일 집합과 해시와 보호 목록) 확인한 뒤 ``start_gate`` 를 해야 한다.

한계와 체크아웃 종속성. 정체성의 해시는 작업 트리 파일의 바이트이다(파이썬이 실행하는 바이트이고 Git 없이 검사할 수 있다). ``core.autocrlf`` 와 ``eol``
설정에 따라 같은 커밋의 체크아웃마다 바이트가 다를 수 있으므로, 정체성은 체크아웃에 묶인다. 다른 체크아웃과의 비교는 커밋과 ``start_gate`` 로 해야
한다. ``start_gate`` 는 ``git status`` 를 쓰므로 ``skip-worktree`` 나 ``assume-unchanged`` 표시가 붙은 불러온 파일의 수정은 보지 못한다(그 파일은 ``verify``
가 capture 이후의 변경만 잡는다). 저장소 밖의 코드(서드파티, LEGACY ``gameplay``)는 환경 잠금과 ``legacy.verify()`` 핀의 몫이다. 이 모듈은 capture 시점에
불러온 모듈만 안다.
"""
from __future__ import annotations

import importlib
import os
import re
import site
import subprocess
import sys
import sysconfig
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from c_ept.run import io as run_io

SCHEMA = 'ept-run-identity-v1'
# The protected split modules, as in comparison.protocol.PROTECTED (a test pins the equality). Their bodies are never opened.
PROTECTED_PATHS = ('c_ept/cohort.py', 'research_pipeline/cohort_policy.py', 'research_pipeline/holdout_r3')
PROTECTED_MODULES = tuple(path.removesuffix('.py').replace('/', '.') for path in PROTECTED_PATHS)
PROTECTED_SEGMENT = 'holdout_r3'                                   # a path part or a dotted-name part with this text is protected wherever it is
_LIBRARY_FOLDERS = ('site-packages', 'dist-packages')              # installed third-party code is part of the environment lock, not of the identity
# Variables that move Git to another repository, work tree, index, object store or configuration (set, for example, inside a hook).
_GIT_REDIRECTS = frozenset({'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_CEILING_DIRECTORIES', 'GIT_OBJECT_DIRECTORY',
                            'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE', 'GIT_PREFIX', 'GIT_CONFIG', 'GIT_CONFIG_PARAMETERS'})

_GIT = ('git',)                                                    # the Git command; a test replaces it by a child that hangs or fails
_DEFAULT_TIMEOUT_S = 30                                            # as run_s4a.repository_state
_STATUS_CHUNK = 100                                                # paths per ``git status`` call: Windows limits a command line to 32,767 characters
_NAMED = 20                                                        # at most this many paths are written into a message (the attribute holds all)
_RETRIES = 3                                                       # a file that cannot be read is tried this many times ...
_RETRY_PAUSE_S = 0.5                                               # ... with this pause between the tries (about 1 s in all) before ``Paused``
_HASH = re.compile(r'[0-9a-f]{64}')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_FIELDS = {'schema', 'commit', 'files', 'protected'}


class IdentityChanged(Exception):
    """The code of the run is not the code the identity names. K5 maps it to the gate refusal; the run must not be resumed on this code.

    ``paths`` holds the repository-relative paths that differ (empty when the commit differs). ``proven`` is True: the difference was seen.
    The message names paths only, at most 20 of them; it never carries file content.
    """

    proven = True

    def __init__(self, reason: str, paths: Iterable[str] = ()):
        self.paths = tuple(sorted(set(paths)))
        names = ', '.join(self.paths[:_NAMED]) + (f' and {len(self.paths) - _NAMED} more' if len(self.paths) > _NAMED else '')
        super().__init__(f'{reason}: {names}' if self.paths else reason)


class IdentityUnconfirmed(IdentityChanged):
    """Git could not confirm the identity (non-zero exit, an unexpected answer, a folder that is not a repository root). Still a refusal and
    still an ``IdentityChanged``, but ``proven`` is False: nothing was seen to differ. K5 can tell it from a proven change: fix the environment
    and start again."""

    proven = False


class Paused(Exception):
    """The identity could not be checked now (Git too slow, Git that cannot start, a file that cannot be read now). This is a resumable
    state, not a failure and not a change: the run is started again later. K5 maps it to its own exit path."""


def _default_repo() -> Path:
    """The repository that holds the ``c_ept`` package this module belongs to: ``<R>/c_ept/run/identity.py`` -> ``R``."""
    return Path(__file__).resolve().parents[2]


def _is_protected_module(name: str) -> bool:
    return any(name == module or name.startswith(module + '.') for module in PROTECTED_MODULES) or PROTECTED_SEGMENT in name.split('.')


def _is_protected_path(relative: str) -> bool:
    """The rule of ``comparison.protocol.verify_source`` (equal to a protected path or below it, case-insensitively), and any part ``holdout_r3``."""
    low = relative.replace('\\', '/').casefold()
    return PROTECTED_SEGMENT in low.split('/') or any(low == path.casefold() or low.startswith(path.casefold() + '/') for path in PROTECTED_PATHS)


def _protected_by_text(file: str) -> bool:
    """True when the text of a module's ``__file__`` ends in, or runs through, a protected path, or has a part ``holdout_r3``. Decides by the text
    alone: the file system is not asked about a protected file (no ``resolve``, no ``stat``), whatever the folder in front of ``c_ept/...`` is."""
    low = file.replace('\\', '/').casefold()
    return any(low.endswith('/' + path.casefold()) or ('/' + path.casefold() + '/') in low for path in PROTECTED_PATHS) or _is_protected_path(low)


def _valid_relative(path) -> bool:
    """A repository-relative POSIX path of a file: a str with no backslash, drive colon, control character or empty part, not absolute, no ``..``.

    Also refused, because Windows maps them to another file: a part that ends in ``.`` or a space (``cohort.py.``, ``c_ept./``), and a part with ``~``
    (8.3 short names such as ``COHORT~1.PY``). No module path of this repository has either.
    """
    if not isinstance(path, str) or not path or '\\' in path or ':' in path or any(ord(ch) < 32 for ch in path):
        return False
    pure = PurePosixPath(path)
    if any(part.endswith(('.', ' ')) or '~' in part for part in pure.parts):
        return False
    return bool(pure.parts) and not pure.is_absolute() and '..' not in pure.parts and pure.as_posix() == path


@dataclass(frozen=True)
class Identity:
    """``commit``: the HEAD commit (40 hex digits). ``files``: repository-relative POSIX path -> SHA-256 of every imported module file of the
    repository. ``protected``: the names of the protected modules that were imported and deliberately not hashed.

    Validated and normalised on construction (``IORefused`` for anything else, without echoing a key or a value): ``files`` becomes a read-only
    copy sorted by path and ``protected`` a sorted tuple without repeats. A protected path, and a path that Windows would alias to another file,
    are refused here, so that ``verify`` is never asked to open one.
    """

    commit: str
    files: Mapping[str, str]
    protected: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.commit, str) or _COMMIT.fullmatch(self.commit) is None:
            raise run_io.IORefused('identity commit must be 40 lowercase hex digits')
        if not isinstance(self.files, Mapping):
            raise run_io.IORefused('identity files must be a mapping of path to hash')
        files = {}
        for path, digest in self.files.items():
            if not _valid_relative(path):
                raise run_io.IORefused('identity file path must be a repository-relative POSIX path')
            if _is_protected_path(path):
                raise run_io.IORefused('a protected module cannot be a file of the identity')
            if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
                raise run_io.IORefused('identity file hash must be 64 lowercase hex digits')
            files[path] = digest
        if isinstance(self.protected, (str, bytes)) or not isinstance(self.protected, Iterable):
            raise run_io.IORefused('identity protected must be a list of module names')
        names = list(self.protected)
        if not all(isinstance(name, str) and name for name in names):
            raise run_io.IORefused('identity protected must be a list of module names')
        object.__setattr__(self, 'files', MappingProxyType(dict(sorted(files.items()))))
        object.__setattr__(self, 'protected', tuple(sorted(set(names))))

    def __hash__(self):
        return hash((self.commit, tuple(self.files.items()), self.protected))

    def to_record(self) -> dict:
        """The plain-JSON form (``dict``, ``list``, ``str``) that ``run.io.write_json_atomic`` writes."""
        return {'schema': SCHEMA, 'commit': self.commit, 'files': dict(self.files), 'protected': list(self.protected)}

    @classmethod
    def from_record(cls, record) -> Identity:
        """The identity of a record read by ``run.io.read_json``; ``IORefused`` for a missing or extra field, another schema or a bad value."""
        if not isinstance(record, Mapping) or set(record) != _FIELDS:
            raise run_io.IORefused('identity record fields mismatch')
        if record['schema'] != SCHEMA:
            raise run_io.IORefused('identity record schema mismatch')
        if not isinstance(record['files'], dict) or not isinstance(record['protected'], list):
            raise run_io.IORefused('identity record files must be an object and protected a list')
        return cls(commit=record['commit'], files=record['files'], protected=tuple(record['protected']))


def _deadline(timeout_s) -> float:
    if not timeout_s > 0:
        raise ValueError('timeout_s must be positive')
    return time.monotonic() + timeout_s


def _git_env() -> dict:
    """The environment for Git: the process environment without the variables that move Git to another repository, work tree, index, object
    store or configuration (``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``, ... and every ``GIT_CONFIG_*``), with ``GIT_OPTIONAL_LOCKS=0`` so that a
    read-only check does not rewrite the index of a checkout in use."""
    env = {key: value for key, value in os.environ.items() if key.upper() not in _GIT_REDIRECTS and not key.upper().startswith('GIT_CONFIG_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    return env


def _run_git(repo: Path, args: Sequence[str], deadline: float, timeout_s: float) -> str:
    """The standard output of ``git -C repo <args>``. ``Paused`` when the deadline has passed, Git is too slow or cannot be started;
    ``IdentityUnconfirmed`` when Git answers with a non-zero exit code (Git that cannot answer is never read as clean)."""
    remaining = deadline - time.monotonic()
    slow = f'git did not answer within {timeout_s:g} s; the run is paused, not failed, and may be started again'
    if remaining <= 0:
        raise Paused(slow)
    try:
        done = subprocess.run([*_GIT, '--literal-pathspecs', '-C', str(repo), *args], env=_git_env(), capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=remaining, check=False)
    except subprocess.TimeoutExpired as error:
        raise Paused(slow) from error
    except OSError as error:
        raise Paused(f'git could not be started ({type(error).__name__}); the run is paused, not failed, and may be started again') from error
    if done.returncode != 0:
        raise IdentityUnconfirmed(f'git could not answer (exit {done.returncode}), so the code identity cannot be confirmed')
    return done.stdout


def _head(repo: Path, deadline: float, timeout_s: float) -> str:
    """The HEAD commit of ``repo``, which must be the top level of its Git repository (as ``run_s4a.repository_state`` requires).

    The top-level check matters: a path in the identity is relative to R, so asked from a subfolder a pathspec would match nothing and the
    status would be empty, a false "clean".
    """
    lines = _run_git(repo, ('rev-parse', '--show-toplevel', 'HEAD'), deadline, timeout_s).splitlines()
    if len(lines) != 2 or _COMMIT.fullmatch(lines[1].strip()) is None:
        raise IdentityUnconfirmed('git answered in an unexpected form, so the code identity cannot be confirmed')
    if Path(lines[0].strip()).resolve() != Path(repo).resolve():
        raise IdentityUnconfirmed('the folder is not the root of its git repository, so the paths of the identity cannot be checked in it')
    return lines[1].strip()


def _read_hash(path: Path):
    """The SHA-256 of the file ``path``; ``None`` when it is not a file (missing, or a folder). An ``OSError`` that is not "missing" (a lock held by
    another process) is tried ``_RETRIES`` times with a pause of ``_RETRY_PAUSE_S`` before the last one is raised."""
    last = None
    for attempt in range(_RETRIES):
        try:
            if not path.is_file():
                return None
            return run_io.sha_file(path)
        except FileNotFoundError:
            return None                                               # removed between the check and the open
        except OSError as error:                                      # is_file raises for a stat error that is not "missing" on Python up to 3.12
            last = error
            if attempt + 1 < _RETRIES:
                time.sleep(_RETRY_PAUSE_S)
    raise last


def _module_file(module):
    """The ``__file__`` text of a module taken from its namespace, so that a module-level ``__getattr__`` is never run; ``None`` when it has none."""
    namespace = getattr(module, '__dict__', None)
    file = namespace.get('__file__') if isinstance(namespace, dict) else None
    return file if isinstance(file, str) else None


def _interpreter_roots(base: Path) -> list:
    """Folders of the interpreter and its installed libraries (``sys.prefix`` and the like, the ``sysconfig`` install paths, the site folders),
    resolved, leaving out any that is the repository ``base`` or contains it (then no file of the repository would be left)."""
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots.update(value for value in sysconfig.get_paths().values() if isinstance(value, str))
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except AttributeError:                                            # an old virtualenv's site module has neither
        pass
    resolved = []
    for root in roots:
        try:
            resolved.append(Path(root).resolve())
        except (OSError, ValueError):
            continue
    return [root for root in resolved if not base.is_relative_to(root)]


def capture(modules: Iterable[str], *, repo=None, timeout_s=_DEFAULT_TIMEOUT_S) -> Identity:
    """Import the entry modules, then bind the identity: the HEAD commit and the SHA-256 of every loaded module whose file lies under the repository.

    ``modules``: dotted names of the entry modules (not one string). ``repo``: the repository root, by default the one that holds this
    package. ``timeout_s``: the time budget of the one Git call (``Paused`` beyond it). Git is asked first, so that a missing Git stops the
    run before the (long) imports. Every loaded module whose resolved file lies under the repository is hashed, whatever its name (``c_ept``,
    ``tools``, a non-protected ``research_pipeline`` module, a repository ``__main__``); files under the interpreter roots and any ``site-packages``
    or ``dist-packages`` folder below the repository are left out. A protected module is only named, by its name, by the text of its path
    or by its resolved path, and its file is not opened; a module in a ``holdout_r3`` folder anywhere is protected. A file that cannot be read is
    tried 3 times, then ``Paused``. A bare string, a protected entry module, a non-positive ``timeout_s`` and an identity with no file at all
    are refused (``TypeError`` / ``ValueError``). The hash is that of the file at this moment; the import error of an entry module propagates.

    The identity covers the modules that are loaded now. On a resume the caller captures again with the same entry modules and requires the
    result to equal the saved identity (see the module docstring, "Resume").
    """
    if isinstance(modules, (str, bytes)):
        raise TypeError('modules must be an iterable of module names, not one string')
    entries = list(modules)
    for name in entries:
        if not isinstance(name, str):
            raise TypeError('a module name must be a str')
        if _is_protected_module(name):
            raise ValueError('a protected module cannot be an entry module of a run identity')
    deadline = _deadline(timeout_s)
    root = Path(repo) if repo is not None else _default_repo()
    commit = _head(root, deadline, timeout_s)
    for name in entries:
        importlib.import_module(name)
    base = root.resolve()
    skipped = _interpreter_roots(base)
    files, protected = {}, set()
    for name, module in list(sys.modules.items()):
        if _is_protected_module(name):
            protected.add(name)                                      # named; its file is not looked at
            continue
        file = _module_file(module)
        if file is None:
            continue                                                  # a namespace package or a built-in has no file
        if _protected_by_text(file):
            protected.add(name)
            continue
        try:
            path = Path(file).resolve()
        except (OSError, ValueError):
            continue
        if not path.is_relative_to(base) or any(path.is_relative_to(other) for other in skipped):
            continue                                                  # not under the repository, or the interpreter's own file
        relative = path.relative_to(base).as_posix()
        if any(part.casefold() in _LIBRARY_FOLDERS for part in relative.split('/')):
            continue
        if _is_protected_path(relative):
            protected.add(name)                                       # a protected file under another name or through a link
            continue
        try:
            digest = _read_hash(path)
        except OSError as error:
            raise Paused(f'a module file cannot be read now ({type(error).__name__}): {relative}; the run is paused, not failed, '
                         f'and may be started again') from error
        if digest is None:
            raise IdentityChanged('a loaded module file is no longer a file', [relative])
        files[relative] = digest
    if not files:
        raise ValueError('no imported module of the repository was found, so there is nothing to bind the identity to')
    return Identity(commit, files, tuple(protected))


def _resolves_to_itself(base: Path, path: Path, relative: str) -> bool:
    """True when ``path`` (= base / relative) is the file it names: resolved, it is inside ``base`` at the same relative path (compared without
    case) and that path is not protected. ``resolve`` only looks the name up (no read access to the target). A link or junction placed at an
    identity path after capture, or any alias, fails here and the target is not opened."""
    try:
        inside = path.resolve().relative_to(base).as_posix()
    except (OSError, RuntimeError, ValueError):
        return False
    return inside.casefold() == relative.casefold() and not _is_protected_path(inside)


def verify(saved: Identity, *, repo=None) -> None:
    """Re-hash the files of ``saved``; ``IdentityChanged`` naming every path that differs, is no longer a file, or no longer resolves to itself.
    No Git is run. Before a file is opened its path is resolved and must come back to the same repository-relative path (and not a protected one).

    A file that exists but cannot be read (locked by another process) is tried 3 times in about 1 s. If it still cannot be read it is no
    evidence of a change: ``Paused``, unless a change is already proven, in which case ``IdentityChanged`` is raised for the proven ones.
    This is the in-process check of every epoch; a resume needs ``capture(...) == saved`` and ``start_gate`` as well (module docstring).
    """
    root = Path(repo) if repo is not None else _default_repo()
    base = root.resolve()
    changed, unreadable = [], []
    for relative, expected in saved.files.items():
        path = root / relative
        if not _resolves_to_itself(base, path, relative):
            changed.append(relative)                                  # an alias or a link: never opened
            continue
        try:
            actual = _read_hash(path)                                 # None: missing, or replaced by a folder
        except OSError:
            unreadable.append(relative)
            continue
        if actual != expected:
            changed.append(relative)
    if changed:
        raise IdentityChanged('imported code changed since the identity was bound', changed)
    if unreadable:
        raise Paused(f'imported files cannot be read now ({len(unreadable)} of {len(saved.files)}); the run is paused, not failed, and may be '
                     f'started again: {", ".join(sorted(unreadable)[:_NAMED])}')


def start_gate(repo, identity: Identity, *, timeout_s=_DEFAULT_TIMEOUT_S) -> None:
    """The start-time Git check: HEAD must be ``identity.commit``, and no file of the identity may be modified, untracked or ignored.

    ``git status --porcelain --untracked-files=all --ignored=matching -- <the identity files>``: staged and unstaged edits, deletions and renames
    of those files are refused, and so is an imported file that is untracked or ignored (the start commit could not reproduce the run); every file
    outside the identity (documentation, harness, unimported modules, untracked or not) does not matter. The paths go in chunks of 100 so the
    command line stays short. All Git calls share one deadline ``timeout_s``; beyond it, or when Git cannot be started, ``Paused``.
    ``IdentityChanged`` for another HEAD or a modification; ``IdentityUnconfirmed`` (a subclass, ``proven`` False) when Git cannot answer or the
    folder is not the root of its repository. Git runs without the variables that redirect it (``GIT_DIR`` and the like).
    With no file in the identity only the commit is checked (an empty pathspec would list the whole tree).
    """
    root = Path(repo)
    deadline = _deadline(timeout_s)
    head = _head(root, deadline, timeout_s)
    if head != identity.commit:
        raise IdentityChanged(f'the HEAD commit {head[:12]} is not the commit {identity.commit[:12]} of the identity')
    paths = list(identity.files)
    changed = set()
    dirty = False
    for start in range(0, len(paths), _STATUS_CHUNK):
        output = _run_git(root, ('status', '--porcelain', '--untracked-files=all', '--ignored=matching', '--', *paths[start:start + _STATUS_CHUNK]),
                          deadline, timeout_s)
        for line in output.splitlines():
            if not line.strip():
                continue
            dirty = True
            for part in line[3:].split(' -> '):                       # "XY path", or "XY old -> new" for a rename; "??" untracked, "!!" ignored
                if part.strip():
                    changed.add(part.strip().strip('"'))
    if dirty:
        raise IdentityChanged('imported files are modified, untracked or ignored in the working tree', changed)
