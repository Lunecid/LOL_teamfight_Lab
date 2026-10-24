"""Date-stamped run folders, the RUN.json manifest and the runs.sqlite index.

A run folder is ``<RUNS_ROOT>/<YYYYMMDD>_c_ept_<stage>_<n>``; the manifest records what produced it.
The manifest may be extended with new keys through ``amend_manifest``, which does not replace existing keys.


방법에서의 위치. 이 모듈은 실행의 설정·코드·규칙·하드웨어 상태를 매니페스트와 실행 색인에 연결한다.
실행 폴더는 날짜·단계·번호로 구분하고, 계산 중 얻은 결과는 기존 키를 덮지 않고 추가한다(Task 1; Task 8a 판정 R7).
입력은 실행 경로와 호출자 정보이며, 출력은 실행 폴더·JSON 기록·SQLite 색인 행이다.
식 목록.
    (RUNLOG-1) 호스트의 전체 물리 RAM을 GiB로 바꾸어 기록한다.
공통 가정. 이 RAM 값은 호스트 용량이며 실행 프로세스의 사용량이나 피크가 아니다.
Git 조회 실패는 미확인 상태로 남기고, 프로세스 탐지의 음성 결과는 탐지 규칙에 걸린 항목이 없다는 뜻이다.
E6는 모델 서버가 실행 측정을 방해하지 않도록 두는 규칙이며, 탐지와 실행 거부는 다른 함수의 역할이다.
키 충돌 방지는 write_manifest·amend_manifest의 검사이고, SQLite 등록은 같은 실행 이름의 행을 교체한다.
근거 위치: 설계서 §9; S1–S2 계획 Task 1; S1–S2 판정 R31·R33·R47.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 9 (run record) and guidelines C7/I (immutable dated runs, run index).
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:      # hardware and Strata probes then report None
    psutil = None

from c_ept import config as C
from c_ept import legacy

HUB = Path(__file__).resolve().parents[1]      # repository root that holds c_ept/
_STAGE = re.compile(r'[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*')
_RUN_NAME = re.compile(r'\d{8}_c_ept_(.+)_\d+')
_STRATA_NAME = re.compile(r'llama|strata|ollama', re.IGNORECASE)
STRATA_BRIDGE = 'strata_worker_mcp'     # the idle MCP bridges of other sessions (Ruling 31): no model, never counted as Strata
STRATA_RULE = ('process name has llama, ollama or strata; or the command line has both strata and server.py (slashes and case ignored); '
               f'a command line with {STRATA_BRIDGE} never counts')
# computed provenance fields: the caller's info may not set them
_COMPUTED = ('run', 'created', 'code_commit', 'code_dirty', 'config_sha256', 'spec_sha256', 'legacy_head', 'legacy_pins_sha256',
             'split_policy_sha256', 'python', 'numpy', 'cpu', 'ram_gib', 'gpu', 'strata_running', 'strata_detail')


def new_run(stage: str, root: Path = C.RUNS_ROOT, today: str | None = None) -> Path:
    """Create and return ``<root>/<today>_c_ept_<stage>_<n>``; n is one above the highest number already used for that day and stage.

    S1–S2 계획 Task 1의 날짜별 실행 이름을 할당하며, 이름 충돌이 나면 다음 번호로 다시 폴더 생성을 시도한다.
    """
    if not _STAGE.fullmatch(stage):
        raise ValueError(f'stage must be letters, digits and single underscores, got {stage!r}')
    today = dt.date.today().strftime('%Y%m%d') if today is None else today
    if not re.fullmatch(r'\d{8}', today):
        raise ValueError(f'today must be YYYYMMDD, got {today!r}')
    root = Path(root)
    prefix = f'{today}_c_ept_{stage}_'
    used = [int(m.group(1)) for p in root.glob(prefix + '*') if (m := re.fullmatch(re.escape(prefix) + r'(\d+)', p.name))]
    n = max(used, default=0) + 1
    while True:
        run = root / f'{prefix}{n}'
        try:
            run.mkdir(parents=True)
            return run
        except FileExistsError:
            n += 1


def sha256_json(obj) -> str:
    """SHA-256 (lower-case hex) of ``obj`` as sorted, compact JSON in UTF-8 (``sort_keys``, separators ``,`` ``:``, ``ensure_ascii`` on).

    The one definition of a JSON hash in a run record: RUN.json (``config_sha256``, the split-policy hash), TERRAIN.json (the hash of the
    ``files`` map of the consumed S1 store) and the S2 run record call this function and do not write the formula again.


    S1–S2 판정 R47의 공통 JSON 해시 정의이며, 같은 직렬화 결과를 비교하는 식별자로 사용한다.
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _split_policy_sha256() -> str:
    """Hash of the role table {patch: [roles(patch, b) for b in range(100)]}; the salt of the split enters no part of it."""
    split = C.split_module()
    return sha256_json({p: [list(split.roles(p, b)) for b in range(100)] for p in C.PATCHES})


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(['git', '-C', str(HUB), *args], capture_output=True, text=True, encoding='utf-8', errors='replace',
                           timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def code_state() -> dict:
    """``{'code_commit': HEAD of the repository, 'code_dirty': whether c_ept/ has uncommitted changes}``, exactly as RUN.json records them.

    ``code_commit`` is the 40-hex commit, or None when git cannot answer. ``code_dirty`` is True when ``git status --porcelain -- c_ept``
    lists anything (modified, staged, deleted or untracked files), False when it lists nothing, and None when git cannot answer (unknown,
    not clean). RUN.json, TERRAIN.json and the S2 run record take both values from here.


    S1–S2 판정 R47의 공통 코드 상태를 기록하며, 조회 실패의 None을 깨끗한 상태로 해석하지 않는다.
    """
    dirty = _git('status', '--porcelain', '--', 'c_ept')
    return {'code_commit': (_git('rev-parse', 'HEAD') or '').strip() or None,
            'code_dirty': None if dirty is None else bool(dirty.strip())}


def _gpu() -> str | None:
    """nvidia-smi name, memory used and total (a read-only query), or None when it is not available."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _process_table():
    """The running processes as [(pid, name, command line)], the command line a list of arguments or None where the process does not let
    it be read (a protected process); None without psutil."""
    if psutil is None:
        return None
    return [(p.info['pid'], p.info['name'], p.info['cmdline']) for p in psutil.process_iter(['pid', 'name', 'cmdline'])]


def _command_line(cmdline) -> str:
    """A command line (a list of arguments, one string or None) as one lower-case string with forward slashes; '' when it is not known."""
    if not cmdline:
        return ''
    text = cmdline if isinstance(cmdline, str) else ' '.join(map(str, cmdline))
    return text.replace('\\', '/').lower()


def strata_check(processes=None) -> dict | None:
    """E6 probe: is a local model server running? A small record, or None when the processes cannot be listed (no psutil).

    The server is ``python.exe D:\\Strata\\serve\\server.py ...`` (start_strata.ps1), so its process name says nothing. A process counts
    when its name has llama, ollama or strata, or when its command line has both ``strata`` and ``server.py`` (slashes and case
    ignored). The idle MCP bridges of other sessions (``strata_worker_mcp``, no model) are excluded first and only counted.
    ``processes`` replaces the machine's process list in tests: an iterable of (pid, name, command line).

    The record: ``running``; ``matched`` [{'name', 'pid'}] in pid order; ``bridges_ignored``; ``scanned`` processes of which
    ``cmdline_unreadable`` showed no command line; ``rule``. A command line is never recorded.


    S1–S2 판정 R31·R33의 서버 탐지 규칙을 적용하며, 이 함수 자체는 실행을 거부하거나 프로세스를 중지하지 않는다.
    명령행을 읽지 못한 프로세스는 이름으로만 판별하므로 음성 결과가 모든 서버의 부재를 증명하지는 않는다.
    """
    table = _process_table() if processes is None else processes
    if table is None:
        return None
    matched, bridges, scanned, unreadable = [], 0, 0, 0
    for pid, name, cmdline in table:
        cmd = _command_line(cmdline)
        scanned += 1
        if not cmd:
            unreadable += 1
        if STRATA_BRIDGE in cmd:
            bridges += 1
        elif _STRATA_NAME.search(name or '') or ('strata' in cmd and 'server.py' in cmd):
            matched.append({'name': name or '', 'pid': int(pid)})
    matched.sort(key=lambda m: (m['pid'], m['name']))
    return {'running': bool(matched), 'matched': matched, 'bridges_ignored': bridges, 'scanned': scanned, 'cmdline_unreadable': unreadable,
            'rule': STRATA_RULE}


def _ram_gib() -> float | None:
    """(RUNLOG-1)의 호스트 메모리 단위 변환을 계산한다."""
    return None if psutil is None else round(psutil.virtual_memory().total / 2 ** 30, 2)


def _cpu() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except (ImportError, OSError):
        name = platform.processor() or platform.machine()
    return f'{name} ({os.cpu_count()} logical)'


def _json_default(o):
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f'{type(o).__name__} is not JSON serializable')


def _write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` (UTF-8, LF): write a temporary file next to it (``<name>.tmp``), flush it, call file ``os.fsync``,
    then ``os.replace``. Power-loss durability depends on the filesystem and platform; this code does not fsync the containing directory.
    The steps of ``store_core.write_atomic``, written out here and not imported from it: ``terrain`` imports this module, and a
    worker that loads the terrain must not load the store layer (``test_terrain`` and ``test_tokens`` pin that)."""
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(text.encode('utf-8'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_manifest(run: Path, info: dict) -> dict:
    """Write ``<run>/RUN.json`` and return it. ``info`` (args, seeds, sizes, ...) is kept as given.

    Refuses (SpecMismatch / LegacyPinError) when the constants differ from the decision record or the LEGACY files from their pins.
    ``_write_atomic`` writes a temporary file, flushes it, calls file ``os.fsync``, then calls ``os.replace``.
    Power-loss durability depends on the filesystem and platform; the containing directory is not fsynced.


    수식
    ----
    (RUNLOG-1) host_gib = round(host_bytes / 2^30, 2)  (psutil 사용 가능); 아니면 None.

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    host_bytes | 호스트 전체 물리 메모리 | byte | psutil 반환값, 별도 검사 없음 | psutil.virtual_memory().total (_ram_gib)
    host_gib | 기록하는 호스트 RAM 용량 | GiB | 위 변환의 float 또는 None | manifest['ram_gib']
    round(·, 2) | Python의 소수 둘째 자리 반올림 | 입력과 같음 | 이진 float의 반올림 규칙 | round

    해석
    ----
    식 (RUNLOG-1)은 호스트 용량 변환이며 실행 자원의 맥락을 남긴다(S1–S2 계획 Task 1).
    psutil이 없으면 미확인을 None으로 남기며 프로세스 사용량으로 대체하지 않는다(Task 1의 하드웨어 기록).
    지정 설정 필드·LEGACY 핀은 대조하지만 코드 커밋·결정 원본 해시를 외부의 기대 해시와 대조하지는 않는다(Task 1).

    근거
    ----
    S1–S2 계획 Task 1(실행 기록), 설계서 §9(추적 정보).
    """
    run = Path(run)
    clash = sorted(set(info) & set(_COMPUTED))
    if clash:
        raise ValueError(f'info may not set provenance fields {clash}')
    spec_sha = C.check_spec()
    pinned = legacy.verify()
    strata = strata_check()
    m = _RUN_NAME.fullmatch(run.name)
    manifest = {'stage': m.group(1) if m else None}
    manifest.update(info)
    manifest.update(
        run=run.name,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        **code_state(),                                              # code_commit and code_dirty
        config_sha256=sha256_json(C.constants()),
        spec_sha256=spec_sha,
        legacy_head=pinned['head'],
        legacy_pins_sha256=legacy.pins_sha256(),
        split_policy_sha256=_split_policy_sha256(),
        python=f'{platform.python_implementation()} {platform.python_version()}',
        numpy=np.__version__,
        cpu=_cpu(),
        ram_gib=_ram_gib(),
        gpu=_gpu(),
        strata_running=None if strata is None else strata['running'],
        strata_detail=strata,
    )
    # (RUNLOG-1)의 용량은 사용량과 구분해 직렬화한다(S1–S2 계획 Task 1).
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + '\n'
    run.mkdir(parents=True, exist_ok=True)
    _write_atomic(run / 'RUN.json', text)
    return json.loads(text)


def amend_manifest(run: Path, extra: dict) -> dict:
    """Add the keys of ``extra`` to ``<run>/RUN.json`` and return the new manifest; no existing key is ever changed.

    This is how a stage records what it learned while it ran (wall time, peak memory, counts) without ``write_manifest`` re-stamping
    the provenance fields. A key that is already in the file raises ValueError and nothing is written; a key that is not a string
    raises TypeError; a run without RUN.json raises FileNotFoundError. Values are written as ``write_manifest`` writes them. The file is
    replaced through the same temporary-file, flush, file-fsync and ``os.replace`` steps and durability limits as ``write_manifest``.


    S1–S2 Task 8a 판정 R7의 추가 기록 절차로, 이미 존재하는 키를 보존하고 새 키만 합친다.
    """
    run = Path(run)
    if not all(isinstance(k, str) for k in extra):
        raise TypeError(f'manifest keys must be strings, got {sorted({type(k).__name__ for k in extra if not isinstance(k, str)})}')
    path = run / 'RUN.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    clash = sorted(set(extra) & set(manifest))
    if clash:
        raise ValueError(f'{path.name} already has {clash}: amend_manifest adds keys and never overwrites one')
    text = json.dumps({**manifest, **extra}, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + '\n'
    _write_atomic(path, text)
    return json.loads(text)


def register(run: Path, manifest: dict, db: Path = C.CATALOG_DB) -> None:
    """Insert or replace this run's row in the run index (table ``runs``); the row key is the run folder name.

    S1–S2 계획 Task 1의 실행 색인에 매니페스트를 저장하며, 같은 실행 폴더 이름의 기존 행은 교체한다.
    """
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    row = (Path(run).name, manifest.get('stage'), manifest.get('created'),
           json.dumps(manifest, sort_keys=True, ensure_ascii=False, default=_json_default))
    with closing(sqlite3.connect(db)) as con, con:
        con.execute('CREATE TABLE IF NOT EXISTS runs(run TEXT PRIMARY KEY, stage TEXT, created TEXT, manifest TEXT)')
        con.execute('INSERT INTO runs(run, stage, created, manifest) VALUES (?, ?, ?, ?) ON CONFLICT(run) DO UPDATE SET '
                    'stage = excluded.stage, created = excluded.created, manifest = excluded.manifest', row)
