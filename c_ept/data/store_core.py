r"""
Chunked Data Store Format Layer / 청크형 대용량 데이터 저장소 포맷 및 원자적 영속화 계층

[English]
This module manages low-level storage for large-scale match dataset arrays (S1 and S2 stores):
  - Storage Layout: Directory containing `.npy` memory-mapped arrays and a `STORE.json` manifest.
  - Fault Tolerance & Resumability: Writers populate memmap slices and flush them; the orchestrator
    invokes `mark_done`, which fsyncs files to disk before committing block indices in `STORE.json`.
  - Determinism (`block_plan`): Data is partitioned strictly along match boundaries so that array
    layouts are completely invariant to worker concurrency or block sizes.
  - Integrity Verification (`Store.verify`): Computes SHA-256 hashes across all stored arrays upon
    finalization and verifies them against the recorded manifest.

[한국어 설명]
대규모 리그 오브 레전드 경기 데이터를 디스크 상에서 안전하고 고성능으로 다루기 위한 청크형 데이터 저장소 계층입니다:
  - 저장 구조: 배열별 `.npy` 파일(메모리 맵 `memmap` 기반)과 `STORE.json` 메타데이터 매니페스트.
  - 중단 재개(Resume) 및 안정성: 각 프로세스가 담당 청크를 기록한 뒤 `mark_done`을 통해 디스크에 `fsync`하고 블록 완료 상태를 기록.
  - 결정론적 블록 분할(`block_plan`): 경기가 쪼개지지 않도록 경기 경계에서만 블록을 분할하여 병렬 워커 수나 블록 크기에 무관하게 항상 동일한 바이트 배치 보장.
  - 무결성 검증(`verify`): 완료 시점의 모든 `.npy` 파일에 대한 SHA-256 해시를 기록하고 이후 위변조 및 누락 여부를 검증.
"""
# Ref: D:\LOL_Data\runs\20261019_r2_store_impl_1\dev\common_training_r2\store.py (block_plan, require_free_bytes, fsync_file, Store) and
#      common.py (write_atomic, sha): the vendored source; its SHA-256 values are in the header above.
# Ref: docs/superpowers/specs/2026-10-19-chunked-data-store-design.md sections 4.1 (one .npy per array opened as memmap, blocks of whole
#      matches), 4.5 (block size, worker count and restarts do not change the stored values) and 4.6 (STORE.json written atomically;
#      resume).
from __future__ import annotations

import hashlib
import json
import operator
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def sha(path) -> str:
    """SHA-256 of a file as lower-case hex.

    수식
    ----
    (STO-3)  h(file) = SHA-256(file의 바이트)                 (소문자 16진수)

      기호    | 뜻                          | 단위 | 범위        | 코드 변수
      file    | 파일 하나                   | —    | —           | path
      h(file) | 그 파일 내용의 SHA-256 요약 | —    | 64자 문자열 | 반환값

    해석
    ----
    요약값은 파일의 바이트만 보므로 이름, 위치, 시각이 달라도 내용이 같으면 같고, 한 바이트라도 다르면 충돌이 없는 한 달라진다. 그래서 저장소를 끝낸 뒤 기록한 요약값은 파일 내용이
    바뀌지 않았다는 증거가 된다((STO-4)).

    근거
    ----
    저장소 설계 4.1절(매니페스트의 배열별 SHA-256), S1–S2 계획 Task 7.
    Ref: NIST (2015), "Secure Hash Standard (SHS)", FIPS PUB 180-4; doi:10.6028/NIST.FIPS.180-4.
    """
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_atomic(path, data) -> None:
    """Temporary file, flush, fsync, then rename over the target: a reader sees the old or the new file, never half.

    ``data`` is bytes or str (written as UTF-8). The folder of ``path`` must exist.

    임시 파일에 쓰고 fsync한 뒤 이름을 바꾸므로 읽는 쪽은 옛 파일이나 새 파일 전체를 보고 반쯤 쓰인 파일은 보지 않는다. 쓰다가 멈추면 임시 파일만 남고 대상은 그대로이다(저장소 설계 4.6절).
    """
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data.encode('utf-8') if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def block_plan(match_id, block_matches) -> list[tuple[int, int]]:
    """
    Match-Boundary Block Partitioning / 경기 경계 기반 결정론적 블록 분할

    [English]
    Computes [start, end) index ranges for chunked parallel processing:
      - Rows belonging to the same match are guaranteed to stay within the same block.
      - Each block contains at most `block_matches` consecutive matches.
      - Independent of worker count, ensuring bit-for-bit reproducibility.

    [한국어 설명]
    병렬 데이터 처리를 위해 행 인덱스를 [시작, 끝) 구간의 블록으로 분할합니다:
      - 동일한 경기에 속한 행들은 절대 두 블록으로 나뉘지 않고 동일 블록에 보존됩니다.
      - 각 블록은 최대 `block_matches`개의 연속된 경기들을 포함합니다.
      - 워커 스레드/프로세스 수와 무관하게 항상 일관된 파일 레이아웃을 생성합니다.
    """
    if isinstance(block_matches, (bool, np.bool_)):
        raise TypeError(f'block_matches must be an int, not {type(block_matches).__name__}')
    k = operator.index(block_matches)
    if k < 1:
        raise ValueError(f'block_matches must be at least 1, got {k}')
    match_id = np.asarray(match_id)
    if not len(match_id):
        return []
    # 식 (STO-1): 경기가 바뀌는 행 start_u 가운데 block_matches번째마다를 블록의 시작으로 하고, 마지막 블록의 끝은 N이다.
    starts = np.flatnonzero(np.r_[True, match_id[1:] != match_id[:-1]])[::k].tolist() + [len(match_id)]
    return list(zip(starts[:-1], starts[1:]))


def require_free_bytes(path, nbytes, margin=1.1) -> None:
    """Raise OSError unless the disk that holds ``path`` (or its nearest existing parent) has ``nbytes * margin`` bytes free.

    수식
    ----
    (STO-2)  free ≥ margin·nbytes이면 통과하고, free < margin·nbytes이면 OSError를 낸다

      기호   | 뜻                                                                | 단위   | 범위                            | 코드 변수
      free   | path가 놓일 디스크(경로가 없으면 가장 가까운 상위 폴더)의 남은 양 | 바이트 | ≥ 0                             | shutil.disk_usage(anchor).free
      nbytes | 앞으로 쓸 바이트 수                                               | 바이트 | ≥ 0                             | nbytes
      margin | 여유 배수                                                         | —      | 기본 1.1 (값은 검사하지 않는다) | margin

    해석
    ----
    식 (STO-2)는 쓰기를 시작하기 전에 디스크가 모자라 도중에 멈추는 일을 막는 사전 검사이고, 배수는 요구량보다 조금 더 남겨 두는 여유이다. 경로가 아직 없으면 가장 가까운 상위 폴더가 놓인
    디스크를 재며, 남은 양이 필요량과 정확히 같으면 통과한다.

    근거
    ----
    S1–S2 계획 Task 7(require_free_bytes), S1–S2 판정 R13-k(할당 전에 공간 확인).
    """
    # 식 (STO-2)의 free: 경로가 아직 없으면 있는 가장 가까운 상위 폴더가 놓인 디스크를 잰다.
    anchor = Path(path)
    while not anchor.exists():
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free
    need = nbytes * margin
    if free < need:
        raise OSError(f'need {need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB at {anchor}')


def fsync_file(path) -> None:
    """Flush a file's data to disk (opened read-write; O_BINARY where the platform has it).

    완료 표시를 남기기 전에 표시와 함께 넘긴 파일마다 이 호출을 하므로(공통 가정 (나)) 그 파일들의 데이터는 표시보다 먼저 디스크에 보내진다.
    """
    fd = os.open(path, os.O_RDWR | getattr(os, 'O_BINARY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Store:
    """
    Chunked Memory-Mapped Array Store / 청크형 메모리 매핑 배열 저장소 관리자

    [English]
    Manages `.npy` memory-mapped arrays and their JSON manifest:
      - `allocate(name, shape, dtype)`: Creates or maps existing `.npy` array files.
      - `put_rows(start, arrays)`: Writes array blocks into memory-mapped regions with strict dtype checks.
      - `mark_done(stage, block, names)`: Fsyncs written files before recording completed blocks in STORE.json.
      - `finalize()`: Computes cryptographic SHA-256 signatures for all files and locks manifest.
      - `verify()`: Audits files against stored manifest hashes.

    [한국어 설명]
    `.npy` 메모리 맵(memmap) 파일들과 메타데이터 매니페스트(`STORE.json`)를 관리합니다:
      - `allocate`: 배열 파일 사전 생성 및 열기 (자료형/형태 일치 검증).
      - `put_rows`: 지정된 행 구간에 넘파이 배열 데이터를 복사하고 디스크에 flush.
      - `mark_done`: 파일 fsync 완료 후 매니페스트에 완료 블록 번호를 기록 (트랜잭션 안전성).
      - `finalize`: 모든 저장 파일의 SHA-256 해시를 기록하고 완료 표기.
      - `verify`: 기록된 해시와 현재 파일 상태를 대조하여 무결성 검증.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.file = self.root / 'STORE.json'
        self.maps = {}
        self.manifest = (json.loads(self.file.read_text(encoding='utf-8')) if self.file.exists()
                         else {'version': 1, 'stages': {}, 'arrays': {}, 'files': {}, 'complete': False})

    def save(self) -> None:
        """Write STORE.json atomically (the root folder is created when needed)."""
        self.root.mkdir(parents=True, exist_ok=True)
        write_atomic(self.file, json.dumps(self.manifest, ensure_ascii=False, indent=1))

    def path(self, name: str) -> Path:
        return self.root / (name + '.npy')

    def allocate(self, name, shape, dtype):
        """Create the array and record it in the manifest, or open an existing one (r+) after checking that shape and dtype agree.

        배열 파일을 미리 잡아 두고, 이미 있으면 모양과 자료형이 같을 때만 연다(저장소 설계 4.1절). 이어하기가 다른 모양의 파일 위에 쓰는 일을 막는 조건이며, 이미 있는 파일은 열기만 하고
        매니페스트에는 다시 올리지 않는다.
        """
        p = self.path(name)
        shape = tuple(int(s) for s in shape)
        dtype = np.dtype(dtype)
        if p.exists():
            a = np.load(p, mmap_mode='r+')
            if a.shape != shape or a.dtype != dtype:
                raise ValueError(f'{name}: stored {a.shape} {a.dtype}, asked {shape} {dtype}')
            return a
        p.parent.mkdir(parents=True, exist_ok=True)
        a = np.lib.format.open_memmap(p, mode='w+', dtype=dtype, shape=shape)
        self.manifest['arrays'][name] = {'shape': list(shape), 'dtype': dtype.str}
        self.save()
        return a

    def array(self, name):
        """The array as a read-only memmap (kept open for the life of the Store).

        읽는 쪽은 읽기 전용 memmap만 받으므로 읽다가 저장소의 값을 바꾸지 못한다.
        """
        if name not in self.maps:
            self.maps[name] = np.load(self.path(name), mmap_mode='r')
        return self.maps[name]

    def _rows_view(self, name, v):
        """Writable memmap of an allocated array; dtype and row shape must match ``v``: no silent casts.

        Raises AssertionError with the same message tuple r2's ``assert`` carried, but explicitly: ``python -O`` removes asserts.
        """
        a = np.load(self.path(name), mmap_mode='r+')
        if not (a.dtype == v.dtype and a.shape[1:] == v.shape[1:]):
            raise AssertionError((name, a.dtype, v.dtype, a.shape, v.shape))
        return a

    def put_rows(self, start, arrays) -> None:
        """Write rows [start, start+len) of allocated arrays and flush the views (the caller marks the block afterwards).

        dtype and row shape must match the allocation: no silent casts.

        블록의 행을 자기 자리에 쓰고 flush한다. 자료형과 행 모양이 할당과 다르면 형 변환 없이 거부하여 값이 조용히 바뀌는 일을 막는다(저장소 설계 4.5절).
        """
        for name, v in arrays.items():
            a = self._rows_view(name, v)
            a[start:start + len(v)] = v
            a.flush()
            del a

    def put_at(self, positions, arrays) -> None:
        """Like put_rows for scattered row positions.

        행 구간을 쓸 때와 같은 검사를 하고 흩어진 행 위치에 쓴다.
        """
        for name, v in arrays.items():
            a = self._rows_view(name, v)
            a[positions] = v
            a.flush()
            del a

    def done(self, stage) -> set:
        """The blocks of ``stage`` that are marked done: indices into the block plan the run started with (resume with the same plan).

        끝난 블록은 블록 계획의 번호이므로 이어하기는 같은 계획을 써야 한다(공통 가정 (다)).
        """
        return set(self.manifest['stages'].get(stage, {}).get('blocks', []))

    def mark_done(self, stage, block, names) -> None:
        """fsync the listed files, then record ``block`` as done in ``stage`` and save the manifest.

        ``names`` are array names (str) or paths (Path) of other files, such as block files written with ``write_atomic``. ``block`` is
        a plain int: the manifest is JSON, which does not take a numpy integer.

        넘겨받은 파일을 fsync한 뒤에 블록을 기록하므로 완료 표시가 그 파일들보다 앞서지 않고, 전원이 나가면 표시 없는 블록을 처음부터 다시 쓴다(공통 가정 (나); 저장소 설계 4.6절).
        """
        for n in names:
            fsync_file(n if isinstance(n, Path) else self.path(n))
        blocks = self.manifest['stages'].setdefault(stage, {'blocks': []})['blocks']
        if block not in blocks:
            blocks.append(block)
        self.save()

    def set_stage(self, stage, **info) -> None:
        """Add or update information on ``stage`` (its list of finished blocks is kept) and save the manifest; values must be JSON.

        단계의 상태(계획 해시, 호출 기록 따위)를 매니페스트에 남기며 끝난 블록의 목록은 건드리지 않는다.
        """
        self.manifest['stages'].setdefault(stage, {'blocks': []}).update(info)
        self.save()

    def stored_files(self) -> list[Path]:
        """Every file under the root except STORE.json and ``*.tmp`` leftovers of an interrupted write, sorted.

        요약할 파일의 집합을 정한다. 계획과 시간 기록이 들어 있어 빌드마다 달라지는 STORE.json은 빼고(S1–S2 판정 S2-3), 쓰다 멈춘 *.tmp 찌꺼기도 뺀다.
        """
        return sorted(p for p in self.root.rglob('*') if p.is_file() and p != self.file and not p.name.endswith('.tmp'))

    def hashes(self, files) -> dict[str, str]:
        """{path relative to the root (posix): SHA-256} for files under the root (any iterable of paths; it is read once, into a list).

        (STO-3)의 요약값을 파일마다 구해 {상대 경로: h}로 묶는다((STO-4)가 맞대는 두 표를 이 함수가 만든다).
        """
        files = list(files)
        with ThreadPoolExecutor(4) as pool:
            return dict(zip((p.relative_to(self.root).as_posix() for p in files), pool.map(sha, files)))

    def finalize(self) -> None:
        """Record the SHA-256 of every stored file, mark the store complete and save.

        (STO-4)가 나중에 맞대 볼 {파일: h}를 기록하고 저장소를 끝났다고 표시한다.
        """
        self.manifest['files'] = self.hashes(self.stored_files())
        self.manifest['complete'] = True
        self.save()

    def verify(self) -> None:
        """Every recorded file has its recorded SHA-256 and nothing was added or removed.

        수식
        ----
        (STO-4)  통과 ⇔ complete ∧ (recorded = now)
                 recorded = {파일: h}를 완료 때 매니페스트에 적은 표, now = 지금 저장소 파일들의 {파일: h}이다(STORE.json과 *.tmp는 파일에 들지 않는다).
                 recorded = now ⇔ 두 표의 파일 이름 집합이 같고 같은 이름의 h가 모두 같다. 어긋난 이름의 목록은 앞의 20개를 오류 글에 적는다.

          기호     | 뜻                              | 단위 | 범위     | 코드 변수
          recorded | 완료 때 기록한 파일별 요약값 표 | —    | —        | manifest['files']
          now      | 지금 파일들의 요약값 표         | —    | —        | hashes(stored_files())
          complete | 저장소가 끝났다는 표시          | —    | 참, 거짓 | manifest['complete']

        해석
        ----
        식 (STO-4)는 끝난 저장소가 기록한 그대로인지 확인하며, 파일이 바뀌거나 더해지거나 빠졌거나 아직 끝나지 않았으면 어긋난 이름을 알려 주며 거부한다. 기록에 들지 않는 STORE.json은
        단계 상태와 시간 기록이 들어 있어 요약과 무관하게 바뀔 수 있다(S1–S2 판정 S2-3).

        근거
        ----
        저장소 설계 4.6절(매니페스트와 이어하기), S1–S2 계획 Task 7(verify), S1–S2 판정 S2-3(STORE.json을 뺀 해시 비교).
        """
        recorded = self.manifest['files']
        now = self.hashes(self.stored_files())
        # 식 (STO-4)의 recorded ≠ now: 이름이 한쪽에만 있는 파일과, 이름은 같고 h가 다른 파일을 모은다.
        bad = sorted(set(recorded) ^ set(now)) + sorted(k for k in recorded.keys() & now.keys() if recorded[k] != now[k])
        if not self.manifest.get('complete') or bad:
            raise ValueError(f'store {self.root} does not match STORE.json: {bad[:20]}')
