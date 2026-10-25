r"""
S2 Store Builder / 다차원 텐서 뷰(View) 병렬 대용량 저장소(S2) 구축기

[English]
Builds the complete S2 feature store (<run>\s2) containing neural network input views:
  - View Layout: Each S1 frame-clock row (src 0 at regular frame time t0) and event row (src 1 = M3 at
    first kill f; src 2 = M4 at engagement end delta) receives a single precomputed multi-modal view.
  - Chunked Multi-Processing: Matches are split into discrete blocks of whole matches (store_core.block_plan).
    Worker processes compute views using `tokens.build_view`, returning ndarray dictionaries.
  - Parent-Worker Isolation: Only the main parent process allocates `.npy` memmap files, flushes writes,
    and updates `STORE.json` with cryptographic SHA-256 signatures upon completion.
  - Fault-Tolerant Resumption: Rigorously verifies S1 dataset signatures, terrain parameters, and block
    plans before resuming partially built stores, preventing partial or corrupt states.

[한국어 설명]
S1 정규 프레임 행과 교전 사건 행들로부터 신경망 입력용 다차원 상태 뷰(View)를 대량 생성하여 S2 스토어(`<run>\s2`)에 병렬로 기록합니다:
  - 뷰 구성:
      * src 0: 분 단위 정규 프레임 시점($t_0$) 관측 뷰.
      * src 1: 교전 첫 킬 발생 시점($f$, M3 헤드 입력) 관측 뷰.
      * src 2: 교전 종료 시점($\delta$, M4 헤드 입력) 관측 뷰.
  - 안전한 병렬화: 경기가 쪼개지지 않도록 경기 단위 블록으로 분할하여 워커 프로세스들에 배분.
  - 원자적 영속화: 자식 프로세스는 순수 넘파이 배열만 반환하며, 메인 부모 프로세스만이 `.npy` memmap 할당, fsync 쓰기, `STORE.json` 체크포인트 기록을 수행.
  - 무결성 및 중단 재개(Resume): 실행 중단 시 이미 기록된 블록을 건너뛰고 안전하게 재개하며, 지형 파일 해시 및 S1 데이터셋 무결성을 엄격히 사전 검증.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 4 (frame clock and event clock: one view per frame-clock row and per
#      open event row; masked M3 and M4 inputs are the frame-clock views of the onsets, decision D1) and section 11 item 8 (the frame of
#      an open view is the last regular frame strictly before the reference time).
# Ref: docs/superpowers/specs/2026-10-19-chunked-data-store-design.md sections 4.1 (one .npy per array, blocks of whole matches), 4.5 (block
#      size, worker count and restarts do not change the stored values) and 4.6 (STORE.json written atomically; resume with the same plan).
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import gc
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:      # the peak memory is then not recorded
    psutil = None

from c_ept import config as C
from c_ept.data import reader
from c_ept import runlog
from c_ept.data import store_core
from c_ept.preprocessing import terrain, tokens
from c_ept.data.casts import narrow
from c_ept.data.store_core import Store

STORE_DIR = 's2'                                   # under the run folder
SRC_FRAME, SRC_M3, SRC_M4 = 0, 1, 2                # the values of views/src
SRC_NAMES = ('frame_clock', 'M3', 'M4')            # by src value, for the run record
VIEWS_COLUMNS = (('src', 'i1'), ('src_index', 'i4'), ('match', 'i4'), ('ref', 'i8'), ('t0', 'i8'), ('frame_k', 'i2'))
VIEWS_BYTES_PER_VIEW = sum(np.dtype(d).itemsize for _, d in VIEWS_COLUMNS)           # 27
REFUSED_ROLES = tuple(reader.HOLDOUT_ROLES.values())                                   # TEST, EXTERNAL: never built
TERRAIN_FILES = ('terrain.npz', 'geodesic.npy')                                        # TERRAIN.json must list the SHA-256 of both
MAX_WORKERS = 61                                                                       # the limit of ProcessPoolExecutor on Windows
LOCK_NAME = 'S2_BUILD.lock'                        # in the run folder, outside the store root (the store hashes every file under s2\)
# the constant behind each field of the ``grid`` record of TERRAIN.json (``terrain.build_terrain`` writes the record), for the messages
_GRID_SOURCES = {'cells': 'GRID', 'cell_u': 'CELL', 'histogram_cells': 'GRID100', 'histogram_u': 'MAP_LIMIT // GRID100',
                 'min_count': 'MIN_COUNT'}
_ABSENT = object()                                 # "no such key" in a comparison of two records
MASKED_VIEWS = ('The M3 and M4 views at t0 (the masked inputs of the onsets) are not stored (D1). The input of the onset i of the S1 store '
                '(onsets/row[i], onsets/unit[i], onsets/bin[i]) is the view of the frame-clock row onsets/row[i] at its reference time t0: '
                'the S2 view v with views/src[v] == 0 and views/src_index[v] == onsets/row[i]. The open views are the event rows: '
                'views/src 1 = M3 at the first kill f, 2 = M4 at delta, views/src_index = the row in event_rows.')


class StopRequested(RuntimeError):
    """``stop_after_block`` was reached: the block was written but not marked, as after a power cut. A resume builds it again.

    정전을 흉내 낸 중단 신호이다. 이 블록은 쓰였으나 완료 표시가 없어 이어하기가 처음부터 다시 쓴다(저장소 형식 층의 공통 가정 (나)).
    """


class S2CheckError(RuntimeError):
    """A per-match check of the build failed: the reader rejects a match that S1 accepted, the record is another match, or a frame time
    differs from the one S1 stored. The message holds the S1 match row and the patch, never the match key.

    S1이 받아들인 경기의 보기가 S1 행과 어긋난 채 저장되는 일이 없도록 빌드를 멈추는 신호이다(S1–S2 판정 R13-r).
    """


class S2WorkerError(RuntimeError):
    """An unexpected error inside a worker whose text (or whose chain of causes) held the match key or the game id inside it. The key is
    replaced by <key>, the game id by <id>, and the S1 match row is given instead.

    오류 글에 경기를 이름으로 남기지 않는 규칙(reader 모듈의 공통 가정 (나))을 작업자 안의 예상 밖 오류에도 적용하는 신호이다.
    """


class RunLocked(RuntimeError):
    """Another S2 build holds the lock of the run folder, or a build that was cut (a power cut, a killed process) left it behind. Only one
    build may work on a run folder at a time. The message names the lock file and says to delete it if no build runs; the builder never
    deletes it itself. It holds no match key.

    한 실행 폴더에서는 빌드가 하나만 돌게 하는 잠금이 이미 있다는 신호이다. 비정상 종료가 남긴 잠금은 사람이 지우며 빌더는 스스로 지우지 않는다.
    """


# ---- the layout -----------------------------------------------------------------------------------------------------------------------
def x_layout() -> dict:
    """{'x/<key>': (shape after the row axis, dtype)} from ``tokens.VIEW_SPEC``, in the order of the dict that ``build_view`` returns.

    Read in the process that asks: ``VIEW_SPEC`` is a read-only mapping and does not travel through a pool.

    (BLD-2)의 키별 모양과 자료형을 모형의 보기 명세에서 꺼낸다.
    """
    return {f'x/{name}': (tuple(int(s) for s in shape), np.dtype(dtype)) for name, (shape, dtype) in tokens.VIEW_SPEC.items()}


def layout_arrays(n_views: int) -> dict:
    """{array name: (full shape, dtype)} of an S2 store with ``n_views`` views: the six ``views/`` columns, then every ``x/`` key.

    수식
    ----
    (BLD-2)  shape(views/c) = (N,),   shape(x/key) = (N, *shape_key),   bytes(arr) = ∏ shape(arr) · itemsize(arr)
             bytes_per_view = x_bytes_per_view + views_bytes_per_view
             x_bytes_per_view = Σ_key ∏ shape_key · itemsize_key,   views_bytes_per_view = Σ_c itemsize_c
             쓰기 전 공간 검사는 아직 없는 배열의 바이트 합 need = Σ_{arr 없음} bytes(arr)에 store_core의 (STO-2)를 쓴다.

      기호                | 뜻                                                | 단위      | 범위 | 코드 변수
      N                   | 보기의 수                                         | 개        | ≥ 1  | n_views
      c, key              | views의 열 이름, x의 키 이름                      | —         | —    | VIEWS_COLUMNS, VIEW_SPEC
      shape_key, itemsize | 키 하나의 모양(행 축 뒤), 원소 하나의 바이트      | —, 바이트 | —    | shape, dtype.itemsize
      bytes(arr)          | 배열 arr의 바이트                                 | 바이트    | ≥ 0  | _nbytes
      bytes_per_view      | 보기 하나의 저장 바이트(전체, x 키들, views 열들) | 바이트    | > 0  | bytes_per_view, x_bytes_per_view, views_bytes_per_view
      need                | 아직 할당하지 않은 배열의 바이트 합               | 바이트    | ≥ 0  | require_free_bytes의 인자

    해석
    ----
    식 (BLD-2)는 저장소의 모든 배열이 보기 수 N을 첫 축으로 갖고 키마다의 모양과 자료형이 모형의 보기 명세에서 오도록 정한 배치이다. 이름, 모양, 자료형은 빌드가 시작될 때 명세에서 만들고
    이 모듈에 따로 적어 두지 않으므로 키를 더하거나 모양을 바꾸면 배치 해시가 달라져 이어하기가 거부된다((BLD-3)). 공간 검사는 아직 없는 배열만 더하므로 이어하기는 이미 할당한 배열의
    바이트를 다시 요구하지 않는다(S1–S2 판정 R13-k).

    근거
    ----
    S1–S2 계획 Task 13(배열 이름은 views/열과 x/키), S1–S2 판정 R13-k(공간 확인)·R13-o(보기당 바이트), 저장소 설계 4.1절(배열마다 .npy 하나).
    """
    out = {f'views/{c}': ((n_views,), np.dtype(d)) for c, d in VIEWS_COLUMNS}
    for name, (shape, dtype) in x_layout().items():
        out[name] = ((n_views, *shape), dtype)
    return out


def _nbytes(shape, dtype) -> int:
    """(BLD-2)의 bytes(arr) = ∏ shape · itemsize이다."""
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


# ---- the view table -------------------------------------------------------------------------------------------------------------------
def _match_rows(matches) -> np.ndarray:
    """The match list as a 1-d int64 array. TypeError unless it is a list (tuple, array) of whole numbers (not bool, float or text);
    ValueError for an empty list or a row that is listed twice."""
    if isinstance(matches, (str, bytes)) or not hasattr(matches, '__iter__'):
        raise TypeError('matches must be a list of S1 match row numbers (ints)')
    rows = []
    for m in matches:
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
            raise TypeError(f'matches must be S1 match row numbers (ints), not {type(m).__name__}')
        rows.append(int(m))
    if not rows:
        raise ValueError('no matches to build')
    if len(set(rows)) != len(rows):
        raise ValueError(f'{len(rows) - len(set(rows))} of the {len(rows)} listed match rows repeat an earlier one: a match is listed once')
    return np.array(rows, dtype=np.int64)


def _check_range(rows: np.ndarray, n_s1: int) -> None:
    bad = (rows < 0) | (rows >= n_s1)
    if bad.any():
        raise ValueError(f'{int(bad.sum())} of the listed match rows are not rows of the S1 store (it holds {n_s1} matches, '
                         f'rows 0 .. {n_s1 - 1})')


def _build_table(s1: Store, rows: np.ndarray) -> tuple[dict, np.ndarray]:
    """(the views table, the number of views of every listed match) for the validated rows ``rows``.

    (BLD-1)의 보기 표와 경기별 보기 수 N_u를 만든다. 식은 view_table에 있다.
    """
    row_off = np.array(s1.array('matches/row_off'), dtype=np.int64)
    event_off = np.array(s1.array('matches/event_off'), dtype=np.int64)
    if len(row_off) != len(event_off):
        raise ValueError('the S1 offsets matches/row_off and matches/event_off differ in length')
    _check_range(rows, len(row_off) - 1)
    frame_k, frame_t0 = s1.array('frame_rows/k'), s1.array('frame_rows/t0')
    frame_match = s1.array('frame_rows/match')
    kind, event_ref = s1.array('event_rows/kind'), s1.array('event_rows/ref')
    event_k, event_t0 = s1.array('event_rows/frame_k'), s1.array('event_rows/t0')
    if row_off[-1] != len(frame_t0) or event_off[-1] != len(event_ref):
        raise ValueError('the S1 offsets do not end at the lengths of frame_rows and event_rows')
    f_lo, n_f = row_off[rows], row_off[rows + 1] - row_off[rows]
    e_lo, n_e = event_off[rows], event_off[rows + 1] - event_off[rows]
    count = n_f + n_e
    total = int(count.sum())
    owner = np.repeat(np.arange(len(rows)), count)                       # position in the list of the match each view belongs to
    local = np.arange(total, dtype=np.int64) - (np.cumsum(count) - count)[owner]        # the number of the view inside its match
    is_frame = local < n_f[owner]
    src_index = np.where(is_frame, f_lo[owner] + local, e_lo[owner] + local - n_f[owner])
    fr, ev = np.flatnonzero(is_frame), np.flatnonzero(~is_frame)
    fi, ei = src_index[fr], src_index[ev]
    src = np.zeros(total, dtype=np.int8)
    ref, t0 = np.zeros(total, dtype=np.int64), np.zeros(total, dtype=np.int64)
    k = np.zeros(total, dtype=np.int16)
    if fr.size:
        if not np.array_equal(np.asarray(frame_match)[fi], rows[owner[fr]]):
            raise ValueError('frame_rows/match disagrees with the S1 offsets matches/row_off')
        t0[fr] = ref[fr] = np.asarray(frame_t0)[fi]
        k[fr] = np.asarray(frame_k)[fi]
    if ev.size:
        src[ev] = np.asarray(kind)[ei]
        ref[ev], t0[ev], k[ev] = np.asarray(event_ref)[ei], np.asarray(event_t0)[ei], np.asarray(event_k)[ei]
        if not np.isin(src[ev], (SRC_M3, SRC_M4)).all() or not (t0[ev] < ref[ev]).all():
            raise ValueError('event_rows/kind is not 1 or 2, or an event row has a frame that is not before its reference time')
    if (k < 0).any():
        raise ValueError('a view has a negative frame number')
    table = {'src': src, 'src_index': narrow('views/src_index', src_index, np.int32),
             'match': narrow('views/match', rows[owner], np.int32), 'ref': ref, 't0': t0, 'frame_k': k}
    return table, count


def view_table(s1: Store, match_rows) -> dict[str, np.ndarray]:
    """The views table of the S1 matches ``match_rows`` (S1 match row numbers, in the order given): {column: array}, the columns and dtypes
    of the module text (``VIEWS_COLUMNS``), one row per view.

    Per match first its frame-clock rows in S1 order (src 0; ref = t0 = ``frame_rows/t0``, frame_k = ``frame_rows/k``), then its event rows
    in S1 order, that is by (unit, kind) (src = ``event_rows/kind``; ref, t0 and frame_k from ``event_rows``). Reads from S1 only the
    columns named in the module text. ValueError for a row that the S1 store does not hold.

    수식
    ----
    (BLD-1)  N_u = frame_u + event_u,   first_u = Σ_{u'<u} N_{u'},   N = Σ_u N_u
             경기 u의 보기 v (first_u ≤ v < first_{u+1})와 경기 안 번호 off = v − first_u에 대해
             off < frame_u이면 프레임 시계 보기: src = 0, src_index = frame_lo_u + off, ref = t₀ = frame_rows/t0[src_index],
             frame_k = frame_rows/k[src_index]
             그 밖이면 사건 보기: src_index = event_lo_u + (off − frame_u), src = event_rows/kind[src_index] ∈ {1, 2}, t₀ < ref이고
             ref, t₀, frame_k는 event_rows/ref, event_rows/t0, event_rows/frame_k의 같은 행에서 온다.

      기호                   | 뜻                                                        | 단위      | 범위          | 코드 변수
      u                      | 목록에서 경기의 위치(M은 목록의 경기 수)                  | —         | 1, …, M       | owner (0부터)
      frame_u, event_u       | 경기 u의 S1 프레임 시계 행의 수, 사건 행의 수             | 개        | ≥ 0           | n_f, n_e
      N_u, N                 | 경기 u의 보기 수, 전체 보기 수                            | 개        | ≥ 0           | count, total
      first_u                | 경기 u의 첫 보기 번호                                     | —         | 0, …, N       | np.cumsum(count) − count
      off                    | 경기 안의 보기 번호                                       | —         | 0, …, N_u − 1 | local
      frame_lo_u, event_lo_u | 경기 u의 첫 프레임 시계 행, 첫 사건 행의 S1 행 번호       | —         | ≥ 0           | f_lo, e_lo
      src, src_index         | 보기의 종류(0 프레임 시계, 1 M3, 2 M4), S1의 원천 행 번호 | —         | 0–2; ≥ 0      | src, src_index
      ref, t₀, frame_k       | 기준 시각, 값을 꺼내는 프레임의 시각, 그 프레임의 번호    | ms, ms, — | ≥ 0           | ref, t0, frame_k

    해석
    ----
    식 (BLD-1)은 경기마다 프레임 시계 보기를 먼저, 그다음 사건 보기를 S1에 저장된 (교전, 종류) 순서 그대로 늘어놓은 보기 표이다(S1–S2 판정 R13-d). 모든 경기와 블록의 행 구간이 보기를
    만들기 전에 정해지므로 보기가 놓이는 자리는 작업자 수, 블록 크기, 재시작과 상관없다(S1–S2 판정 S2-3, R13-c). 사건 보기의 프레임은 기준 시각 바로 앞의 정규 프레임이어서 t₀ < ref여야
    하며(설계서 11절 항목 8), 이를 어기거나 S1의 오프셋과 배열이 맞지 않으면 표를 만들지 않고 멈춘다.

    근거
    ----
    설계서 4절(프레임 시계 행과 사건 행마다 보기 하나), 11절 항목 8(열린 보기의 프레임), S1–S2 판정 R13-c·R13-d(고정된 행 위치, 보기 순서), S2-3.
    """
    return _build_table(s1, _match_rows(match_rows))[0]


# ---- one block, in a worker -----------------------------------------------------------------------------------------------------------
def _text(error) -> str:
    try:
        return str(error)
    except Exception:
        return ''


def _chain_text(error) -> str:
    """The text of an error and of every error in its chain of causes and contexts."""
    parts, seen, todo = [], set(), [error]
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        parts.append(_text(e))
        todo += [e.__cause__, e.__context__]
    return '\n'.join(parts)


def _key_parts(key: str) -> tuple:
    """What names a match in a text: the key (``<platform>_<game id>``) and the game id on its own, the number after the last underscore
    (or the whole key when it is only digits). A message that quotes a value of the file can hold the number without the platform.
    Longest first; empty for an empty key; one part where the key has no such number."""
    if not key:
        return ()
    number = re.search(r'(?:^|_)(\d+)$', key)
    return (key,) if number is None or number.group(1) == key else (key, number.group(1))


def _scrub(text: str, key: str) -> str:
    """``text`` with the key replaced by <key> and the game id inside it by <id>."""
    for part, label in zip(_key_parts(key), ('<key>', '<id>')):
        text = text.replace(part, label)
    return text


class _BlockBuilder:
    """Makes the views of the matches of one block; one per worker process (and one in the parent process for ``workers=1``)."""

    def __init__(self, terr, records):
        self.terr, self.records = terr, records

    def build(self, job):
        """``job`` = (block number, [(S1 match row, key, patch, refs, frame_ks), ...]) with the (ref, frame_k) pairs of the match as two
        int64 arrays. Returns (block number, {'x/<key>': array of n views}, t0 of every view as the record has it): plain numpy arrays and
        ints. An error stops the block; nothing is returned for it."""
        b, items = job
        spec = tokens.VIEW_SPEC                                       # read here: a mapping proxy does not travel through a pool
        n = sum(len(refs) for _, _, _, refs, _ in items)
        out = {name: np.zeros((n, *shape), dtype=dtype) for name, (shape, dtype) in spec.items()}
        t0_seen = np.zeros(n, dtype=np.int64)
        pos = 0
        for match, key, patch, refs, frame_ks in items:
            try:
                rec = self.records(key, patch)
                if rec.key != key or rec.patch != patch:
                    raise S2CheckError(f'S1 match #{match} (patch {patch}): the record that was read is another match than the one '
                                       'asked for')
                ctx = tokens.MatchContext(rec, self.terr)
                frame_ts = rec.frame_ts
                for ref, frame_k in zip(refs.tolist(), frame_ks.tolist()):
                    view = tokens.build_view(ctx, ref, frame_k)
                    if view.keys() != spec.keys():
                        raise ValueError(f'build_view returned the keys {sorted(view)}, not the keys of VIEW_SPEC')
                    for name, (shape, dtype) in spec.items():
                        v = view[name]
                        if v.dtype != dtype or v.shape != tuple(shape):
                            raise ValueError(f'view key {name}: {v.dtype} {v.shape}, VIEW_SPEC says {dtype} {tuple(shape)}')
                        out[name][pos] = v
                    t0_seen[pos] = frame_ts[frame_k]
                    pos += 1
            except (reader.MatchFormatError, reader.MatchReadError, reader.HoldoutRefused, S2CheckError):
                raise                                                 # key-free by construction
            except reader.MatchRejected as e:
                raise S2CheckError(f'S1 match #{match} (patch {patch}): the reader rejects it now ({e.code}) although S1 accepted it'
                                   ) from None
            except Exception as e:
                chain = _chain_text(e)
                if any(part in chain for part in _key_parts(key)):                # the key, or the game id on its own
                    raise S2WorkerError(f'{type(e).__name__}: {_scrub(_text(e), key)} (S1 match #{match}, patch {patch})') from None
                raise
        return b, {f'x/{name}': a for name, a in out.items()}, t0_seen


_BUILDER = None        # in a worker process: the builder of that process


def _init_worker(terrain_path: str, records) -> None:
    global _BUILDER
    _BUILDER = _BlockBuilder(terrain.Terrain(terrain_path), records)


def _run_job(job):
    return _BUILDER.build(job)


# ---- memory ---------------------------------------------------------------------------------------------------------------------------
class _Peaks:
    """Peaks of the private memory (Windows: the private bytes; elsewhere rss), sampled from a daemon thread every ``interval`` seconds, at
    the start, at the end and (called by the parent) after every block: ``tree`` the sum over this process and its children, the value
    that the 48 GiB gate compares (as ``run_s1._PeakMemory`` sums it), ``main`` this process alone and ``worker`` the largest single child
    process (0 when no child was ever seen). Without psutil there is no peak.

    수식
    ----
    (BLD-5)  tree = max_{표본} Σ_{p∈트리} used_p,   main = max_{표본} used_본체,   worker = max_{표본} max_{p∈자식} used_p
             트리 = 표본 시각의 본체와 그 모든 자손 프로세스, 자식 = 트리에서 본체를 뺀 것이다(자식이 없으면 worker = 0).

      기호   | 뜻                                                                | 단위   | 범위                  | 코드 변수
      p      | 프로세스(본체 또는 그 자손)                                       | —      | —                     | procs의 원소
      used_p | p가 표본 시각에 쓰는 메모리(Windows는 private bytes, 그 밖은 rss) | 바이트 | ≥ 0                   | used
      tree   | 본체와 자손의 used 합의 표본 최댓값(48 GiB 관문이 비교하는 값)    | 바이트 | ≥ 0                   | tree
      main   | 본체 하나의 used의 표본 최댓값                                    | 바이트 | ≥ 0                   | main
      worker | 자손 하나의 used의 표본 최댓값                                    | 바이트 | ≥ 0 (자손이 없으면 0) | worker

    해석
    ----
    식 (BLD-5)는 빌드 동안 프로세스 트리가 쓴 메모리의 표본 최댓값이고, 합은 본체와 작업자가 따로 갖는 메모리를 모두 더하므로 작업자가 늘면 늘어나는 비용을 포함한다(S1–S2 판정 R13-f 개정).
    이 값은 측정 구간에서 프로세스별 메모리를 순차 조회한 합의 최댓값이다. 조회에 실패한 프로세스는 합에서 제외되며, 표본 사이의 순간 피크를 놓칠 수 있다. 동시점 프로세스 트리 메모리의 참 최댓값에 대한 엄밀한 상한이나 하한은 보장하지 않는다. psutil이 없으면 재지 못하고 기록에 값이 없다.

    근거
    ----
    S1–S2 판정 R13-f 개정(프로세스 트리 전체의 합이 관문이 비교하는 값), S1–S2 계획 전역 제약(최대 private 메모리), 설계서 9절(S2에서 잰다).
    """

    def __init__(self, interval: float = 2.0):
        self.interval, self.samples = interval, 0
        self.tree = self.main = self.worker = 0
        self._lock, self._stop, self._thread = threading.Lock(), threading.Event(), None

    def sample(self) -> None:
        if psutil is None:
            return
        try:
            me = psutil.Process()
            procs = [me, *me.children(recursive=True)]
        except psutil.Error:
            return
        tree = main = worker = 0
        for p in procs:
            try:
                mi = p.memory_info()
            except psutil.Error:
                continue
            used = getattr(mi, 'private', None) or mi.rss
            tree += used
            if p.pid == me.pid:
                main = used
            else:
                worker = max(worker, used)
        with self._lock:
            self.samples += 1
            self.tree, self.main, self.worker = max(self.tree, tree), max(self.main, main), max(self.worker, worker)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        if psutil is not None:
            self.sample()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self.sample()
        return False


def _gib(value: int):
    """Bytes as GiB rounded to 0.001; None when psutil is missing or nothing was measured (a zero).

    (BLD-5)의 바이트를 기록용 GiB로 바꾼다.
    """
    return None if psutil is None or not value else round(value / 2 ** 30, 3)


# ---- checks before the build ----------------------------------------------------------------------------------------------------------
def _check_workers(workers) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f'workers must be an int of at least 1, got {workers!r}')
    if workers > MAX_WORKERS:
        raise ValueError(f'workers must be at most {MAX_WORKERS} (the limit of ProcessPoolExecutor on Windows), got {workers}')


def _check_stop(stop_after_block) -> None:
    if stop_after_block is not None and (isinstance(stop_after_block, bool) or not isinstance(stop_after_block, int)
                                         or stop_after_block < 0):
        raise ValueError(f'stop_after_block must be None or an int of at least 0, got {stop_after_block!r}')


def _check_info(info) -> dict:
    """``info`` as a dict (empty for None): TypeError unless it is a dict that JSON can write; it goes into RUN.json as ``args.info``."""
    if info is None:
        return {}
    if not isinstance(info, dict):
        raise TypeError(f'info must be a dict, not {type(info).__name__}')
    try:
        json.dumps(info)
    except (TypeError, ValueError) as e:
        raise TypeError(f'info must hold only numbers, text, lists and dicts: {e}') from None
    return dict(info)


def _s1_folder(s1_run) -> Path:
    """The S1 run folder, given the run folder or its ``s1`` store folder."""
    root = Path(s1_run)
    return root.parent if (root / 'STORE.json').is_file() else root


def _check_s1_matches(s1: Store, rows: np.ndarray) -> tuple[list, list]:
    """(keys, patches) of the listed S1 matches. ValueError for a row the store does not hold and for a holdout match (a TEST or EXTERNAL
    role, a holdout patch): S2 builds the matches of the S1 store that are FIT, STOP, DEV, CAL or AUDIT, and nothing else."""
    row_off = s1.array('matches/row_off')
    _check_range(rows, len(row_off) - 1)
    roles, patches = np.asarray(s1.array('matches/role'))[rows], np.asarray(s1.array('matches/patch'))[rows]
    if (roles < 0).any() or (roles >= len(C.ROLES)).any() or (patches < 0).any() or (patches >= len(C.PATCHES)).any():
        raise ValueError('the S1 store holds a role or patch number that config does not know')
    names = [C.ROLES[r] for r in roles.tolist()]
    refused = sorted({n for n in names if n in REFUSED_ROLES})
    if refused:
        raise ValueError(f'the listed matches include a holdout role {refused}: S2 never builds the holdout matches')
    held = sorted({C.PATCHES[p] for p in patches.tolist() if C.PATCHES[p] in reader.HOLDOUT_PATCHES})
    if held:
        raise ValueError(f'the listed matches include the holdout patch(es) {held}: S2 never builds the holdout matches')
    keys = [k.decode('ascii') for k in np.asarray(s1.array('matches/key'))[rows].tolist()]
    return keys, [C.PATCHES[p] for p in patches.tolist()]


def terrain_parameters() -> dict:
    """What a terrain must have been built with, in the plain form that TERRAIN.json holds it (``terrain.build_terrain`` writes both): the
    ``grid`` record (``cells``, ``cell_u``, ``histogram_cells``, ``histogram_u``, ``min_count``) and the ``zone_rules`` (a list where the
    rules have a tuple), taken from the constants of the ``terrain`` module (GRID, CELL, GRID100, MAP_LIMIT, MIN_COUNT, ZONE_RULES).

    빌드가 받아들이는 지형의 매개변수(격자, 보행 문턱, 구역 규칙)를 TERRAIN.json과 같은 꼴로 내놓는다. 보행 문턱은 S2 관문의 결정 S2G-2가 정했다.
    """
    return {'grid': {'cells': terrain.GRID, 'cell_u': int(terrain.CELL), 'histogram_cells': terrain.GRID100,
                     'histogram_u': int(terrain.MAP_LIMIT // terrain.GRID100), 'min_count': terrain.MIN_COUNT},
            'zone_rules': {k: (list(v) if isinstance(v, tuple) else v) for k, v in terrain.ZONE_RULES.items()}}


def terrain_parameter_problems(manifest: dict) -> list:
    """The differences between a parsed TERRAIN.json and ``terrain_parameters()``, as sentences; an empty list when it was built with
    the parameters of this code.

    A terrain whose files match its own TERRAIN.json passes ``Terrain.verify`` whatever it was built with, so the parameters are checked
    apart: a terrain built with another walkable threshold (the first smoke run used ``min_count`` 5, the gate S2G-2 made it 150), grid
    or zone rules would give other geodesic distances, wall fractions and zones to every view. A missing record counts as a difference
    (nothing proves the parameters). The sentences name fields and numbers only.

    TERRAIN.json의 기록이 이 코드의 매개변수와 다른 곳을 문장으로 알린다. 파일이 자기 기록과 맞아도 다른 매개변수로 만든 지형이면 모든 보기의 거리와 구역이 달라지므로 기록을 따로 맞대 본다.
    """
    want, out = terrain_parameters(), []
    grid = manifest.get('grid')
    if not isinstance(grid, dict):
        out.append('TERRAIN.json has no grid record')
    else:
        for key, value in want['grid'].items():
            if key not in grid or grid[key] != value:
                got = 'missing' if key not in grid else repr(grid[key])
                out.append(f'grid.{key} is {got} in TERRAIN.json, terrain.{_GRID_SOURCES[key]} is {value!r}')
    rules = manifest.get('zone_rules')
    if not isinstance(rules, dict):
        out.append('TERRAIN.json has no zone_rules record')
    else:
        for key in sorted(set(rules) | set(want['zone_rules'])):
            have, now = rules.get(key, _ABSENT), want['zone_rules'].get(key, _ABSENT)
            if have != now:
                out.append(f'zone_rules.{key} is {"missing" if have is _ABSENT else repr(have)} in TERRAIN.json, terrain.ZONE_RULES has '
                           f'{"no such rule" if now is _ABSENT else repr(now)}')
    return out


def terrain_problems(folder) -> list:
    """Why the terrain folder cannot be used as it is, as sentences (an empty list: TERRAIN.json is there, parses, and records the
    parameters of this code). Reads TERRAIN.json only: the two data files are not opened.

    TERRAIN.json 하나만 읽어 지형 폴더를 쓸 수 없는 이유를 문장으로 모은다.
    """
    folder = Path(folder)
    path = folder / 'TERRAIN.json'
    if not path.is_file():
        return ['it has no TERRAIN.json (a terrain without provenance)']
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        return [f'TERRAIN.json cannot be read ({type(e).__name__})']
    if not isinstance(manifest, dict):
        return ['TERRAIN.json is not a record (a JSON object)']
    return terrain_parameter_problems(manifest)


def terrain_refusal(folder):
    """The sentence that refuses a terrain folder (``terrain_problems`` and what to do), or None when the folder may be used. The one text
    of ``build_s2`` (a ValueError) and of the command line.

    그 이유와 할 일을 한 문장으로 만들어, build_s2와 명령줄이 같은 글로 거부하게 한다.
    """
    problems = terrain_problems(folder)
    if not problems:
        return None
    folder = Path(folder)
    where = os.path.join(folder.parent.name, folder.name) if folder.parent.name else folder.name          # <run>\terrain says more than terrain
    return (f'the terrain folder {where} is refused: {"; ".join(problems)}. Build the terrain again with the current code '
            '(python -m c_ept.run_s2 ... --build-terrain)')


def _open_terrain(path) -> tuple:
    """(Terrain, {file name: SHA-256} of terrain.npz, geodesic.npy and TERRAIN.json). Refused (ValueError), before a data file is opened:
    a folder without TERRAIN.json (no provenance) and one built with other parameters than this code uses (``terrain_refusal``). Then
    one whose files differ from TERRAIN.json (``Terrain.verify``) and one whose TERRAIN.json does not list both data files."""
    path = Path(path)
    refusal = terrain_refusal(path)
    if refusal is not None:
        raise ValueError(refusal)
    terr = terrain.Terrain(path)
    listed = terr.verify()
    missing = [n for n in TERRAIN_FILES if n not in listed]
    if missing:
        raise ValueError(f'TERRAIN.json of {path.name} does not list the SHA-256 of {missing}')
    hashes = {**listed, 'TERRAIN.json': store_core.sha(path / 'TERRAIN.json')}
    return terr, dict(sorted(hashes.items()))


def _check_resume(saved: dict, block_matches, plan_sha: str, n_matches: int, terrain_sha: str, layout_sha: str) -> None:
    """ValueError ('resume refused') unless the saved plan is the one of this call; nothing has been written when it is raised.

    (BLD-3)의 이어하기 조건을 확인한다.
    """
    if saved.get('block_matches') != block_matches:
        raise ValueError(f'resume refused: the saved run used block_matches={saved.get("block_matches")}, this one {block_matches}')
    if saved.get('plan_sha256') != plan_sha or saved.get('n_matches') != n_matches:
        raise ValueError('resume refused: the match list or the S1 store differs from the saved plan (plan_sha256)')
    if saved.get('terrain_sha256') != terrain_sha:
        raise ValueError('resume refused: the terrain differs from the saved plan (terrain_sha256)')
    if saved.get('layout_sha256') != layout_sha:
        raise ValueError('resume refused: the view layout differs from the saved plan (layout_sha256: tokens.VIEW_SPEC changed)')


def _warn(message: str, sink: list) -> None:
    """One ``WARNING:`` line on stderr; the text is also appended to ``sink`` (the call record, and so RUN.json)."""
    print(f'WARNING: {message}', file=sys.stderr, flush=True)
    sink.append(message)


def _short(commit) -> str:
    return commit[:12] if commit else 'unknown'


def _recorded_code_state(run: Path):
    """(code_commit, code_dirty) that RUN.json holds from the first call; None while the run has no RUN.json."""
    path = run / 'RUN.json'
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding='utf-8'))
    return manifest.get('code_commit'), manifest.get('code_dirty')


def require_run_record(run, had_store: bool = True) -> None:
    """ValueError when the run folder has an S2 store (``had_store``) and no RUN.json.

    RUN.json is written before STORE.json exists, so a store without it means that RUN.json was deleted or that the run folder was copied
    without it. Writing a new one now would put this call's commit, GPU and plan into the record of rows that an earlier call built.

    저장소는 있는데 RUN.json이 없는 폴더를 거부한다. 새로 쓴 RUN.json이 앞선 호출이 만든 행에 이번 호출의 코드와 계획을 적어 출처 기록을 틀리게 하기 때문이다(S1–S2 판정 S2-5).
    """
    run = Path(run)
    if had_store and not (run / 'RUN.json').is_file():
        raise ValueError(f'{run.name} has an S2 store but no RUN.json (deleted, or the run folder was copied without it): RUN.json holds '
                         'the commit, the GPU and the plan of the first call, and a new one written now would claim this call\'s for rows '
                         'that an earlier call built. Restore RUN.json from the original run folder, or build into a new run folder')


def _ensure_run_json(run: Path, run_info: dict) -> dict:
    """RUN.json of the run as a dict; ``runlog.write_manifest`` writes it from ``run_info`` first when there is none. Only a run without
    a store reaches the write: ``require_run_record`` has refused a store without RUN.json before."""
    path = run / 'RUN.json'
    if not path.exists():
        run.mkdir(parents=True, exist_ok=True)
        runlog.write_manifest(run, run_info)
    return json.loads(path.read_text(encoding='utf-8'))


def _lock_message(path: Path) -> str:
    """The text of ``RunLocked``: whose lock it is as far as the file says, and what to do about it. Nothing is changed."""
    try:
        held = json.loads(path.read_text(encoding='utf-8'))
        pid, created = held.get('pid'), held.get('created')
    except (OSError, ValueError, AttributeError):
        pid = created = None
    who = f'made {created} by pid {pid}' if pid is not None else 'its content cannot be read'
    hint = ''
    if isinstance(pid, int) and psutil is not None:
        try:
            hint = ', a process with that pid is running now' if psutil.pid_exists(pid) else ', no process with that pid is running now'
        except Exception:                                          # a pid that the system cannot even ask about says nothing
            hint = ''
    return (f'another S2 build holds the lock {path} ({who}{hint}). Only one build may work on a run folder at a time. If no S2 build is '
            f'running on {path.parent.name} (a power cut or a killed process leaves the lock behind), delete that file yourself and resume '
            'the run; the builder never deletes it.')


def _acquire_lock(run: Path) -> Path:
    """Create ``<run>/S2_BUILD.lock`` exclusively (``O_EXCL``: the system lets one process win) and return its path; RunLocked when the
    file is there. The run folder is made when needed. The file holds the pid and the time and nothing that names a match."""
    run.mkdir(parents=True, exist_ok=True)
    path = run / LOCK_NAME
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    except FileExistsError:
        raise RunLocked(_lock_message(path)) from None
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps({'pid': os.getpid(), 'created': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}) + '\n')
    except BaseException:
        path.unlink(missing_ok=True)                               # a lock without content would only block the next call
        raise
    return path


def _release_lock(path: Path) -> None:
    """Remove the lock this call made. A failure is said on stderr and never hides the error that ended the build."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f'WARNING: the lock {path} could not be removed ({type(e).__name__}); delete it before the next call on this run',
              file=sys.stderr, flush=True)


def _allocate(s: Store, name: str, shape, dtype, sink: list) -> None:
    """``Store.allocate``, with one repair: an array file that STORE.json does not list and that cannot be opened is removed first, so
    that ``allocate`` creates it again.

    A cut during the creation of an array can leave 0 bytes or half a header, and the entry is saved only after the file exists, so such
    a file is unlisted. Nothing can have been written into it: a block is marked only after every array is listed. A file that opens is
    left to ``allocate`` (it checks shape and dtype), and a LISTED file that cannot be opened is damage to an array that may hold the rows
    of marked blocks: it raises, as before. The repair is said in one ``WARNING:`` line (and so in the call record).
    """
    path = s.path(name)
    if path.exists() and name not in s.manifest['arrays']:
        try:
            probe = np.load(path, mmap_mode='r')
        except Exception as e:                                     # EOFError, ValueError (half a header), OSError (too short for its header)
            path.unlink()
            _warn(f'{name}.npy was not listed in STORE.json and could not be opened ({type(e).__name__}): a cut during its creation left '
                  'it unreadable; it was created again', sink)
        else:
            del probe
    s.allocate(name, shape, dtype)


def _check_manifest_arrays(s: Store, layout: dict) -> None:
    """After the arrays are allocated STORE.json must list exactly the arrays of ``layout``, in plain form. ``Store.allocate`` opens an
    array file that exists after checking its shape and dtype, and does not list it: a power cut between the creation of a file and the
    save of its entry leaves such a file, and the entry is added here (no block has been marked yet, so nothing depends on it). An entry
    that is not in the layout, or whose shape or dtype differs, is a ValueError: nothing is built."""
    want = {name: {'shape': list(shape), 'dtype': dtype.str} for name, (shape, dtype) in layout.items()}
    have = s.manifest['arrays']
    odd = sorted(set(have) - set(want)) + sorted(n for n in set(have) & set(want) if have[n] != want[n])
    if odd:
        raise ValueError(f'STORE.json lists arrays that do not match the layout: {odd[:10]}')
    lost = [n for n in want if n not in have]
    for n in lost:
        have[n] = want[n]
    if lost:
        s.save()
    if have != want:
        raise ValueError('STORE.json does not list the arrays of the layout')


def _result_record(*, calls: list, finalize_s, complete: bool, recovered: bool, n_blocks: int, block_matches: int, n_matches: int,
                   n_views: int, by_src, files: list, code_end: dict) -> dict:
    """The ``s2_result`` of RUN.json from the call records of the stage (``calls``): the peaks are the maxima over all calls, the times and
    the match counts their sums, so a resumed build reports the whole build. ``finalize_s`` is the time of the ``finalize`` of the call
    that made the record, None when that call did not finalize (a repaired record). ``code_end`` is ``code_state_at_end``: the code state
    of the call that ended the build; a call that only repairs the record built nothing, so it passes that of the last call on record.

    ``files`` are the stored files of the store, ``Store.stored_files()``: every array, and neither STORE.json nor a ``*.tmp`` leftover,
    as in the store hashes (S2-3). So ``store_files`` and ``store_bytes`` are a function of the plan alone: the same for every build of
    it, whatever its timings, workers, block size or restarts, and for a repaired record. STORE.json is left out because it records the
    plan and the times and memory peaks of every call, so its size differs from one build to the next by a few bytes (the numbers print
    with a different number of digits); counting it made a repaired record differ from the record of a fresh build by one byte.

    수식
    ----
    (BLD-6)  peak = max_{call} peak_call (tree, main, worker),   times = Σ_{call},   matches_built = Σ_{call} matches_built_call
             wall_s = round(Σ_{call} allocate_s + Σ_{call} views_s + (finalize_s 또는 0), 1)
             seconds_per_match = round(Σ_{call} views_s / matches_built, 4)   (단 matches_built = 0이면 None)
             기록하는 메모리 값은 _gib(peak)이며, psutil이 없거나 입력 바이트 peak가 0이면 None이고, 그 밖에는 round(peak / 2^30, 3) GiB이다.
             bytes_per_view = x_bytes_per_view + views_bytes_per_view
             store_files = |stored_files|,   store_bytes = Σ_{file∈stored_files} 파일 크기

      기호                | 뜻                                                    | 단위   | 범위 | 코드 변수
      call                | 이 단계에 기록된 호출(이어하기마다 하나)              | —      | —    | calls의 원소 c
      peak_call           | 그 호출의 메모리 최댓값 세 가지((BLD-5))              | 바이트 | ≥ 0  | peak_private_bytes
      allocate_s, views_s | 그 호출의 할당 시간, 보기 만들기 시간                 | s      | ≥ 0  | allocate_s, views_s
      finalize_s          | 최종화를 한 호출의 최종화 시간(고쳐 만든 기록은 없음) | s      | ≥ 0  | finalize_s
      matches_built       | 이 호출에서 만든 경기 수(호출들의 합이 전체)          | 경기   | ≥ 0  | matches_built
      bytes_per_view      | 보기 하나의 저장 바이트((BLD-2)의 세 값)              | 바이트 | > 0  | x_bytes + VIEWS_BYTES_PER_VIEW
      stored_files        | STORE.json과 *.tmp를 뺀 저장소 파일                   | —      | —    | files

    해석
    ----
    식 (BLD-6)은 호출마다 적은 기록을 한 빌드의 기록으로 합치는 규칙이다. 메모리 최댓값은 모든 호출의 최댓값이고 시간과 경기 수는 합이어서, 이어한 빌드는 마지막 호출이 아니라 빌드 전체를 보고한다.
    저장소의 파일 수와 바이트는 STORE.json을 뺀 배열 파일만 세므로 같은 계획의 모든 빌드에서 같고 시간, 작업자 수, 블록 크기, 재시작과 상관없다(S1–S2 판정 S2-3).

    근거
    ----
    S1–S2 판정 S2-3(STORE.json을 뺀 개수), S2-5(출처 기록)와 R13-f 개정·R13-o(보기당 바이트).
    """
    def peak(key):
        return max((c.get('peak_private_bytes', {}).get(key, 0) for c in calls), default=0)
    allocate_s, views_s = sum(c.get('allocate_s', 0.0) for c in calls), sum(c.get('views_s', 0.0) for c in calls)
    built, last, x_bytes = sum(c.get('matches_built', 0) for c in calls), (calls[-1] if calls else {}), tokens.bytes_per_view()
    per_call = [{'workers': c.get('workers'), 'wall_s': round(c.get('allocate_s', 0.0) + c.get('views_s', 0.0), 1),
                 'allocate_s': round(c.get('allocate_s', 0.0), 1), 'views_s': round(c.get('views_s', 0.0), 1),
                 'matches_built': c.get('matches_built', 0), 'blocks_built': c.get('blocks_built', 0),
                 'memory_samples': c.get('memory_samples', 0),
                 'peak_private_gib': _gib(c.get('peak_private_bytes', {}).get('tree', 0)),
                 'peak_private_main_gib': _gib(c.get('peak_private_bytes', {}).get('main', 0)),
                 'peak_private_worker_gib': _gib(c.get('peak_private_bytes', {}).get('worker', 0)),
                 'code_commit': c.get('code_commit'), 'code_dirty': c.get('code_dirty'), 'warnings': list(c.get('warnings', []))}
                for c in calls]
    return {
        'complete': complete, 'wall_s': round(allocate_s + views_s + (finalize_s or 0.0), 1),
        'stage_s': {'allocate': round(allocate_s, 1), 'views': round(views_s, 1),
                    'finalize': None if finalize_s is None else round(finalize_s, 1)},
        'peak_private_gib': _gib(peak('tree')), 'peak_private_main_gib': _gib(peak('main')),
        'peak_private_worker_gib': _gib(peak('worker')),
        'memory_samples': sum(c.get('memory_samples', 0) for c in calls),
        'workers': last.get('workers'), 'block_matches': block_matches, 'n_blocks': n_blocks,
        'blocks_resumed': n_blocks - last.get('blocks_built', 0),
        'matches': n_matches, 'matches_built': built, 'seconds_per_match': round(views_s / built, 4) if built else None,
        'views': n_views, 'views_by_src': {name: int(by_src[i]) for i, name in enumerate(SRC_NAMES)},
        'x_bytes_per_view': x_bytes, 'views_bytes_per_view': VIEWS_BYTES_PER_VIEW, 'bytes_per_view': x_bytes + VIEWS_BYTES_PER_VIEW,
        'store_files': len(files), 'store_bytes': sum(p.stat().st_size for p in files),
        'code_state_at_end': code_end,                           # the top-level code_commit is the first call's: they differ after a resume
        'calls': len(calls), 'per_call': per_call, 'warnings': [w for c in per_call for w in c['warnings']], 'recovered': recovered}


# ---- the build ------------------------------------------------------------------------------------------------------------------------
def build_s2(s1_run, terrain_path, matches, run, workers: int = 8, block_matches: int = 250, stop_after_block: int | None = None,
             records=reader.read_match, progress=None, info: dict | None = None) -> Path:
    r"""The S2 store of the S1 matches ``matches`` in the run folder ``run``; returns the store root ``<run>\s2``.

    ``s1_run`` is the S1 run folder (or its ``s1`` store folder; read only, every file is checked against its STORE.json),
    ``terrain_path`` a terrain folder with TERRAIN.json (read only; ``Terrain.verify`` must pass), ``matches`` the S1 match row numbers in
    the order wanted (a row once; the order is the order of the views). ``records(key, patch)`` returns the ``MatchRecord``
    (``reader.read_match``, default root); tests replace it. ``progress(blocks_done, n_blocks)`` is called after every block. ``info`` is a
    JSON-ready dict kept under ``args.info`` of RUN.json (it cannot overwrite a computed field). ``workers`` is 1 to ``MAX_WORKERS`` (61,
    the limit of ``ProcessPoolExecutor`` on Windows); 1 builds in this process.

    Order of the refusals, none of which writes anything: arguments that need no file (``workers``, ``block_matches``, ``stop_after_block``,
    ``info``, the match list); the E6 check (``run_s1.StrataRunning`` while a model server runs: nothing is stopped); a ``run`` that is or
    lies inside the S1 run; the S1 store (complete, hashes right); matches the store does not hold and holdout matches; the terrain (no
    TERRAIN.json, or built with other parameters than this code uses: ``terrain_parameter_problems``; files that differ from it); a store
    without views; too little free disk space (``store_core.require_free_bytes`` for every array not yet allocated, 10 % more than
    needed). Then the lock (one build per run folder: ``RunLocked`` while ``<run>\S2_BUILD.lock`` exists; see below) and, under it, the
    refusals that read the run folder: a saved plan that differs (``resume refused``: block size, match list, terrain, layout) and a
    store without RUN.json (``require_run_record``). Then RUN.json (unless the folder has one), the stage, the arrays (STORE.json must
    then list exactly the layout; an unlisted array file that cannot be opened is created again), the blocks, ``finalize`` (every block
    must be marked) and ``amend_manifest(run, {'s2_result': ...})``. A complete store is returned as it is; when its RUN.json has no
    ``s2_result`` (a power cut after ``finalize``) the record is made from the stage first. Masked M3 and M4 views at t0 are not stored:
    see the module text (D1).

    One build per run folder. The lock is an ``O_EXCL`` file ``<run>\S2_BUILD.lock`` (pid and time inside, outside the store root), held
    for the whole call and removed at its end, whether it finishes, stops or fails. A second build or resume on the run refuses while it
    exists. A power cut or a killed process leaves it behind: the message (``RunLocked``) then says which file to delete once no build
    runs. It is never deleted automatically.

    E6 again between blocks. After every finished block but the last the E6 check is made again: the author keeps Strata for coding, and
    a model server that starts during the build disturbs its timing and memory record. If one has appeared, the build stops cleanly
    (``run_s1.StrataRunning``, the message names the block and says to stop the server and resume): the blocks done are marked and
    counted, nothing partial is written, the lock is removed, and the run resumes to the bytes of an uninterrupted build.

    ``stop_after_block=k`` raises ``StopRequested`` after the rows of block k are written and before its mark (a power cut). Resuming needs
    the same list, ``block_matches``, S1 store, terrain and ``tokens.VIEW_SPEC``; the worker count may change. The peaks, times and match
    counts of ``s2_result`` cover all calls. A terrain from another S1 run and a resume on other code print one ``WARNING:`` line each.

    수식
    ----
    (BLD-3)  SHA(x) = SHA-256(x의 정렬된 간결 JSON, UTF-8)   (소문자 16진수)
             files_sha = SHA(S1 STORE.json의 파일 표),   plan_sha = SHA({s1_files_sha256: files_sha, matches: 경기 목록})
             terrain_sha = SHA({파일 이름: SHA-256}),   layout_sha = SHA({배열 이름: [모양, dtype]})
             이어하기의 계획 일치 검사 통과 ⇔ 저장된 (block_matches, plan_sha, n_matches, terrain_sha, layout_sha)가 이번 호출의 값과 모두 같다.
             이 검사는 이어하기의 필요 조건 하나이며, RUN.json 등 다른 조건도 이어하기에 적용된다.

      기호        | 뜻                                                                     | 단위 | 범위        | 코드 변수
      SHA(·)      | JSON 값의 SHA-256                                                      | —    | 64자 문자열 | runlog.sha256_json
      files_sha   | 소비하는 S1 저장소의 파일 요약 표의 해시                               | —    | 64자 문자열 | files_sha
      plan_sha    | 경기 목록과 S1 저장소를 묶은 계획 해시                                 | —    | 64자 문자열 | plan_sha
      terrain_sha | 지형 세 파일(terrain.npz, geodesic.npy, TERRAIN.json)의 요약 표의 해시 | —    | 64자 문자열 | terrain_sha
      layout_sha  | 배열 배치((BLD-2))의 해시                                              | —    | 64자 문자열 | layout_sha
      n_matches   | 목록의 경기 수                                                         | 경기 | ≥ 1         | len(rows)

    해석
    ----
    식 (BLD-3)은 이어하기가 같은 일을 이어 가는지 가르는 계획 일치 기준이다: 같은 S1 저장소와 경기 목록(순서 포함), 같은 지형, 같은 보기 배치, 같은 블록 크기일 때만 완료 표시가 같은 행을 가리킨다(저장소
    형식 층의 공통 가정 (다); S1–S2 판정 S2-2). 하나라도 다르면 저장소와 RUN.json에 쓰기 전에 거부하므로 다른 계획의 행 위에 이어 쓰는 일이 없다(S1–S2 판정 R13-b). 작업자 수는 계획에 들지 않아
    이어하는 호출에서 바꿀 수 있고 저장되는 바이트는 달라지지 않는다(S1–S2 판정 S2-3).

    근거
    ----
    S1–S2 판정 S2-2·R13-b(저장된 계획, 쓰기 전 거부), R13-n(계획 해시), S2-3(작업자 수와 상관없는 해시), S2-5(출처 기록), 저장소 설계 4.6절(입력 해시가 다르면 이어하기 거부).
    """
    from c_ept.workflows.run_s1 import StrataRunning, load_s1, require_no_strata          # here, not at module level: a spawned worker imports us (R13-a)
    rows = _match_rows(matches)
    _check_workers(workers)
    plan = store_core.block_plan(np.arange(len(rows)), block_matches)
    _check_stop(stop_after_block)
    info = _check_info(info)
    require_no_strata()                                          # E6: before the store or a terrain file is touched
    run = Path(run)
    s1_folder = _s1_folder(s1_run)
    if run.resolve().is_relative_to(s1_folder.resolve()):
        raise ValueError(f'{run} is the S1 run folder or lies inside it: an S1 run is read only, give the folder of the S2 run')
    s1 = load_s1(s1_run)
    terr = lock = None
    try:
        keys, patches = _check_s1_matches(s1, rows)
        terr, terrain_hashes = _open_terrain(terrain_path)
        table, count = _build_table(s1, rows)
        n_views = len(table['src'])
        if not n_views:
            raise ValueError('the listed matches have no views (no frame-clock row and no event row)')
        by_src = np.bincount(table['src'], minlength=len(SRC_NAMES))
        # 식 (BLD-3)의 해시: 이어하기가 같은 계획인지 가르는 값(계획, 지형, 배치)을 만든다.
        files_sha = runlog.sha256_json(s1.manifest['files'])
        plan_sha = runlog.sha256_json({'s1_files_sha256': files_sha, 'matches': rows.tolist()})
        terrain_sha = runlog.sha256_json(terrain_hashes)
        layout = layout_arrays(n_views)
        layout_sha = runlog.sha256_json({name: [list(shape), dtype.str] for name, (shape, dtype) in layout.items()})
        built_from_this_s1 = terr.manifest.get('s1_files_sha256') == files_sha
        run_info = {
            'args': {'workers': workers, 'block_matches': block_matches, 'n_matches': len(rows), 'plan_sha256': plan_sha, 'info': info},
            's1': {'run': s1_folder.name, 'files_sha256': files_sha},
            'terrain': {'sha256': terrain_hashes, 's1_run': terr.manifest.get('s1_run'),
                        's1_files_sha256': terr.manifest.get('s1_files_sha256'), 'built_from_this_s1': built_from_this_s1,
                        'grid': dict(terr.manifest['grid'])},         # checked by _open_terrain: it is the grid record of this code
            'masked_views': MASKED_VIEWS}
        root = run / STORE_DIR
        # 식 (BLD-2)의 공간 검사: 아직 없는 배열의 바이트 합에 여유 배수를 곱한 만큼의 빈 공간이 있어야 한다.
        store_core.require_free_bytes(root, sum(_nbytes(shape, dtype) for name, (shape, dtype) in layout.items()
                                                if not (root / f'{name}.npy').exists()))
        lock = _acquire_lock(run)                                # from here on the run folder is this call's alone (RunLocked otherwise)
        s = Store(root)                                          # read under the lock: another call cannot change what it reads
        saved = s.manifest['stages'].get('views')
        had_store = s.file.exists()
        if saved is not None:
            _check_resume(saved, block_matches, plan_sha, len(rows), terrain_sha, layout_sha)
        require_run_record(run, had_store)
        if saved is not None:
            if s.manifest.get('complete'):
                if 's2_result' not in _ensure_run_json(run, run_info):          # a power cut between finalize and the run record
                    last = (saved.get('calls') or [{}])[-1]       # the code that built the store is the last call's, not the repairing call's
                    runlog.amend_manifest(run, {'s2_result': _result_record(
                        calls=saved.get('calls', []), finalize_s=None, complete=True, recovered=True, n_blocks=len(plan),
                        block_matches=block_matches, n_matches=len(rows), n_views=n_views, by_src=by_src,
                        files=s.stored_files(),
                        code_end={'code_commit': last.get('code_commit'), 'code_dirty': last.get('code_dirty')})})
                return root
        call_warnings, now = [], runlog.code_state()
        if not built_from_this_s1:
            _warn('the terrain was built from another S1 run than the one consumed here (TERRAIN.json: s1_files_sha256 differs); '
                  'RUN.json keeps terrain.built_from_this_s1 = false', call_warnings)
        recorded = _recorded_code_state(run)
        if recorded is not None and recorded != (now['code_commit'], now['code_dirty']):
            _warn(f'resuming on other code: the first call ran on commit {_short(recorded[0])} (dirty: {recorded[1]}), this one on '
                  f'{_short(now["code_commit"])} (dirty: {now["code_dirty"]}); both are recorded', call_warnings)
        _ensure_run_json(run, run_info)
        if saved is None:
            s.set_stage('views', block_matches=block_matches, plan_sha256=plan_sha, terrain_sha256=terrain_sha, layout_sha256=layout_sha,
                        n_matches=len(rows), n_blocks=len(plan), n_views=n_views, calls=[])
        t_start = time.perf_counter()
        for name, (shape, dtype) in layout.items():
            _allocate(s, name, shape, dtype, call_warnings)      # an existing array is opened after its shape and dtype are checked
        _check_manifest_arrays(s, layout)
        t_allocated = time.perf_counter()
        calls = [dict(c) for c in s.manifest['stages']['views'].get('calls', [])]
        call = {'workers': workers, 'allocate_s': round(t_allocated - t_start, 3), 'views_s': 0.0, 'matches_built': 0, 'blocks_built': 0,
                'memory_samples': 0, 'peak_private_bytes': {'tree': 0, 'main': 0, 'worker': 0},
                'code_commit': now['code_commit'], 'code_dirty': now['code_dirty'], 'warnings': call_warnings}
        calls.append(call)
        s.set_stage('views', calls=calls)                        # this call is on record from now on; it is updated after every block
        # 식 (BLD-5)의 표본: 빌드가 도는 동안 프로세스 트리의 메모리를 재고, 블록이 끝날 때마다 한 번 더 잰다.
        with _Peaks() as peaks:
            def persist():
                call['views_s'] = round(time.perf_counter() - t_allocated, 3)
                call['memory_samples'] = peaks.samples
                call['peak_private_bytes'] = {'tree': peaks.tree, 'main': peaks.main, 'worker': peaks.worker}
                s.set_stage('views', calls=calls)

            def guard(blocks_done, n_blocks):
                """E6 again after a finished block: a model server that starts now would disturb the timing and memory record."""
                try:
                    require_no_strata()
                except StrataRunning as e:
                    raise StrataRunning(
                        f'{e} A model server started while the S2 build was running, so the build stopped after block {blocks_done} of '
                        f'{n_blocks}: every finished block is marked and nothing partial was written. Stop the model server, then resume '
                        f'the run {run.name} with the same arguments.') from None
            try:
                _build_blocks(s, plan, rows, keys, patches, table, count, workers, terr, terrain_path, records, stop_after_block, progress,
                              peaks, call, persist, guard)
            except BaseException:
                try:
                    persist()                                    # what the call did up to its last finished block stays on record
                except Exception:
                    pass                                         # the error that stopped the build is the one to show
                raise
            t_views = time.perf_counter()
            persist()
            missing = sorted(set(range(len(plan))) - s.done('views'))
            if missing:
                raise RuntimeError(f'{len(missing)} block(s) are not marked done (the first: {missing[:5]}): the store is not finalized')
            s.finalize()
        t_end = time.perf_counter()
        call['views_s'], call['memory_samples'] = round(t_views - t_allocated, 3), peaks.samples
        call['peak_private_bytes'] = {'tree': peaks.tree, 'main': peaks.main, 'worker': peaks.worker}       # with the sample after finalize
        s.maps.clear()
        runlog.amend_manifest(run, {'s2_result': _result_record(
            calls=calls, finalize_s=t_end - t_views, complete=bool(s.manifest['complete']), recovered=False, n_blocks=len(plan),
            block_matches=block_matches, n_matches=len(rows), n_views=n_views, by_src=by_src,
            files=s.stored_files(), code_end=runlog.code_state())})
        return root
    finally:
        s1.maps.clear()
        del terr
        gc.collect()                                             # Windows does not delete a folder that holds a mapped file
        if lock is not None:
            _release_lock(lock)


def _build_blocks(s, plan, rows, keys, patches, table, count, workers, terr, terrain_path, records, stop_after_block, progress, peaks, call,
                  persist, guard=None):
    """Build the blocks that are not marked done, one after the other in block order, at most ``workers`` of them in flight.

    ``call`` is the record of this call in the stage: every finished block is counted in it and ``persist()`` writes it. The counters are
    raised before ``mark_done``, whose save then carries them: a cut between the mark and the next save cannot leave a marked block that
    the record does not count (and a mark that fails takes the counters back). ``guard(blocks_done, n_blocks)`` is called after every
    finished block but the last (E6 again: it raises when a model server has appeared). On an error the blocks in flight are cancelled,
    which stops only a block that has not started: a block that is running finishes before the pool shuts down, its result is dropped
    and nothing of it is written.

    수식
    ----
    (BLD-4)  진행 중인 블록 수 ≤ workers,   그 블록들의 배열 바이트 ≤ workers · max_k N_k · x_bytes_per_view   [도출]
             N_k = Σ_{u ∈ 블록 k} N_u는 블록 k의 보기 수이고, 새 블록은 끝난 블록 하나를 소비한 뒤에만 내보낸다(workers = 1이면 한 번에 한 블록).
             x 배열의 몫만 센 상한이다: 경기 맥락, 프로세스 사이의 복사본, 부모의 보기 표와 작업자의 기본 메모리는 넣지 않았다.

      기호             | 뜻                                 | 단위   | 범위 | 코드 변수
      workers          | 작업자 프로세스의 수               | 개     | ≥ 1  | workers
      k, N_k           | 블록 번호, 블록 k의 보기 수        | —, 개  | ≥ 0  | b(본문의 지역 변수), r1 − r0
      x_bytes_per_view | 보기 하나의 x 배열 바이트((BLD-2)) | 바이트 | > 0  | tokens.bytes_per_view()

    해석
    ----
    식 (BLD-4)는 한꺼번에 진행 중인 블록을 작업자 수 이하로 묶는 코드 구조(S1–S2 판정 R13-k)에서 도출한 배열 몫의 상한이다.

    근거
    ----
    S1–S2 판정 R13-k(진행 중인 블록은 작업자 수 이하), 저장소 설계 4.7절(자원 추정은 검증 대상), S1–S2 계획 전역 제약(최대 private 메모리).
    """
    done = s.done('views')
    todo = [b for b in range(len(plan)) if b not in done]
    first = np.concatenate([[0], np.cumsum(count)]).astype(np.int64)       # the first view of every listed match (and the end)
    names = list(layout_arrays(1))

    def make_job(b):
        lo, hi = plan[b]
        items = []
        for j in range(lo, hi):
            a, z = int(first[j]), int(first[j + 1])
            if z > a:                                                      # a match without views is not read at all
                items.append((int(rows[j]), keys[j], patches[j], table['ref'][a:z], table['frame_k'][a:z].astype(np.int64)))
        return b, items

    def consume(b, result):
        got, arrays, t0_seen = result
        lo, hi = plan[b]
        r0, r1 = int(first[lo]), int(first[hi])
        if got != b or len(t0_seen) != r1 - r0:
            raise ValueError(f'block {b}: a worker returned {len(t0_seen)} views of block {got}, the view table has {r1 - r0}')
        bad = np.flatnonzero(t0_seen != table['t0'][r0:r1])
        if bad.size:                                                       # the files were read again: they must give S1's frames
            m = int(table['match'][r0 + int(bad[0])])
            raise S2CheckError(f'block {b}: the frame time of {bad.size} view(s) differs from the one S1 stored (the first in S1 match '
                               f'#{m}): the match files changed since S1, or the frame_k of S1 is wrong')
        s.put_rows(r0, {**{f'views/{c}': a[r0:r1] for c, a in table.items()}, **arrays})
        if stop_after_block == b:
            raise StopRequested(f'stop_after_block={b}: block {b} is written and not marked')
        call['matches_built'] += hi - lo                                   # counted before the mark: the save of mark_done carries the counters
        call['blocks_built'] += 1
        try:
            s.mark_done('views', int(b), names)
        except BaseException:
            if b not in s.done('views'):                                   # the mark did not land: the counters follow the marks
                call['matches_built'] -= hi - lo
                call['blocks_built'] -= 1
            raise
        done.add(b)
        peaks.sample()
        persist()
        if progress is not None:
            progress(len(done), len(plan))
        if guard is not None and len(done) < len(plan):                    # after the last block there is nothing left to protect
            guard(len(done), len(plan))

    if workers == 1:
        builder = _BlockBuilder(terr, records)
        for b in todo:
            consume(b, builder.build(make_job(b)))
    elif todo:
        # 식 (BLD-4): 처음에 workers개 블록을 내보내고 그 뒤에는 결과 하나를 소비할 때마다 블록 하나를 더 내보내므로 진행 중인 블록이 workers개를 넘지 않는다.
        pending, queue = deque(), iter(todo)
        with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_init_worker,
                                    initargs=(str(terrain_path), records)) as pool:
            try:
                for b in itertools.islice(queue, workers):
                    pending.append((b, pool.submit(_run_job, make_job(b))))
                while pending:
                    b, future = pending.popleft()
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        raise RuntimeError('a worker process died or could not start (out of memory, or the terrain cannot be opened '
                                           'in it): nothing of this block was written') from None
                    consume(b, result)
                    nxt = next(queue, None)
                    if nxt is not None:
                        pending.append((nxt, pool.submit(_run_job, make_job(nxt))))
            except BaseException:
                for _, future in pending:
                    future.cancel()                      # only a block that has not started is stopped by this; a running one finishes
                raise
