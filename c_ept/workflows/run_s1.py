r"""S1 runner: the cached matches of the patches 15.18-15.21 -> one S1 store (engagement units, targets, rows) under ``<run>\s1``.

``extract_match`` turns one match into its share of every S1 table with local indices (a unit number counts inside the match). The
runner cuts the ordered match list into blocks of whole matches (``store_core.block_plan``), extracts a block with a process pool
(``Pool.map`` keeps the order) and writes the block's tables to ``<run>\s1_blocks\b<0000>.npz``, outside the store root. After the last
block ``assemble`` sizes every array from the block counts, turns the local indices into global ones (the offsets of ``matches/*_off``)
and writes the rows with ``Store.put_rows``; ``finalize`` then hashes the arrays. Every row depends on its own match alone and the blocks
are joined in match order, so the stored bytes do not depend on the block size, the worker count or restarts.

Index rules of the assembly (-1 means none and stays -1):
    kills/unit, units/kill0, onsets/row, onsets/unit, event_rows/unit    plus the global start of the match in the table they point into
    objectives/unit, objectives/conv_unit, event_rows/frame_row          the same, but only where the value is >= 0
    units/match, frame_rows/match                                        the global match index (written by the assembly)
    event_rows/frame_k                                                   stays local: it indexes that match's own frames

Resume safety. The first run saves the block plan in STORE.json (``block_matches`` and ``plan_sha256``, the SHA-256 of the ordered
(patch, key, macro) list); a restart with another block size or another list is refused, because the format layer cannot see a
mismatched resume. Wall time, memory and the counts go to RUN.json through ``runlog.amend_manifest`` and never to STORE.json, which is
therefore the only file of the store that depends on the block plan (leave it out when comparing two stores).

Match list. The macro role of a match is the cohort split's alone. Before anything is written ``verify_roles`` recomputes it with
``cohort.role_of`` for every listed (patch, key) and refuses a list that names a match twice or whose roles differ (``MatchListError``:
``DuplicateMatch``, ``RoleMismatch``), so ``matches/role`` is never taken on trust from the caller. ``run_s1``, ``build_store`` and
``golden`` take the role function as a parameter the way they take ``records`` (tests with synthetic keys answer for the split).

Privacy. A match key is written only to ``matches/key`` of the store. Messages, REJECTS.json (counts by patch and code) and GOLDEN.json
(hashes) carry no key.


방법에서의 위치. 이 모듈은 한 경기의 교전·정답·질문 행을 추출하고 경기 순서대로 S1 저장소에 조립한다.
교전은 연결된 킬 묶음이고, 발생 행은 분 프레임 뒤의 첫 킬을 세는 질문이며 사건 행은 교전별 예측 기준이다.
입력은 순서 있는 경기·패치·역할 목록과 판독 함수이고, 출력은 S1 배열·거부 집계·실행 기록이다.
식 목록.
    (RUN1-1) 경기별 행 수의 누적 오프셋과 지역 참조의 전역 참조 변환.
    (RUN1-2) 순차 관측한 프로세스 트리 메모리 합의 최대와 GiB 기록.
    (RUN1-3) 요청·저장·거부 회계와 기록 시간·저장 바이트.
공통 가정. 배열의 경기·행·프레임 번호는 0부터 시작하며 참가자 ID와 구분한다.
패치·역할은 설정 목록의 0부터 시작하는 색인이고, 모든 저장 시각·노출은 ms이다.
정답 계산은 각 소유 모듈의 식을 재사용하며, 이 모듈은 열 형식과 참조·집계의 일관성을 검사한다.
float32 열은 narrow를 통과해도 실수의 정밀도 감소가 가능하며 비트 보존을 뜻하지 않는다(casts.narrow).
블록 재개는 같은 목록·블록 크기를 요구하지만 코드 버전의 동일성을 비교하는 검사는 아니다(Task 8a 재개 안전).
메모리는 관측 시점 사이의 진짜 최대치를 보장하지 않는다(Task 8a 실행 기록).
기존 완료 저장소의 조기 반환은 결과 기록과 끝의 요청 회계를 다시 수행하지 않는다(Task 8).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 3-4 and 11 (tables of the S1 store: units, targets K / O / conversions,
#      frame-clock rows, onsets, event-clock rows) and docs/superpowers/plans/2026-10-20-c-ept-s1-s2.md (S1 arrays, task 8).
# Ref: docs/superpowers/specs/2026-10-19-chunked-data-store-design.md sections 4.1, 4.5 and 4.6 (blocks of whole matches; block size,
#      worker count and restarts do not change the stored values; STORE.json written atomically; resume with the same plan).
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import hashlib
import io
import json
import multiprocessing as mp
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:      # the peak memory is then not recorded
    psutil = None

from c_ept import config as C
from c_ept import cohort, event_rows, frame_rows, reader, runlog, store_core, targets, units
from c_ept.casts import narrow      # also re-exported: callers and tests use run_s1.narrow
from c_ept.reader import MatchFormatError, MatchRejected
from c_ept.store_core import Store

S1_PATCHES = C.PATCHES[:4]                 # 15.18-15.21; the holdout patch 15.22 and the external patch 16.13 are never opened here
REFUSED_ROLES = ('TEST', 'EXTERNAL')       # macro roles that S1 never extracts
STORE_DIR, BLOCK_DIR = 's1', 's1_blocks'   # under the run folder; the block files stay outside the store root
KEY_BYTES = 24

# ---- the layout of the S1 store --------------------------------------------------------------------------------------------------
_N_FIELDS = len(C.FRAME_FIELDS)
MATCH_COLUMNS = (('key', f'S{KEY_BYTES}', ()), ('patch', 'i1', ()), ('role', 'i1', ()), ('T_ms', 'i8', ()), ('team', 'i2', (10,)),
                 ('champ', 'i4', (10,)), ('end_xy', 'f4', (10, 2)))
# table -> ((column, dtype, shape after the row axis), ...); the array is named 'table/column'
TABLES = {
    'frames': (('ts', 'i8', ()), ('num', 'f4', (10, _N_FIELDS))),
    'kills': (('t', 'i8', ()), ('x', 'f8', ()), ('y', 'f8', ()), ('killer', 'i1', ()), ('victim', 'i1', ()), ('amask', 'i2', ()),
              ('bounty', 'i4', ()), ('shutdown', 'i4', ()), ('pos_missing', '?', ()), ('unit', 'i4', ())),
    'units': (('match', 'i4', ()), ('kill0', 'i4', ()), ('n_kills', 'i2', ()), ('f', 'i8', ()), ('last', 'i8', ()), ('tau', 'i8', ()),
              ('delta', 'i8', ()), ('nb', 'i1', ()), ('nr', 'i1', ()), ('size', 'i1', ()), ('mark', 'i2', ()), ('in_scope', '?', ()),
              ('spread', 'f4', ()), ('K', 'i4', (len(targets.K_FIELDS),)), ('K_at_f', 'i4', (len(targets.K_FIELDS),)),
              ('exec', 'i2', (2,)), ('O', 'i2', (targets.N_CELLS,)), ('O_at_f', 'i2', (targets.N_CELLS,)),
              ('conv', 'i2', (targets.N_CELLS,)), ('conv_exposure_ms', 'i4', ())),
    'objectives': (('t', 'i8', ()), ('kind', 'i1', ()), ('taker', 'i1', ()), ('x', 'f4', ()), ('y', 'f4', ()), ('has_pos', '?', ()),
                   ('cred', 'i2', ()), ('unit', 'i4', ()), ('conv_unit', 'i4', ()), ('sub', 'i1', ())),
    'frame_rows': (('match', 'i4', ()), ('k', 'i2', ()), ('t0', 'i8', ()), ('hi', 'i8', ()), ('exposure_ms', 'i4', (C.N_BINS,))),
    'onsets': (('row', 'i4', ()), ('bin', 'i1', ()), ('mark', 'i2', ()), ('unit', 'i4', ())),
    'event_rows': (('unit', 'i4', ()), ('kind', 'i1', ()), ('ref', 'i8', ()), ('frame_k', 'i2', ()), ('t0', 'i8', ()),
                   ('age_ms', 'i8', ()), ('frame_row', 'i4', ())),
}
OFFSETS = {'kills': 'kill_off', 'objectives': 'obj_off', 'frames': 'frame_off', 'units': 'unit_off', 'frame_rows': 'row_off',
           'onsets': 'onset_off', 'event_rows': 'event_off'}      # table -> its (n_matches + 1) offset array in matches/
ASSEMBLED = ('units/match', 'frame_rows/match')                    # no per-match value: the assembly writes the match index
# array -> (table whose global start is added, only where the value is >= 0); see the module text
SHIFTS = {'kills/unit': ('units', False), 'units/kill0': ('kills', False), 'objectives/unit': ('units', True),
          'objectives/conv_unit': ('units', True), 'onsets/row': ('frame_rows', False), 'onsets/unit': ('units', False),
          'event_rows/unit': ('units', False), 'event_rows/frame_row': ('frame_rows', True)}
SPEC = {**{f'matches/{c}': (np.dtype(d), shape) for c, d, shape in MATCH_COLUMNS},
        **{f'{t}/{c}': (np.dtype(d), shape) for t, cols in TABLES.items() for c, d, shape in cols}}
_PER_MATCH = [name for name in SPEC if name not in ASSEMBLED]      # the arrays extract_match returns, in layout order
_COLUMNS = {t: [f'{t}/{c}' for c, _, _ in cols if f'{t}/{c}' not in ASSEMBLED] for t, cols in TABLES.items()}
_E6 = ('E6: a Strata / llama / ollama process is running, so S1 does not start. It does not stop the process: stop it yourself and run '
       'S1 again.')


class StrataRunning(RuntimeError):
    """E6: a Strata / llama / ollama model server is running, so no extraction starts. Nothing was written and nothing was stopped.

    지침 E6에 따른 실행 시작 거부를 구분하는 예외이다.
    """


def require_no_strata() -> dict | None:
    """E6 guard: raise ``StrataRunning`` while a model server runs, else return the check record of ``runlog.strata_check``.

    The server is found by its command line (``runlog.strata_check``); the idle MCP bridges do not count. The message names the matched
    processes by name and pid and never shows a command line. The process is never stopped. Without psutil nothing can be checked: a
    ``RuntimeWarning`` and the return value None. ``main``, ``run_s1`` and ``build_store`` call this before run/store creation or
    raw-record processing; preliminary argument, list and role validation may run first. Library callers use this guard as well.


    지침 E6와 S1–S2 판정 R33의 시작 검사를 수행하며, psutil 부재 시 경고와 미확인 반환으로 계속한다.
    """
    check = runlog.strata_check()
    if check is None:
        warnings.warn('psutil is not installed, so E6 (no Strata process) could not be checked', RuntimeWarning, stacklevel=2)
    elif check['running']:
        found = ', '.join(f'{m["name"]} (pid {m["pid"]})' for m in check['matched'])
        raise StrataRunning(f'{_E6} Found: {found}.')
    return check


class MatchListError(ValueError):
    """The ordered match list of a run cannot be trusted. The message holds counts, role names and list positions, never a match key.

    S1–S2 pre-s2-guards-report.md Guard 3의 입력 목록 검사 실패를 나타내는 기본 예외이다.
    """


class DuplicateMatch(MatchListError):
    """A (patch, key) is listed more than once.

    S1–S2 pre-s2-guards-report.md Guard 3의 같은 경기 중복 입력을 구분한다.
    """


class RoleMismatch(MatchListError):
    """A listed macro role differs from the role the cohort split gives that match.

    S1–S2 pre-s2-guards-report.md Guard 3의 호출자 역할과 분할 역할 불일치를 구분한다.
    """


class S1CheckError(RuntimeError):
    """A per-match check of the extraction failed: an input that the design says cannot occur. The message names the check and the
    patch, never the match key; ``key`` and ``patch`` are attributes (a counted rejection is a ``MatchRejected``, not this).

    S1–S2 Task 8a의 경기별 정합성 검사 실패를 구분한다.
    """

    def __init__(self, key: str, patch: str, reason: str):
        super().__init__(f'{reason} (patch {patch})')
        self.key, self.patch, self.reason = key, patch, reason

    def __reduce__(self):                  # the default would call __init__(message) alone: keep it picklable for worker pools
        return type(self), (self.key, self.patch, self.reason)


class S1WorkerError(RuntimeError):
    """An unexpected error inside a worker whose message held the match key; the key is replaced by <key> and the position of the match
    in the ordered list is given instead.

    S1–S2 Task 8a의 작업자 실패를 경기 식별자 없는 메시지로 전달한다.
    """


# ``narrow`` (the range-checked cast) moved unchanged to c_ept/casts.py, imported above: ``run_s1.narrow`` is that function.


def _key_bytes(key) -> bytes:
    """The key as ASCII bytes for the S24 column. ValueError (never echoing the key) for an empty key, a key that is not ASCII text or one
    of more than 24 bytes; numpy would cut such a key without a message."""
    try:
        b = key.encode('ascii')
    except (AttributeError, UnicodeEncodeError):
        raise ValueError(f'a match key must be 1 to {KEY_BYTES} ASCII bytes; one is not ASCII text') from None
    if not 1 <= len(b) <= KEY_BYTES:
        raise ValueError(f'a match key must be 1 to {KEY_BYTES} ASCII bytes; one has {len(b)}')
    return b


def _check_one(patch, key, macro) -> tuple[int, int, bytes]:
    """(patch index, role index, key bytes) of a match, or ValueError: S1 reads only the patches 15.18-15.21 and never the macro roles
    TEST and EXTERNAL."""
    if patch not in S1_PATCHES:
        raise ValueError(f'patch {patch!r} is not an S1 patch {S1_PATCHES}')
    if macro not in C.ROLES:
        raise ValueError(f'unknown macro role {macro!r}: not one of {C.ROLES}')
    if macro in REFUSED_ROLES:
        raise ValueError(f'macro role {macro} is never extracted by S1')
    return C.PATCHES.index(patch), C.ROLES.index(macro), _key_bytes(key)


def _check_matches(matches, workers: int = 1) -> list[tuple[str, str, str]]:
    """The ordered match list as a list of (patch, key, macro) tuples, after every refusal that needs no file: a worker count below 1,
    an empty list, a patch or role S1 does not read, a key that does not fit the key column."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f'workers must be an int of at least 1, got {workers!r}')
    out = [tuple(m) for m in matches]
    if not out:
        raise ValueError('no matches to extract')
    for m in out:
        if len(m) != 3:
            raise ValueError('a match is a (patch, key, macro) triple')
        _check_one(*m)
    return out


def plan_sha256(matches) -> str:
    """SHA-256 of the ordered list of (patch, key, macro): the plan a resumed run must repeat. Hashed in memory; nothing is written.

    S1–S2 Task 8a의 재개 안전 검사에 쓰는 순서 있는 목록 해시이며, 코드 버전의 동일성까지 묶지는 않는다.
    """
    h = hashlib.sha256(f'{len(matches)}\n'.encode('ascii'))
    for patch, key, macro in matches:
        h.update(f'{patch}\x1f{key}\x1f{macro}\x1e'.encode('utf-8'))
    return h.hexdigest()


def verify_roles(matches, role_of=cohort.role_of) -> None:
    """Refuse a match list that names a match twice or whose macro roles are not the cohort split's. Pre-S2 guard (final review of S1,
    Minor 7): the stored role of a match is never taken on trust from the caller.

    ``matches`` is the list [(patch, key, macro), ...] that a run is asked to extract (after ``_check_matches``). No (patch, key) may be
    listed twice (``DuplicateMatch``, checked first, it needs no split), and for every entry ``role_of(key, patch)[1]`` - the macro role
    that the cohort split gives the match, recomputed here - must equal the listed macro (``RoleMismatch``). Both are a ``MatchListError``
    (a ValueError) whose message holds counts, role names and list positions, never a match key. Nothing is read from the raw files and
    nothing is written. ``role_of`` is replaced the way ``records`` is, by tests whose synthetic keys are not in the split.


    S1–S2 pre-s2-guards-report.md Guard 3에 따라 중복을 먼저 검사한 뒤 각 역할을 재계산하며, 중복 실패 때 뒤의 역할 검사는 실행하지 않는다.
    """
    seen, repeats, first_repeat = set(), 0, None
    for i, (patch, key, _) in enumerate(matches):
        if (patch, key) in seen:
            repeats += 1
            first_repeat = i if first_repeat is None else first_repeat
        seen.add((patch, key))
    if repeats:
        raise DuplicateMatch(f'{repeats} of the {len(matches)} listed matches repeat an earlier (patch, key), the first at list position '
                             f'{first_repeat} (0-based); a match is listed once')
    differ, first = {}, None
    for i, (patch, key, macro) in enumerate(matches):
        split_macro = role_of(key, patch)[1]
        if split_macro != macro:
            differ[(macro, split_macro)] = differ.get((macro, split_macro), 0) + 1
            first = i if first is None else first
    if differ:
        pairs = '; '.join(f'{listed} listed, {split_macro} in the split: {n}' for (listed, split_macro), n in sorted(differ.items()))
        raise RoleMismatch(f'{sum(differ.values())} of the {len(matches)} listed matches have a macro role that differs from the cohort '
                           f'split ({pairs}); the first at list position {first} (0-based) in patch {matches[first][0]}')


# ---- one match --------------------------------------------------------------------------------------------------------------------
def extract_match(patch: str, key: str, macro: str, records=reader.read_match) -> dict:
    """The share of one match in every S1 table, with local indices (a unit or row number counts inside the match): a dict
    {array name: array} over the layout without the offsets and the two assembled columns. Every table's arrays have as many rows as the
    match has entries of it; the arrays of ``matches/`` have one row.

    ``records(key, patch)`` returns the ``MatchRecord``. A ``MatchRejected`` (from the reader, or ``same_team_kill`` from the units) is
    returned as ``{'reject': code}``. A ``MatchFormatError`` and every other error propagate and stop the run. Loud checks, each an
    ``S1CheckError``: the record is the match asked for; kill times do not decrease; every kill belongs to a unit; the M3 event rows are
    one per in-scope unit (ruling 27); every onset is a unit in scope. A value that does not fit its column raises ValueError.


    S1–S2 Task 8a의 경기별 배열·형상·참조 검사를 수행하며, 좁은 dtype의 검사는 casts.narrow에 위임한다.
    대상의 첫 킬 절단 값은 S1–S2 판정 R34의 사후 정답 분해이며 입력의 인과성을 보증하는 열이 아니다.
    """
    pi, ri, kb = _check_one(patch, key, macro)
    try:
        return _extract(records(key, patch), key, patch, pi, ri, kb)
    except MatchRejected as e:
        return {'reject': e.code}


def _extract(rec, key: str, patch: str, pi: int, ri: int, kb: bytes) -> dict:
    def fail(reason: str) -> S1CheckError:
        return S1CheckError(key, patch, reason)

    if rec.key != key or rec.patch != patch:
        raise fail('the record that was read is another match than the one asked for')
    kt = units.kill_table(rec)
    if len(kt.t) > 1 and np.any(kt.t[1:] < kt.t[:-1]):
        raise fail('kill times decrease')
    u = units.build_units(kt, rec.team, rec.T_ms)
    n_units = len(u.f)
    unit_of = np.full(len(kt.t), -1, dtype=np.int64)
    for i, g in enumerate(u.comp):
        unit_of[g] = i
    if np.any(unit_of < 0):
        raise fail('a kill belongs to no unit')
    obj = targets.objective_table(rec, u, kt)
    conv, exposure, conv_unit = targets.conversions(obj, u, rec.T_ms)
    K = np.zeros((n_units, len(targets.K_FIELDS)), dtype=np.int64)
    K_at_f, ex = np.zeros_like(K), np.zeros((n_units, 2), dtype=np.int64)
    O, O_at_f = (np.zeros((n_units, targets.N_CELLS), dtype=np.int64) for _ in range(2))
    for i, g in enumerate(u.comp):
        # K_at_f and O_at_f cut the unit's targets at its first kill f. They are HINDSIGHT target splits, never model inputs: which kills
        # form the unit and which objectives it claims were decided with all kills of the unit and with units that start after f (Ruling
        # 34; the final review found 12,606 objectives in 11,528 units that an information-up-to-f rule would attribute differently, and
        # 8 units whose K_at_f would differ). Whether to add a causal companion column is the author's decision at the S1 gate.
        f = int(u.f[i])
        K[i], K_at_f[i] = targets.unit_K(kt, g, rec.team), targets.unit_K(kt, g, rec.team, upto_ms=f)
        ex[i] = targets.unit_exec(kt, g, rec.team)
        O[i], O_at_f[i] = targets.unit_O(obj, i), targets.unit_O(obj, i, upto_ms=f)
    rows = frame_rows.frame_rows(rec)
    on = frame_rows.onsets(rows, u)
    ev = event_rows.event_rows(rec, u, rows)
    if int((ev.kind == event_rows.KIND_M3).sum()) != int(u.in_scope.sum()):
        raise fail('the M3 event rows are not one per in-scope unit')
    if on.unit.size and not bool(u.in_scope[on.unit].all()):
        raise fail('an onset belongs to a unit that is not in scope')

    values = {
        'matches/key': np.array([kb], dtype=SPEC['matches/key'][0]), 'matches/patch': [pi], 'matches/role': [ri],
        'matches/T_ms': [rec.T_ms], 'matches/team': rec.team[None], 'matches/champ': rec.champ[None], 'matches/end_xy': rec.end_xy[None],
        'frames/ts': rec.frame_ts, 'frames/num': rec.frame_num,
        'kills/t': kt.t, 'kills/x': kt.x, 'kills/y': kt.y, 'kills/killer': kt.killer, 'kills/victim': kt.victim, 'kills/amask': kt.amask,
        'kills/bounty': kt.bounty, 'kills/shutdown': kt.shutdown, 'kills/pos_missing': kt.pos_missing, 'kills/unit': unit_of,
        'units/kill0': np.array([g[0] for g in u.comp], dtype=np.int64), 'units/n_kills': u.n_kills, 'units/f': u.f, 'units/last': u.last,
        'units/tau': u.tau, 'units/delta': u.delta, 'units/nb': u.nb, 'units/nr': u.nr, 'units/size': u.size, 'units/mark': u.mark,
        'units/in_scope': u.in_scope, 'units/spread': u.spread, 'units/K': K, 'units/K_at_f': K_at_f, 'units/exec': ex, 'units/O': O,
        'units/O_at_f': O_at_f, 'units/conv': conv, 'units/conv_exposure_ms': exposure,
        'objectives/t': obj.t, 'objectives/kind': obj.kind, 'objectives/taker': obj.taker, 'objectives/x': obj.x, 'objectives/y': obj.y,
        'objectives/has_pos': obj.has_pos, 'objectives/cred': obj.cred, 'objectives/unit': obj.unit, 'objectives/conv_unit': conv_unit,
        'objectives/sub': obj.sub,
        'frame_rows/k': rows.k, 'frame_rows/t0': rows.t0, 'frame_rows/hi': rows.hi, 'frame_rows/exposure_ms': rows.exposure_ms,
        'onsets/row': on.row, 'onsets/bin': on.bin, 'onsets/mark': on.mark, 'onsets/unit': on.unit,
        'event_rows/unit': ev.unit, 'event_rows/kind': ev.kind, 'event_rows/ref': ev.ref, 'event_rows/frame_k': ev.frame_k,
        'event_rows/t0': ev.t0, 'event_rows/age_ms': ev.age_ms, 'event_rows/frame_row': ev.frame_row,
    }
    if set(values) != set(_PER_MATCH):                       # an explicit raise: asserts vanish under python -O
        raise RuntimeError(f'extract_match and the layout differ in {sorted(set(values) ^ set(_PER_MATCH))}')
    out = {}
    for name in _PER_MATCH:
        dtype, shape = SPEC[name]
        out[name] = narrow(name, values[name], dtype)
        if out[name].shape[1:] != shape:
            raise fail(f'{name} has the row shape {out[name].shape[1:]}, not {shape}')
    for t, cols in _COLUMNS.items():
        if len({len(out[c]) for c in cols}) != 1:
            raise fail(f'the arrays of {t} differ in length')
    return out


# ---- blocks -----------------------------------------------------------------------------------------------------------------------
def _task(item, records):
    """One match in a worker. An unexpected error whose text holds the match key is raised again without it (``S1WorkerError``); every
    other error propagates as it is."""
    i, patch, key, macro = item
    try:
        return extract_match(patch, key, macro, records)
    except (MatchFormatError, S1CheckError):
        raise
    except Exception as e:
        try:
            text = str(e)
        except Exception:
            text = ''
        if key and key in text:
            raise S1WorkerError(f'{type(e).__name__}: {text.replace(key, "<key>")} (patch {patch}, match #{i})') from None
        raise


def _block_path(work: Path, b: int) -> Path:
    return work / f'b{b:04d}.npz'


def _block_arrays(patches, results) -> dict:
    """The arrays of one block: the per-match arrays of the matches that were not rejected, joined in order, the row count of every table
    per match (``cnt/<table>``) and the rejections of the block as JSON text {patch: {code: n}}."""
    ok = [r for r in results if 'reject' not in r]
    out = {}
    for name in _PER_MATCH:
        dtype, shape = SPEC[name]
        out[name] = np.concatenate([r[name] for r in ok]) if ok else np.empty((0, *shape), dtype=dtype)
    for t, cols in _COLUMNS.items():
        out[f'cnt/{t}'] = np.array([len(r[cols[0]]) for r in ok], dtype=np.int64)
    rejects = {}
    for patch, r in zip(patches, results):
        if 'reject' in r:
            codes = rejects.setdefault(patch, {})
            codes[r['reject']] = codes.get(r['reject'], 0) + 1
    out['rejects'] = np.array(json.dumps(rejects, sort_keys=True))
    return out


@contextlib.contextmanager
def _mapper(workers: int):
    """``map`` for one worker (no process), else the ``map`` of a spawn pool that is stopped when the block of code ends or fails."""
    if workers == 1:
        yield map
        return
    with mp.get_context('spawn').Pool(workers) as pool:
        yield pool.map


# ---- assembly ---------------------------------------------------------------------------------------------------------------------
def _merge_rejects(total: dict, part: dict) -> None:
    for patch, codes in part.items():
        for code, n in codes.items():
            d = total.setdefault(patch, {})
            d[code] = d.get(code, 0) + n


def _write_block(s: Store, d: dict, m0: int, off: dict) -> None:
    """Write one block (``d``: the arrays of its npz) whose first match has the global number ``m0``.

    (RUN1-1)의 유효 참조 이동과 음수 보존을 적용한다.
    """
    m = len(d['matches/key'])
    if not m:
        return
    cnt = {t: d[f'cnt/{t}'] for t in TABLES}
    start = {t: off[t][m0:m0 + m] for t in TABLES}                 # the global row where each match of the block begins, per table
    s.put_rows(m0, {name: d[name] for name in d if name.startswith('matches/')})
    for t, cols in TABLES.items():
        rows = int(cnt[t].sum())
        if rows == 0:
            continue
        out = {}
        for c, _, _ in cols:
            name = f'{t}/{c}'
            if name in ASSEMBLED:
                a = np.repeat(np.arange(m0, m0 + m, dtype=np.int64), cnt[t])
            else:
                a = d[name]
                if len(a) != rows:
                    raise ValueError(f'{name}: a block holds {len(a)} rows, its counts say {rows}')
                if name in SHIFTS:
                    other, only_valid = SHIFTS[name]
                    a = a.astype(np.int64)
                    valid = a >= 0
                    if not only_valid and not valid.all():
                        raise ValueError(f'{name}: a negative index where none is allowed')
                    if np.any(a[valid] >= np.repeat(cnt[other], cnt[t])[valid]):
                        raise ValueError(f'{name}: an index points past the rows of its own match')
                    a = np.where(valid, a + np.repeat(start[other], cnt[t]), a)
            out[name] = narrow(name, a, SPEC[name][0])
        s.put_rows(int(off[t][m0]), out)


def _assemble(s: Store, work: Path, n_blocks: int, rejects_path: Path | None) -> None:
    """Allocate every array at its total size, write all blocks with global indices, fsync the arrays and mark the stage.

    (RUN1-1)의 경기별 누적 오프셋과 표별 전체 크기를 계산한다.
    """
    counts = {t: [] for t in TABLES}
    rejects = {}
    for b in range(n_blocks):
        with np.load(_block_path(work, b)) as blk:
            for t in TABLES:
                counts[t].append(blk[f'cnt/{t}'])
            _merge_rejects(rejects, json.loads(str(blk['rejects'][()])))
    n = sum(len(c) for c in counts['frames'])
    off = {}
    for t in TABLES:
        off[t] = np.zeros(n + 1, dtype=np.int64)
        if n:
            off[t][1:] = np.cumsum(np.concatenate(counts[t]))
    if rejects_path is not None:
        store_core.write_atomic(rejects_path, json.dumps(rejects, indent=2, sort_keys=True) + '\n')
    shapes = {}
    for name, (dtype, shape) in SPEC.items():
        t = name.split('/')[0]
        shapes[name] = ((n if t == 'matches' else int(off[t][-1]), *shape), dtype)
    for t, name in OFFSETS.items():
        shapes[f'matches/{name}'] = ((n + 1,), np.dtype('i8'))
    store_core.require_free_bytes(s.root, sum(int(np.prod(sh)) * dt_.itemsize for sh, dt_ in shapes.values()))
    for name, (shape, dtype) in shapes.items():
        s.allocate(name, shape, dtype)
    m0 = 0
    for b in range(n_blocks):
        with np.load(_block_path(work, b)) as blk:
            d = {name: blk[name] for name in blk.files}
        _write_block(s, d, m0, off)
        m0 += len(d['matches/key'])
        del d
    if m0 != n:
        raise ValueError(f'the blocks hold {m0} matches, their counts say {n}')
    s.put_rows(0, {f'matches/{name}': off[t] for t, name in OFFSETS.items()})
    s.mark_done('assemble', 0, list(shapes))                       # fsyncs every array before the store may be called complete


def build_store(matches, root, work, workers: int = 8, block_matches: int = 500, records=reader.read_match, progress=None,
                rejects_path: Path | None = None, role_of=cohort.role_of) -> dict:
    """Extract ``matches`` in blocks into the store at ``root`` (block files in ``work``, outside it) and finalize it.

    Resumes an interrupted store of the same plan: blocks that are marked done (and whose file exists) are not extracted again, and an
    assembly that was marked done is not repeated. A store that is complete is left alone. A saved plan that differs (the block size or
    the ordered match list) raises ValueError before anything is changed. ``progress(blocks_done, n_blocks)`` is called after every
    block. ``rejects_path`` receives the rejection counts {patch: {code: n}} (no keys). Returns the stage seconds of this call, the
    block counts and whether the store was already complete. ``MatchListError`` (``verify_roles``: a match listed twice, or a macro role
    that ``role_of`` - the cohort split - does not give) and ``StrataRunning`` (E6) while a model server runs, both before the store is
    opened: every way into the extraction (``run_s1``, ``golden``, a direct call) stops there.


    수식
    ----
    (RUN1-1) offset_{table_name,0} = 0;
             offset_{table_name,match_row+1} = Σ_{match_pos=0}^{match_row} count_{table_name,match_pos}.
             global_index = local_index + offset_{target_table,match_row} (local_index ≥ 0);
             global_index = local_index (음수 허용 참조의 local_index < 0).
             stored_match = block_first + block_local;
             allocation_bytes = Σ_{array_entry} prod(array_shape)·dtype_bytes.

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    table_name, target_table | 저장 표와 참조 대상 표 | — | TABLES의 키 | t, other
    match_row, match_pos | 저장된 경기의 0부터 시작하는 순서 | 경기 색인 | 0 이상 | m0와 경기별 위치
    count_{table_name,match_pos} | 해당 경기의 표 행 수 | 행 | 0 이상을 전제 | cnt / counts
    offset_{table_name,match_row} | 경기 시작의 누적 행 수 | 행 | int64, 표현 범위 내 합을 전제 | off / start
    local_index | 한 경기 안의 참조 | 행 색인 | 유효값은 대상 행 수 미만 | a
    global_index | 전체 저장소에서의 참조 | 행 색인 | 같은 음수 또는 이동한 유효값 | out[name]
    stored_match, block_first, block_local | 조립 열의 경기 번호·블록 시작·블록 안 순서 | 경기 색인 | 0 이상 | a, m0, arange의 상대 위치
    array_entry, array_shape, dtype_bytes | 할당 배열·형상·원소 크기 | —; 행별 차원; byte | SPEC·오프셋 형상 | shapes, shape, dt_.itemsize
    allocation_bytes | 사전 공간 검사에 넘기는 배열 바이트 합 | byte | 정수 곱·합의 표현 범위 전제 | require_free_bytes의 인자

    해석
    ----
    식 (RUN1-1)은 경기 순서를 유지하며 참조만 전체 저장소의 행 번호로 바꾼다(S1–S2 Task 8a 지역→전역 색인).
    오브젝트의 교전·전환 참조와 사건 행의 분 프레임 참조는 음수를 그대로 두며, 나머지 이동 열은 음수를 거부한다(Task 8a).
    통상 미지정 값은 −1이지만 이 이동 단계는 음수 허용 열의 음수 전체를 보존한다(Task 8a의 미지정 보존).
    사건 행의 경기 내부의 정기 프레임 번호는 이동하지 않고, 빈 표는 행 쓰기를 건너뛴다(Task 8a 저장 배치).
    완료 저장소도 목록·역할·E6·저장된 계획 검사를 거친 뒤 반환하지만 재조립하지는 않는다(Task 8a 재개 안전).

    근거
    ----
    S1–S2 Task 8a rulings(지역→전역 색인·재개 안전), S1–S2 판정 R28(블록 분리).
    """
    matches = _check_matches(matches, workers)
    plan = store_core.block_plan(np.arange(len(matches)), block_matches)
    verify_roles(matches, role_of)                                                  # the roles are the split's, no match twice: before E6
    require_no_strata()                                                             # E6: after preliminary argument/list/role validation, before store creation or raw-record processing
    root, work = Path(root), Path(work)
    s = Store(root)
    sha = plan_sha256(matches)
    saved = s.manifest['stages'].get('extract')
    if saved is None:
        s.set_stage('extract', block_matches=block_matches, plan_sha256=sha, n_matches=len(matches), n_blocks=len(plan))
    else:
        if saved.get('block_matches') != block_matches:
            raise ValueError(f'resume refused: the saved run used block_matches={saved.get("block_matches")}, this one {block_matches}')
        if saved.get('plan_sha256') != sha or saved.get('n_matches') != len(matches):
            raise ValueError('resume refused: the match list differs from the saved plan (plan_sha256)')
    if s.manifest.get('complete'):
        return {'already_complete': True}
    work.mkdir(parents=True, exist_ok=True)
    done = {b for b in s.done('extract') if _block_path(work, b).is_file()}
    resumed, seconds = len(done), {}
    t0 = time.perf_counter()
    todo = [b for b in range(len(plan)) if b not in done]
    if todo:
        task = functools.partial(_task, records=records)
        with _mapper(workers) as mapper:
            for b in todo:
                lo, hi = plan[b]
                results = list(mapper(task, [(i, *matches[i]) for i in range(lo, hi)]))
                buf = io.BytesIO()
                np.savez(buf, **_block_arrays([m[0] for m in matches[lo:hi]], results))
                path = _block_path(work, b)
                store_core.write_atomic(path, buf.getvalue())
                s.mark_done('extract', b, [path])
                done.add(b)
                if progress is not None:
                    progress(len(done), len(plan))
    seconds['extract'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if 0 not in s.done('assemble'):
        _assemble(s, work, len(plan), rejects_path)
    seconds['assemble'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s.finalize()
    seconds['finalize'] = time.perf_counter() - t0
    return {'already_complete': False, 'n_blocks': len(plan), 'blocks_resumed': resumed, 'stage_s': seconds}


# ---- run record -------------------------------------------------------------------------------------------------------------------
class _PeakMemory:
    """Sampled peak of the summed private bytes of this process and its children, falling back to RSS when private is absent or zero.
    Samples run every ``interval`` seconds from a daemon thread and at the start and end; unreadable processes are omitted and
    inter-sample peaks may be missed. Without psutil there is no peak.

    (RUN1-2)의 관측 시점별 프로세스 트리 합과 최대를 보관한다.
    """

    def __init__(self, interval: float = 2.0):
        self.interval, self.peak, self.samples = interval, 0, 0
        self._stop, self._thread = threading.Event(), None

    def sample(self) -> None:
        """(RUN1-2)의 한 번의 순차 메모리 관측과 최대 갱신을 계산한다."""
        total = 0
        try:
            me = psutil.Process()
            for p in [me, *me.children(recursive=True)]:
                try:
                    mi = p.memory_info()
                except psutil.Error:
                    continue
                total += getattr(mi, 'private', None) or mi.rss          # Windows: the private bytes
        except psutil.Error:
            return
        self.samples += 1
        self.peak = max(self.peak, total)

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


def _table_rows(s: Store) -> dict:
    """Rows of every table from the manifest: {'matches': n, 'frames': ..., 'kills': ..., ...}."""
    arrays = s.manifest['arrays']
    return {'matches': arrays['matches/key']['shape'][0],
            **{t: arrays[f'{t}/{TABLES[t][0][0]}']['shape'][0] for t in TABLES}}


def _match_counts(s: Store) -> tuple[dict, dict, dict]:
    """Stored matches by macro, by patch and by macro and patch (names from the config lists, zero counts left out)."""
    patch, role = np.array(s.array('matches/patch')), np.array(s.array('matches/role'))
    by_macro, by_patch, by_both = {}, {}, {}
    for r, p in zip(role.tolist(), patch.tolist()):
        macro, name = C.ROLES[r], C.PATCHES[p]
        by_macro[macro] = by_macro.get(macro, 0) + 1
        by_patch[name] = by_patch.get(name, 0) + 1
        by_both.setdefault(macro, {})
        by_both[macro][name] = by_both[macro].get(name, 0) + 1
    return by_macro, by_patch, by_both


def run_s1(matches, run, workers: int = 8, block_matches: int = 500, records=reader.read_match, info: dict | None = None,
           progress=None, role_of=cohort.role_of) -> Path:
    r"""S1 for the ordered ``matches`` [(patch, key, macro), ...] in the run folder ``run``; returns the store root ``<run>\s1``.

    Refusals that need no file come first and write nothing (empty list, workers < 1, bad block size, a patch other than 15.18-15.21, the
    macro roles TEST and EXTERNAL, a key that does not fit 24 ASCII bytes, then ``verify_roles``: a match listed twice or a macro role
    that ``role_of`` - ``cohort.role_of``, the split - does not give, ``MatchListError``), then E6 (``StrataRunning`` while a model
    server runs; also before anything is written). Then RUN.json (``runlog.write_manifest``; ``info`` goes into
    its ``args``) unless the folder has one already, the blocks, the assembly, REJECTS.json (counts by patch and code, no keys) and, last,
    ``runlog.amend_manifest(run, {'s1_result': ...})`` with the wall time, the peak private memory, the counts and the size of the
    store. Registering the run in the run index is the caller's step (``main`` does it). A run whose store is complete is returned as it
    is. Resuming needs the same ``block_matches`` and the same match list (ValueError otherwise).


    수식
    ----
    (RUN1-2) sample_bytes = Σ_{process_entry ∈ readable_processes} (private_bytes or rss_bytes);
             peak_bytes = max({0} ∪ {sample_bytes: 성공한 관측});
             peak_gib = round(peak_bytes / 2^30, 2) (psutil 있음); 아니면 None.
    (RUN1-3) stored_matches + rejected_matches = requested_matches;
             store_bytes = Σ_{file_entry ∈ store_files} file_bytes;
             wall_s = round(elapsed_s, 1), stage_s = round(stage_elapsed_s, 1).

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    readable_processes, process_entry | 주 프로세스와 재귀 자식 중 읽은 항목 | — | 관측 중 읽기 실패 항목 제외 | me, p
    private_bytes, rss_bytes | private 속성과 대체 RSS | byte | psutil 반환값을 전제 | mi.private, mi.rss
    sample_bytes, peak_bytes | 한 관측 합과 관측 최대 | byte | 0 이상, 초기 최대 0 | total, mem.peak
    peak_gib | 기록한 관측 최대 | GiB | float 또는 None | peak_private_gib
    stored_matches | 실제 저장 경기 수 | 경기 | 0 이상 | rows['matches']
    rejected_matches | 패치·코드별 거부 수의 합 | 경기 | 0 이상 | n_rejected
    requested_matches | 요청 목록 길이 | 경기 | 1 이상 | len(matches)
    store_files, file_entry, file_bytes | 저장소 아래 파일 집합·항목·크기 | —; —; byte | 파일만 포함 | files, p, stat().st_size
    store_bytes | 저장소 파일 크기의 합 | byte | 0 이상 | s1_result['store_bytes']
    elapsed_s, stage_elapsed_s | 전체·단계 경과 시간 | s | perf_counter의 차 | wall, stats['stage_s']
    wall_s, stage_s | 기록용 시간 | s | Python round의 float | s1_result의 해당 필드
    round | Python의 자리수 지정 반올림 | 입력과 같음 | 이진 float 규칙 | round

    해석
    ----
    식 (RUN1-2)는 시작·주기·종료의 순차 관측 합에서 최대를 고르며, private가 없거나 0이면 RSS를 쓴다(Task 8a 실행 기록).
    읽기에 실패한 자식은 그 관측에서 빠지고, psutil이 있어도 성공 관측이 없으면 초기값이 남으므로 실제 최대의 상한이 아니다(Task 8a).
    식 (RUN1-3)의 회계는 완료 후 요청 누락을 검사하고, 바이트 합은 블록 폴더를 제외한 저장소 내부 파일을 센다(Task 8a 저장 배치).
    이미 완료된 저장소의 반환은 이 회계와 결과 추가 기록보다 먼저 일어난다(Task 8a 재개 안전).
    반올림은 기록 정밀도이며 초 단위 측정값을 ms 저장 시각과 혼동하지 않는다(Task 8a 실행 기록).

    근거
    ----
    S1–S2 Task 8a rulings(실행 기록·저장 배치·재개 안전), S1–S2 계획 Task 8(추출 회계).
    """
    matches = _check_matches(matches, workers)
    store_core.block_plan(np.arange(len(matches)), block_matches)                    # the block size is valid, nothing is written yet
    # A refused list leaves no run folder either. (build_store verifies again: golden and direct callers come in there, not here.)
    verify_roles(matches, role_of)
    require_no_strata()                                                              # E6: a refused start leaves no run folder either
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    root = run / STORE_DIR
    if not (run / 'RUN.json').exists():
        runlog.write_manifest(run, {'args': {'workers': workers, 'block_matches': block_matches, 'n_matches': len(matches),
                                             'plan_sha256': plan_sha256(matches), **(info or {})}})
    t_start = time.perf_counter()
    # (RUN1-2)의 관측 구간은 빌더 호출이며, 실행 전체의 실제 최대를 증명하지 않는다(Task 8a 실행 기록).
    with _PeakMemory() as mem:
        stats = build_store(matches, root, run / BLOCK_DIR, workers, block_matches, records, progress, run / 'REJECTS.json', role_of)
    if stats['already_complete']:
        return root
    wall = time.perf_counter() - t_start
    s = Store(root)
    rejects = json.loads((run / 'REJECTS.json').read_text(encoding='utf-8'))
    n_rejected = sum(sum(codes.values()) for codes in rejects.values())
    rows = _table_rows(s)
    # (RUN1-3)의 요청 회계는 조기 완료 반환을 지난 경로에서만 검사한다(S1–S2 Task 8a).
    if rows['matches'] + n_rejected != len(matches):
        raise ValueError(f'{len(matches)} matches asked, {rows["matches"]} stored and {n_rejected} rejected')
    by_macro, by_patch, by_both = _match_counts(s)
    s.maps.clear()
    files = [p for p in root.rglob('*') if p.is_file()]
    # (RUN1-2)·(RUN1-3)의 단위 변환과 반올림은 기록에만 적용한다(Task 8a 실행 기록).
    runlog.amend_manifest(run, {'s1_result': {
        'complete': bool(s.manifest['complete']), 'wall_s': round(wall, 1),
        'stage_s': {k: round(v, 1) for k, v in stats['stage_s'].items()},
        'peak_private_gib': round(mem.peak / 2 ** 30, 2) if psutil is not None else None, 'memory_samples': mem.samples,
        'workers': workers, 'block_matches': block_matches, 'n_blocks': stats['n_blocks'], 'blocks_resumed': stats['blocks_resumed'],
        'matches_requested': len(matches), 'matches_stored': rows['matches'], 'matches_by_macro': by_macro, 'matches_by_patch': by_patch,
        'matches_by_macro_patch': by_both, 'rejects_total': n_rejected, 'rejects': rejects, 'rows': rows,
        'store_files': len(files), 'store_bytes': sum(p.stat().st_size for p in files)}})
    return root


def load_s1(run, verify: bool = True) -> Store:
    r"""The S1 store of a run folder, or of the store folder ``<run>\s1`` itself. ValueError unless it is complete and, with ``verify``,
    every file still has its recorded SHA-256 (and none was added or removed).

    S1–S2 계획 Task 8의 완료 저장소를 열며, verify를 끄면 이 함수에서 파일 해시 검사는 수행하지 않는다.
    """
    run = Path(run)
    root = run if (run / 'STORE.json').is_file() else run / STORE_DIR
    if not (root / 'STORE.json').is_file():
        raise FileNotFoundError(f'no S1 store under {run}')
    s = Store(root)
    if not s.manifest.get('complete'):
        raise ValueError(f'the S1 store {root} is not complete')
    if verify:
        s.verify()
    return s


# ---- golden twenty ----------------------------------------------------------------------------------------------------------------
# ``casts`` joined the list on 2026-10-21 (Ruling 40): since 24e40ea ``run_s1.narrow`` is ``casts.narrow``, the range-checked cast that
# shapes every S1 array, so a change to it changes the arrays. The first golden lock (2026-10-20) lists the other ten: it is taken again.
EXTRACTION_MODULES = ('config', 'reader', 'units', 'targets', 'frame_rows', 'event_rows', 'store_core', 'run_s1', 'legacy', 'cohort',
                      'casts')


# Fixed source inventory includes implementations and all compatibility/package glue.
EXTRACTION_PATHS = (
    'c_ept/__init__.py',
    'c_ept/_compat/__init__.py',
    'c_ept/config.py',
    'c_ept/data/reader.py',
    'c_ept/_compat/reader.py',
    'c_ept/domain/units.py',
    'c_ept/_compat/units.py',
    'c_ept/domain/targets.py',
    'c_ept/_compat/targets.py',
    'c_ept/data/frame_rows.py',
    'c_ept/_compat/frame_rows.py',
    'c_ept/data/event_rows.py',
    'c_ept/_compat/event_rows.py',
    'c_ept/data/store_core.py',
    'c_ept/_compat/store_core.py',
    'c_ept/workflows/run_s1.py',
    'c_ept/_compat/run_s1.py',
    'c_ept/legacy.py',
    'c_ept/cohort.py',
    'c_ept/data/casts.py',
    'c_ept/_compat/casts.py',
    'c_ept/data/__init__.py',
    'c_ept/domain/__init__.py',
    'c_ept/workflows/__init__.py',
)


def extraction_state() -> dict:
    """Which code made the arrays of a golden lock, as git sees it now: ``code_commit`` (HEAD), ``extraction_clean`` and ``dirty_paths``.

    ``extraction_clean`` is True only when git shows no uncommitted change (modified, staged, deleted or untracked) in the extraction
    fixed ``EXTRACTION_PATHS`` inventory (implementations, compatibility files and package glue, with the runner itself); the tests and the
    report do not count. ``dirty_paths`` lists the changed paths of those modules, repository-relative and sorted, as paths only.
    When git cannot answer, both are None: the state is unknown, not clean. ``extraction_modules`` names the paths that were checked.


    S1–S2 판정 R40의 기준값 잠금에 쓰는 지정 모듈의 Git 상태이며, 조회 실패를 깨끗한 코드로 기록하지 않는다.
    """
    paths = list(EXTRACTION_PATHS)
    commit = (runlog._git('rev-parse', 'HEAD') or '').strip() or None
    status = runlog._git('status', '--porcelain', '--', *paths)
    if status is None:
        return {'code_commit': commit, 'extraction_clean': None, 'dirty_paths': None, 'extraction_modules': paths}
    dirty = sorted({line[3:].split(' -> ')[-1].strip().strip('"') for line in status.splitlines() if line.strip()})
    return {'code_commit': commit, 'extraction_clean': not dirty, 'dirty_paths': dirty, 'extraction_modules': paths}


def golden(n: int, out, records=reader.read_match, matches=None, role_of=cohort.role_of) -> dict:
    """Lock the S1 arrays of the first ``n`` FIT matches of patch 15.18 (sorted key order; ``matches`` replaces that selection in tests).

    Builds an ordinary store for them in ``out`` (one block, one worker) and writes ``<out>/GOLDEN.json`` with the SHA-256 of every
    array file, the plan hash and the state of the code (``extraction_state``: the commit, whether the extraction modules were clean and
    the dirty paths): no keys. An existing GOLDEN.json is never overwritten (FileExistsError). Returns the dict that was written.
    Later runs repeat the extraction and compare the hashes (task 8b). ``MatchListError`` and ``StrataRunning`` (E6) as for
    ``build_store``.


    S1–S2 Task 8b와 판정 R40의 기준값 잠금 기능이며, 기존 잠금 파일을 덮지 않는다.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f'n must be an int of at least 1, got {n!r}')
    out = Path(out)
    default = matches is None
    sel = _check_matches((cohort.match_list('15.18', 'FIT') if default else matches)[:n])
    if len(sel) != n:
        raise ValueError(f'the selection has {len(sel)} matches, {n} are needed')
    if (out / 'GOLDEN.json').exists():
        raise FileExistsError(f'{out / "GOLDEN.json"} exists: a lock file is not overwritten')
    state = extraction_state()                       # read before the extraction runs: it is the code that makes the arrays
    build_store(sel, out / STORE_DIR, out / BLOCK_DIR, workers=1, block_matches=n, records=records, role_of=role_of)
    s = Store(out / STORE_DIR)
    info = {'selection': f'first {n} FIT matches of patch 15.18 in sorted key order' if default else f'first {n} matches of a given list',
            'n_matches': n, 'plan_sha256': plan_sha256(sel), 'rows': _table_rows(s), 'files': dict(s.manifest['files']),
            **state, 'created': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    store_core.write_atomic(out / 'GOLDEN.json', json.dumps(info, indent=2, sort_keys=True) + '\n')
    return json.loads(json.dumps(info))


# ---- command line -----------------------------------------------------------------------------------------------------------------
def _refuse(message: str) -> int:
    print(message, file=sys.stderr, flush=True)
    return 2


def main(argv=None) -> int:
    """S1–S2 계획 Task 8의 인자 검사를 거쳐 실행·색인 등록을 연결하며, 기준값 잠금 경로는 일반 실행과 분기한다."""
    ap = argparse.ArgumentParser(prog='python -m c_ept.run_s1', description='S1: the cached matches of 15.18-15.21 -> one S1 store.')
    ap.add_argument('--macros', nargs='+', metavar='ROLE',
                    help='macro roles to extract, e.g. FIT STOP DEV CAL AUDIT (TEST and EXTERNAL are refused)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--block-matches', type=int, default=500)
    ap.add_argument('--golden', type=int, metavar='N', help='lock the first N FIT matches of 15.18 (needs --out)')
    ap.add_argument('--out', type=Path, help='folder of the golden lock')
    ap.add_argument('--runs-root', type=Path, default=C.RUNS_ROOT)
    ap.add_argument('--catalog-db', type=Path, default=C.CATALOG_DB)
    a = ap.parse_args(argv)
    if (a.macros is None) == (a.golden is None):
        return _refuse('give either --macros ROLE... or --golden N --out DIR')
    if a.golden is not None:
        if a.out is None:
            return _refuse('--golden needs --out DIR')
    else:
        bad = [m for m in a.macros if m in REFUSED_ROLES or m not in C.ROLES]
        if bad:
            return _refuse(f'macro role(s) {bad} refused: S1 extracts roles of {C.ROLES} except {REFUSED_ROLES}')
    try:
        require_no_strata()                                  # E6: no Strata / llama process during the S1 run (Ruling 31: before new_run)
    except StrataRunning as e:
        return _refuse(str(e))
    if a.golden is not None:
        info = golden(a.golden, a.out)
        print(f'golden lock written: {a.out / "GOLDEN.json"} ({info["n_matches"]} matches, {len(info["files"])} array files, '
              f'extraction_clean={info["extraction_clean"]}, dirty paths {info["dirty_paths"]})')
        return 0
    matches = cohort.match_list(S1_PATCHES, a.macros)         # lists the cached matches; no raw file is read here
    if not matches:
        return _refuse(f'no cached matches for the roles {a.macros}')
    run = runlog.new_run('s1', root=a.runs_root)
    print(f'S1 run folder: {run}; {len(matches)} matches, {a.workers} workers, blocks of {a.block_matches}', flush=True)
    t0 = time.perf_counter()

    def progress(done, total):
        print(f'  block {done}/{total} done, {time.perf_counter() - t0:.0f} s', flush=True)
    run_s1(matches, run, a.workers, a.block_matches, info={'macros': list(a.macros), 'patches': list(S1_PATCHES)}, progress=progress)
    manifest = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    runlog.register(run, manifest, a.catalog_db)
    r = manifest['s1_result']
    print(f'S1 done: complete={r["complete"]}, {r["matches_stored"]}/{r["matches_requested"]} matches stored, '
          f'{r["rejects_total"]} rejected, {r["wall_s"]} s, peak {r["peak_private_gib"]} GiB private, '
          f'store {r["store_bytes"] / 2 ** 30:.2f} GiB; rows {r["rows"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
