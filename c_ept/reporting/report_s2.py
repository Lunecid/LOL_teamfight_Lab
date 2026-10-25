r"""S2 smoke report: the size, memory, overflow and terrain figures that the S2 gate looks at, from a finished S2 run folder.

    python -m c_ept.report_s2 <S2 run folder> [--s1 <S1 run folder>] [--terrain <terrain folder>] [--no-matches] [--no-verify]

``report(run)`` reads files only: ``<run>\RUN.json`` (``s1``, ``terrain`` and ``s2_result``, as ``store_build`` writes them), the S2 store
``<run>\s2`` (the six ``views/<column>`` arrays and the ``x/<key>`` array of every key of ``tokens.VIEW_SPEC``), the terrain folder
(``<run>\terrain``, or ``terrain_dir``) and, read only, the S1 store of the run it consumed (found next to the S2 run folder by the name in
RUN.json, or ``s1``). It never opens a raw match unless ``records`` is given, and then it needs the E6 check. ``records`` (a function
``(key, patch) -> MatchRecord``, such as ``reader.read_match``) switches on the two measurements that need the matches themselves
(``level_lag``, ``building_kill_matching``). The measurement checks E6 itself (``run_s1.StrataRunning`` while a Strata server runs, so a
caller of ``report(records=...)`` from Python is protected too), and it runs before the first figure is drawn, so a refused report has
written nothing; the command line passes the reader unless ``--no-matches`` is given and refuses to start while a Strata server runs
(exit code 2, nothing read or written). The measurement counts the errors of the reader and of the record format (``MatchRejected``,
``MatchFormatError``, ``MatchReadError``, ``ValueError``) and goes on; any other error stops the report. Nothing is written into the S2
store, the terrain or the S1 run.

Inputs that do not belong together stop the report (``ReportInputError``, a ValueError; nothing is written, and the command line prints
``report refused: ...`` and returns 2): a store that is not complete or whose files do not hash to STORE.json (``verify``), a layout other
than ``VIEW_SPEC``, a RUN.json that lacks a field the gate needs (the message names it) or disagrees with the store (views, views by
source, matches, bytes per view), another S1 store or terrain than the one RUN.json recorded, a terrain whose two data files do not
belong together or differ from its TERRAIN.json, a TERRAIN.json that lacks a field the report reads (``sha256``, ``grid.min_count``, the
position census ``positions``: all checked before the first figure), an output path inside the S2 store or the S1 run. Findings do not
stop it; they are the ``flags`` at the top of S2_REPORT.json (a code and a message) and are printed by the command
line as ``FLAG <code>: ...``: peak private memory over 48 GiB (or not recorded), more than 5 % of the views that drop events, edges, or
either (``EVENT_...``, ``EDGE_...`` and ``EVENTS_OR_EDGES_OVERFLOW_OVER_5_PERCENT``; the cap that keeps the share at 5 % is written for
a kind that is over 5 % on its own, a proposal: no cap is changed), NaN or infinity where ``tokens.NAN_ALLOWED`` allows
none, a frame-clock level that differs from the level clock without a LEVEL_UP on the frame's ms, a model server that ran during the build,
a terrain that was not built from this S1 run, warnings that the build recorded, a run record that was repaired after a crash, errors in
the raw measurements, a views table that disagrees with the arrays. The terrain is reported as it was built, also when its parameters are
not those of the present code (the build refuses such a terrain; the legend of figure 1 and the ``terrain.grid`` record show what an
older run used).

RUN.json is read as ``store_build`` writes it: ``s2_result`` holds the numbers of the whole build (of a resumed one: the maxima, sums and
counts over its calls), ``args`` the plan and, under ``args.info`` (before the Task 13 fix round: in ``args`` itself), the command line's
macros and limit; ``calls``, ``recovered``, ``warnings`` and ``per_call`` of ``s2_result`` are optional and are echoed when they are there.

S2_REPORT.json, in the order of the file (counts, shares and names only; no match key, puuid or player name: the finished text is checked
against the keys of the S1 store before it is written):

    flags, run (the RUN.json record: code commit, GPU at the start, Strata, the command line, the calls, warnings and recovery of the
        build), s1_run, report_code
    views, matches (the matches listed in RUN.json), matches_with_views (a very short match can leave no view), views_per_match (views
        per listed match), views_by_src {frame_clock, M3, M4}, bytes_per_view (the total: the x arrays and the 27-byte views table) and
        bytes_per_view_parts, seconds_per_match (the views stage over the matches built, as RUN.json records it), wall_s, workers
    peak_private_gib (the sum over the process tree: the value of the 48 GiB gate), peak_private_main_gib, peak_private_worker_gib (the
        largest single worker; None when no pool ran), memory_gate
    event_overflow_share, edge_overflow_share (the share of views with ev_dropped / ed_dropped above 0), overflow (caps, counts by source,
        the events and edges of the window before the cap, the cap that keeps 5 % where the share is over 5 % and None, with a note,
        where it is not, the share of views that drop either)
    unknown_item_share (the share of the 10 N player rows that own an item that the patch's item table lacks: unknown_item_count above 0),
        unknown_champion_share (champ_unknown 1), unknown (the counts behind them)
    level_formula_check: lvl_attackDamage + item_attackDamage - cs_attackDamage on the frame-clock views (R14-b); the rows of an unknown
        champion (NaN) and the rows with an unknown item (the item term is short) are left out and counted
    extrapolated_gib: bytes per view x the exact view count of the S1 store for FIT, STOP, DEV, CAL and AUDIT (cache_full_build), and x
        an estimated count for the whole raw FIT cohort (raw_fit_estimate), each with its arithmetic
    suspect_positions: the positions at exactly (0, 0), off the map and NaN, by source, from TERRAIN.json (the FIT matches that made the
        terrain) and in the views of this run (S2-9)
    nan_share: the share of NaN per allowed column and cell and what holds NaN or infinity outside ``tokens.NAN_ALLOWED`` (Ruling 42)
    constant_columns: summary and player columns with one value in every row (for example ``atakhan_in_patch``, the ``rule_uncertain_*``
        flags): constant over the smoke patches only
    frame_age_s: the age of the frame of a view, by source
    level_lag, building_kill_matching, raw_measurement: the measurements on the raw matches (only with ``records``)
    smoke, terrain (the ``grid`` record of TERRAIN.json -- cells, cell_u, histogram_cells, histogram_u, min_count -- and the geodesic path
        between the two bases), store (the files and bytes of the stored arrays, ``Store.stored_files()``, STORE.json left out: the
        same figures as ``store_files`` and ``store_bytes`` of the s2_result that ``store_build`` writes now; a RUN.json of a build
        before the final fix pass counted STORE.json in them as well), checks, figures

Figures (PNG, matplotlib's Agg canvas, no identifier in any of them) in ``<run>\report``, listed in ``figures`` with their path relative
to the report and the numbers they show: ``fig1_occupancy_walkable`` (the FIT position occupancy, the walkable grid with the threshold
``grid.min_count`` of TERRAIN.json, its largest component and the zone boundaries), ``fig2_geodesic_path`` (one shortest path between
the two bases with its length), ``fig3_event_edge_counts`` (event tokens and edges per view: the window totals, stored + dropped, so that
the tail beyond the caps ``config.MAX_EVENTS`` and ``config.MAX_EDGES`` shows), ``fig4_frame_age`` (the age of the frame of a view, by
source).

The base anchors are the midpoints of the two nexus turrets of each team (``state.LANDMARKS``). The path is walked from the geodesic table:
from a cell, the neighbour (the same graph as the table: four neighbours, a diagonal only when both orthogonal cells are walkable) that
minimises the step plus the table distance to the target, which is a shortest path up to the rounding of the table (checked: its length is
within 1 u of the table's distance).

The raw FIT cohort size is the design spec's "about 162k matches" (``RAW_FIT_MATCHES``); it is an estimate, not a count, and ``--raw-fit-
matches`` replaces it. Its views per match are those of the smoke run (R14-c), which covers the first matches of the S1 order only; the
same estimate with the mean views per match of every FIT match of the S1 store is shown beside it as a cross-check.

Large stores are read in chunks of ``CHUNK_VIEWS`` views; no number depends on the chunk size (tested).


방법. 이 모듈은 S1–S2 Task14의 완료된 저장소를 보고 보기 크기·넘침·결측·지형을 정리한다.
보기는 분 프레임의 상태 또는 교전 첫 킬·끝 사건 시계의 입력 한 묶음이며 t₀는 그 입력이 쓰는 프레임 시각이다.
계산은 보고용 진단이고 상한·분할·학습·관문 승인 자체를 바꾸지 않는다.
공통 가정. 저장소의 행·배열·지형 계약과 비어 있지 않은 보기 표를 전제로 하며 각 함수의 실제 검사를 따로 적는다.
저장 시각은 ms이고 보고 나이는 s이며 플레이어 배열 행은 0 기반, 사건 참가자 ID는 1 기반이다.
원시 측정이 없는 경로는 파일 통계만 정리하고 측정 경로의 성공 기록만 원시 진단 분모에 포함한다.
기존 영어의 shortest-path 설명은 유효한 지형 그래프·거리표 계약 아래의 설명이며 모든 임의 입력 보증이 아니다.
식 목록. 비공개 집계의 식은 S3 R69에 따라 가장 가까운 공개 report가 소유한다.
    (REP2-1) 엄격한 넘침 비율 → over_share
    (REP2-2) 기지 앵커 → base_anchors
    (REP2-3) 표를 따라 걷는 경로 → base_path
    (REP2-4) 점유와 구역 경계 → fig_occupancy
    (REP2-5) 경로의 지도 표시 → fig_path
    (REP2-6) 상한 전 창 개수 그림 → fig_counts
    (REP2-7) 프레임 나이 그림 → fig_age
    (REP2-8) 넘침 상한 제안과 요약 → report
    (REP2-9) 열 상수와 결측 빈도 → report
    (REP2-10) 위치와 미지 항목 비율 → report
    (REP2-11) 공격력 점검과 분 시각 편차 → report
    (REP2-12) 저장 시계 대조 → report
    (REP2-13) 레벨 지연 집계 → report
    (REP2-14) 배열 크기 외삽 → report
    (REP2-15) 메모리와 기본 보고 분모 → report
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 11 (S2: the smoke run reports bytes per view, peak private memory, the
#      caps of 128 events and 128 edges, the terrain figures) and docs/superpowers/plans/2026-10-20-c-ept-s1-s2.md task 14 (this report).
# Ref: Che et al. 2018, doi:10.1038/s41598-018-24271-9, as cited in ``tokens`` (the mask that goes with a NaN: ``nan_share`` lists
#      where a NaN may stand and how often it does; the S4 loader builds the mask from ``tokens.NAN_ALLOWED``).
# Ref: LoL Wiki, "Champion statistic" (https://wiki.leagueoflegends.com/en-us/Champion_statistic), as cited in ``traits``: the growth
#      formula behind ``lvl_attackDamage``, whose sum with the item attack damage ``level_formula_check`` compares with the frame's.
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept.data import reader
from c_ept import runlog
from c_ept.domain import state
from c_ept.data import store_core
from c_ept.preprocessing import terrain, tokens, traits
from c_ept.checks import sweep_state
from c_ept.data.store_core import Store

LIMIT_PEAK_GIB = 48.0                              # the memory gate: tree-summed peak private memory
OVERFLOW_LIMIT = Fraction(1, 20)                   # more than 5 % of the views that drop events (or edges) is a finding
FULL_BUILD_MACROS = ('FIT', 'STOP', 'DEV', 'CAL', 'AUDIT')       # the roles of the full S2 build that the gate asks for
RAW_FIT_MATCHES = 162_000                          # "about 162k matches" of raw FIT in the design spec (15.18-15.20): an estimate
RAW_FIT_SOURCE = ('design spec 2026-10-20-c-ept-design.md: about 162k raw FIT matches (patches 15.18-15.20, buckets 0-79); '
                  'an approximate count of the spec, not measured here')
CHUNK_VIEWS = 8192                                 # views read at a time
GIB = 2 ** 30
SRC_NAMES = ('frame_clock', 'M3', 'M4')            # by value of views/src
VIEWS_COLUMNS = (('src', 'i1'), ('src_index', 'i4'), ('match', 'i4'), ('ref', 'i8'), ('t0', 'i8'), ('frame_k', 'i2'))      # R13-d, 27 B
TIME_TOLERANCE_S = 5e-4                            # float32 holds a time in s to 1.2e-4 up to an hour: a millisecond (1e-3) is a difference
FIG_NAMES = ('fig1_occupancy_walkable', 'fig2_geodesic_path', 'fig3_event_edge_counts', 'fig4_frame_age')
CENSUS_SOURCES = ('regular_frames', 'game_end_frame', 'kills')        # the keys of TERRAIN.json positions that suspect_positions copies
_RUN_ARGS = ('workers', 'block_matches', 'n_matches')              # the keys of RUN.json args that the report echoes ...
_RUN_INFO = ('macros', 'limit')                                    # ... and what the command line gave (args.info, or args itself)


class ReportInputError(ValueError):
    """The inputs of the report do not belong together or lack something the gate needs. The message names the field, never a match key.

    보고 입력의 결합이나 필수 필드가 맞지 않을 때 쓰는 예외이다. 메시지는 식별자 대신 필드를 가리킨다.
    """


# ---- small helpers ----------------------------------------------------------------------------------------------------------------------
def over_share(count: int, total: int) -> bool:
    """Whether ``count`` of ``total`` is more than 5 % (exact, with integers).

    수식
    ----
    (REP2-1)  flag = [20·count > total]

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      count, total | 넘침 보기 수와 전체 보기 수 | 보기 | 정수·비음수 계약, 여기서 미검사 | count, total
      flag | 엄격한 비율 초과 표시 | — | bool | 반환값

    해석
    ----
    정수 교차곱으로 S1–S2 R14-e의 엄격한 초과를 판정한다. 입력 개수의 유효성은 호출자의 계약이며 여기서는 검사하지 않는다. 두 개수가 모두 비어 영이면 False이다.

    근거
    ----
    S1–S2 계획 Task14; S1–S2 판정 R14-e(넘침 보고).
    """
    return count * OVERFLOW_LIMIT.denominator > total * OVERFLOW_LIMIT.numerator


def _spans(n: int):
    step = max(1, int(CHUNK_VIEWS))
    for a in range(0, n, step):
        yield a, min(a + step, n)


def _clean(obj):
    """``obj`` for ``json.dumps``: numpy scalars and arrays as Python numbers and lists, NaN and infinity as None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _share(count, total):
    """(REP2-9)의 계산·입력 항을 맡는다."""
    return float(count) / float(total) if total else None


def _by_src(mask: np.ndarray, src: np.ndarray) -> dict:
    """(REP2-15)의 계산·입력 항을 맡는다."""
    counts = np.bincount(src[mask], minlength=len(SRC_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(SRC_NAMES)}


def assert_no_keys(s1: Store, text: str) -> None:
    """RuntimeError if the key of any S1 match occurs in ``text`` (the report). Only the number of hits is named, never a key.

    S1 키 목록에 대한 검사로 출력 직전의 식별자 노출을 막는 단계이다. 정해진 키 검색의 통과가 모든 개인정보 부재를 보증하지는 않는다.
    """
    hits = sum(1 for k in np.asarray(s1.array('matches/key')).tolist() if k and k.decode('ascii') in text)
    if hits:
        raise RuntimeError(f'{hits} match keys occur in the report text')


def _need(doc: dict, *path: str, source: str = 'RUN.json'):
    """``doc[path[0]][path[1]]...``; ReportInputError naming the field in ``source`` (RUN.json, or TERRAIN.json) when a step is missing."""
    d = doc
    for i, step in enumerate(path):
        if not isinstance(d, dict) or step not in d:
            where = '.'.join(path[:i]) or 'the top level'
            raise ReportInputError(f'{source} has no {".".join(path[:i + 1])} (looked in {where}): the build did not record what the gate '
                                   'needs; rebuild or add the field')
        d = d[step]
    return d


# ---- the inputs -------------------------------------------------------------------------------------------------------------------------
def _read_run_json(run: Path) -> dict:
    path = run / 'RUN.json'
    if not path.is_file():
        raise FileNotFoundError(f'{run.name} has no RUN.json: this is not a finished S2 run folder')
    return json.loads(path.read_text(encoding='utf-8'))


def _run_fields(doc: dict) -> dict:
    """The fields of RUN.json that the report reads, as one dict. ReportInputError for a missing field or a build that did not finish."""
    result = _need(doc, 's2_result')
    if result.get('complete') is not True:
        raise ReportInputError('RUN.json s2_result.complete is not true: the build did not finish, so there is nothing to report')
    out = {name: _need(doc, 's2_result', name) for name in ('views', 'views_by_src', 'matches', 'bytes_per_view', 'seconds_per_match',
                                                           'peak_private_gib')}
    out.update({name: result.get(name) for name in ('peak_private_main_gib', 'peak_private_worker_gib', 'wall_s', 'workers',
                                                    'x_bytes_per_view', 'views_bytes_per_view', 'calls', 'recovered', 'per_call',
                                                    'code_state_at_end')})
    out['warnings'] = [str(w) for w in (result.get('warnings') or [])]
    args = doc.get('args') if isinstance(doc.get('args'), dict) else {}
    info = args['info'] if isinstance(args.get('info'), dict) else args           # the command line's info: one key of args, or args itself
    out['args'] = {**{k: args[k] for k in _RUN_ARGS if k in args}, **{k: info[k] for k in _RUN_INFO if k in info}}
    out['s1_run'] = _need(doc, 's1', 'run')
    out['s1_files_sha256'] = _need(doc, 's1', 'files_sha256')
    out['terrain_sha256'] = _need(doc, 'terrain', 'sha256')
    return out


def _layout(n: int) -> dict:
    """{array name: (shape, dtype)} of an S2 store with ``n`` views, from the rulings (R13-d) and ``tokens.VIEW_SPEC``."""
    out = {f'views/{c}': ((n,), np.dtype(d)) for c, d in VIEWS_COLUMNS}
    out.update({f'x/{key}': ((n, *shape), np.dtype(dtype)) for key, (shape, dtype) in tokens.VIEW_SPEC.items()})
    return out


def _open_s2(run: Path, verify: bool) -> Store:
    """The S2 store of the run, complete, with the layout of ``VIEW_SPEC`` (ReportInputError otherwise) and, with ``verify``, every file
    hashed against STORE.json (ValueError)."""
    root = run / 's2'
    if not (root / 'STORE.json').is_file():
        raise FileNotFoundError(f'{run.name} has no s2\\STORE.json')
    s = Store(root)
    if not s.manifest.get('complete'):
        raise ReportInputError(f'the S2 store of {run.name} is not complete (STORE.json says complete is false): '
                               'it is not a finished build')
    arrays = s.manifest.get('arrays', {})
    if 'views/src' not in arrays:
        raise ReportInputError('the S2 store has no views/src array')
    n = int(arrays['views/src']['shape'][0])
    if n < 1:
        raise ReportInputError('the S2 store holds no view')
    want = _layout(n)
    if set(arrays) != set(want):
        raise ReportInputError(f'the S2 store holds the arrays {sorted(set(arrays) ^ set(want))} that tokens.VIEW_SPEC and the views table '
                               'do not (or lacks them): it was built with another layout')
    for name, (shape, dtype) in want.items():
        have = (tuple(arrays[name]['shape']), np.dtype(arrays[name]['dtype']))
        if have != (shape, dtype):
            raise ReportInputError(f'{name}: the store has {have[0]} {have[1].str}, the layout of tokens.VIEW_SPEC is {shape} {dtype.str}')
    if verify:
        try:
            s.verify()
        except ValueError as e:                                    # a file differs from STORE.json, or a file was added or removed
            raise ReportInputError(str(e)) from None
    for name, (shape, dtype) in want.items():                      # the file headers must say the same as the manifest
        a = s.array(name)
        if (a.shape, a.dtype) != (shape, dtype):
            raise ReportInputError(f'{name}: the file holds {a.shape} {a.dtype.str}, the layout is {shape} {dtype.str}')
    return s


def _open_s1(s1_arg, run: Path, fields: dict) -> tuple[Store, Path]:
    """(the S1 store, the S1 run folder). The folder is ``s1_arg`` (a run folder, or its ``s1`` store folder) or, by default, the folder
    named in RUN.json next to the S2 run folder. ReportInputError unless its STORE.json is complete and is the one the build consumed."""
    path = Path(s1_arg) if s1_arg is not None else run.parent / fields['s1_run']
    folder = path.parent if (path / 'STORE.json').is_file() else path
    if not (folder / 's1' / 'STORE.json').is_file():
        raise FileNotFoundError(f'the S1 run {folder.name} has no s1\\STORE.json: give --s1 (RUN.json names {fields["s1_run"]})')
    s1 = Store(folder / 's1')
    if not s1.manifest.get('complete'):
        raise ReportInputError(f'the S1 store of {folder.name} is not complete')
    if runlog.sha256_json(s1.manifest['files']) != fields['s1_files_sha256']:
        raise ReportInputError(f'the S1 store of {folder.name} is not the S1 store this S2 run consumed (the SHA-256 of its files map '
                               'differs from s1.files_sha256 in RUN.json)')
    return s1, folder


def _open_terrain(terrain_arg, run: Path, fields: dict) -> tuple:
    """(Terrain, folder). Refused (ReportInputError, before any figure is drawn): a folder without TERRAIN.json, two data files that do
    not belong together (``Terrain`` itself refuses that), files that differ from TERRAIN.json (``Terrain.verify``), hashes that differ
    from the ones RUN.json recorded, and a TERRAIN.json that lacks a field the report reads later: the hashes, the walkable threshold
    ``grid.min_count`` (the legend of figure 1 and the terrain section) and the position census (``suspect_positions``). The terrain is
    reported as it was built, also when its parameters are not the ones of the present code: the build refuses such a terrain, the
    report of an older run shows what that run used."""
    path = Path(terrain_arg) if terrain_arg is not None else run / 'terrain'
    if not (path / 'terrain.npz').is_file():
        raise FileNotFoundError(f'there is no terrain folder {path.name} (terrain.npz) in {path.parent.name}: give --terrain')
    try:
        terr = terrain.Terrain(path)
    except ValueError as e:
        raise ReportInputError(f'the terrain folder {path.name} cannot be used: {e}') from None
    if terr.manifest is None:
        raise ReportInputError(f'the terrain folder {path.name} has no TERRAIN.json: a terrain without provenance is not reported')
    _need(terr.manifest, 'sha256', source='TERRAIN.json')
    try:
        now = {**terr.verify(), 'TERRAIN.json': store_core.sha(path / 'TERRAIN.json')}
    except ValueError as e:
        raise ReportInputError(f'the terrain folder {path.name}: {e}') from None
    if now != fields['terrain_sha256']:
        differ = sorted(k for k in now if now.get(k) != fields['terrain_sha256'].get(k))
        raise ReportInputError(f'the terrain folder {path.name} is not the terrain this S2 run recorded (terrain.sha256 in RUN.json '
                               f'differs from the SHA-256 of {differ})')
    _need(terr.manifest, 'grid', 'min_count', source='TERRAIN.json')
    for name in CENSUS_SOURCES:
        _need(terr.manifest, 'positions', name, source='TERRAIN.json')
    return terr, path


def _check_output(path: Path, root: Path, what: str, s1_folder: Path) -> Path:
    """``path`` unless it would lie inside the S2 store (its files are hashed) or the S1 run (read only): ReportInputError."""
    p = Path(path).resolve()
    if p.is_relative_to(root.resolve()):
        raise ReportInputError(f'{what} {Path(path).name} would lie inside the S2 store {root.name}: the store hashes every file under it')
    if p.is_relative_to(s1_folder.resolve()):
        raise ReportInputError(f'{what} {Path(path).name} would lie inside the S1 run folder {s1_folder.name}: an S1 run is read only')
    return Path(path)


def _relative(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` with forward slashes (the folder name alone when there is no relative path, such as across drives)."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return Path(path).name


# ---- one pass over the arrays -----------------------------------------------------------------------------------------------------------
def _column_stats(flat: np.ndarray, acc: dict) -> None:
    """Add the NaN and infinity counts and the NaN-ignoring minimum and maximum of every column of ``flat`` (rows x columns) to ``acc``.

    (REP2-9)의 계산·입력 항을 맡는다.
    """
    acc['nan'] += np.isnan(flat).sum(axis=0)
    acc['inf'] += np.isinf(flat).sum(axis=0)
    acc['min'] = np.fmin(acc['min'], np.fmin.reduce(flat, axis=0))
    acc['max'] = np.fmax(acc['max'], np.fmax.reduce(flat, axis=0))
    acc['rows'] += len(flat)


def _new_column_stats(columns: int) -> dict:
    return {'nan': np.zeros(columns, np.int64), 'inf': np.zeros(columns, np.int64), 'min': np.full(columns, np.inf),
            'max': np.full(columns, -np.inf), 'rows': 0}


def _constant_columns(acc: dict, names: tuple) -> dict:
    """{name: value} of the columns with one value in every row (None for NaN in every row).

    (REP2-9)의 계산·입력 항을 맡는다.
    """
    out = {}
    if acc['rows'] == 0:
        return out
    for i, name in enumerate(names):
        if acc['nan'][i] == acc['rows']:
            out[name] = None
        elif acc['nan'][i] == 0 and acc['min'][i] == acc['max'][i]:
            out[name] = float(acc['min'][i])
    return out


def _scan_players(arr, table: dict) -> dict:
    """One pass over ``x/players_num``: NaN, infinity, minimum and maximum per column; the positions that are missing, at (0, 0) or off the
    map; the unknown champions and items; the level formula on the frame-clock views.

    (REP2-10, REP2-11)의 계산·입력 항을 맡는다.
    """
    P = {name: i for i, name in enumerate(tokens.PLAYER_NUM)}
    n, src = len(arr), table['src']
    acc = _new_column_stats(len(tokens.PLAYER_NUM))
    pos = Counter()
    pos_by_src = Counter()
    diffs, used_views = [], 0
    unknown = Counter()
    side = float(C.W)
    for a, b in _spans(n):
        block = np.asarray(arr[a:b])                                     # (k, 10, 81) float32
        s = src[a:b]
        _column_stats(block.reshape(-1, block.shape[-1]), acc)
        x, y = block[:, :, P['x']], block[:, :, P['y']]
        with np.errstate(invalid='ignore'):
            missing = np.isnan(x) | np.isnan(y)
            zero = ~missing & (x == 0) & (y == 0)
            off = ~missing & ((x < 0) | (x > side) | (y < 0) | (y > side))
        pos['nan_rows'] += int(missing.sum())
        pos['zero_zero_rows'] += int(zero.sum())
        pos['off_map_rows'] += int(off.sum())
        vz, vo = zero.any(axis=1), off.any(axis=1)
        pos['views_with_zero_zero'] += int(vz.sum())
        pos['views_with_off_map'] += int(vo.sum())
        pos['views_with_any'] += int((vz | vo).sum())
        pos_by_src.update(_by_src(vz | vo, s))
        frame = block[s == 0]                                            # the frame-clock views
        # (REP2-11): 저장 float32 값의 잔차 산술은 float64로 올려 누적한다.
        d = (frame[:, :, P['lvl_attackDamage']].astype(np.float64) + frame[:, :, P['item_attackDamage']]
             - frame[:, :, P['cs_attackDamage']])
        finite = np.isfinite(d)
        # an item that the item table lacks adds nothing to item_attackDamage: that row's difference is short by its attack damage, so the
        # row is left out and counted (apart from the NaN rows of unknown champions: no row is counted twice)
        short_item = finite & (frame[:, :, P['unknown_item_count']] > 0)
        diffs.append(d[finite & ~short_item])
        used_views += len(frame)
        unknown['nan_excluded'] += int((~finite).sum())
        unknown['unknown_item_excluded'] += int(short_item.sum())
        champ = block[:, :, P['champ_unknown']] == 1
        item = block[:, :, P['unknown_item_count']]
        has_item = item > 0
        unknown['rows_with_unknown_champion'] += int(champ.sum())
        unknown['views_with_unknown_champion'] += int(champ.any(axis=1).sum())
        unknown['rows_with_unknown_item'] += int(has_item.sum())
        unknown['views_with_unknown_item'] += int(has_item.any(axis=1).sum())
        unknown['unknown_items_total'] += int(round(float(item[has_item].astype(np.float64).sum())))
    return {'columns': acc, 'positions': dict(pos), 'positions_by_src': {k: int(pos_by_src[k]) for k in SRC_NAMES},
            'diffs': np.concatenate(diffs) if diffs else np.zeros(0), 'frame_views': used_views, 'unknown': dict(unknown)}


def _scan_summary(arr, table: dict) -> dict:
    """One pass over ``x/summary``: NaN, infinity, minimum and maximum per column, and how far the two clock columns are from the table.

    (REP2-12)의 계산·입력 항을 맡는다.
    """
    acc = _new_column_stats(len(tokens.SUMMARY))
    age_col, time_col = tokens.SUMMARY.index('frame_age_s'), tokens.SUMMARY.index('game_time_s')
    age_dev = time_dev = 0.0
    for a, b in _spans(len(arr)):
        block = np.asarray(arr[a:b])
        _column_stats(block, acc)
        ref, t0 = table['ref'][a:b], table['t0'][a:b]
        age_dev = max(age_dev, float(np.abs(block[:, age_col].astype(np.float64) - (ref - t0) / 1000.0).max()))
        time_dev = max(time_dev, float(np.abs(block[:, time_col].astype(np.float64) - ref / 1000.0).max()))
    return {'columns': acc, 'age_deviation_s': age_dev, 'time_deviation_s': time_dev}


def _scan_cells(arr) -> dict:
    """NaN and infinity counts per cell of the trailing axes of ``arr`` (views x ...), summed over the views, and the number of views.

    (REP2-9)의 계산·입력 항을 맡는다.
    """
    nan = np.zeros(arr.shape[1:], np.int64)
    inf = np.zeros(arr.shape[1:], np.int64)
    for a, b in _spans(len(arr)):
        block = np.asarray(arr[a:b])
        nan += np.isnan(block).sum(axis=0)
        inf += np.isinf(block).sum(axis=0)
    return {'nan': nan, 'inf': inf, 'views': len(arr)}


def _landmark_cell_name(row: int) -> str:
    lm = state.LANDMARKS[row]
    return f'{lm["kind"]}[{lm["slot"]}]'


def _nan_section(n: int, players: dict, summary: dict, cells: dict) -> dict:
    """The ``nan_share`` section: the share of NaN at every cell that ``tokens.NAN_ALLOWED`` allows, and the NaN or infinity outside it.

    (REP2-9)의 계산·입력 항을 맡는다.
    """
    allowed = tokens.NAN_ALLOWED
    out = {'views': n, 'player_rows': 10 * n}
    outside = []

    def add(array, cell, kind, count):
        if count:
            outside.append({'array': array, 'cell': cell, 'kind': kind, 'count': int(count)})

    for key, names, acc, rows in (('players_num', tokens.PLAYER_NUM, players['columns'], 10 * n),
                                  ('summary', tokens.SUMMARY, summary['columns'], n)):
        ok = allowed[key]['cells']
        if not isinstance(ok, tuple):
            raise ReportInputError(f'tokens.NAN_ALLOWED[{key!r}]["cells"] is {ok!r}: this report reads a tuple of column names there')
        out[key] = {name: _share(acc['nan'][names.index(name)], rows) for name in ok}
        for i, name in enumerate(names):
            if name not in ok:
                add(key, name, 'nan', acc['nan'][i])
            add(key, name, 'inf', acc['inf'][i])
    lm = cells['landmarks']
    ok_cells = {tuple(c) for c in allowed['landmarks']['cells']}
    out['landmark_seconds'] = {_landmark_cell_name(r): _share(lm['nan'][r, c], n) for r, c in sorted(ok_cells)}
    for r in range(lm['nan'].shape[0]):
        for c in range(lm['nan'].shape[1]):
            if (r, c) not in ok_cells:
                add('landmarks', f'row {r}, column {c}', 'nan', lm['nan'][r, c])
            add('landmarks', f'row {r}, column {c}', 'inf', lm['inf'][r, c])
    for key, entry in allowed.items():
        if key in ('players_num', 'summary', 'landmarks'):
            continue
        if entry['cells'] != 'all':
            raise ReportInputError(f'tokens.NAN_ALLOWED[{key!r}]["cells"] is {entry["cells"]!r}: this report reads "all" there')
    for key in tokens.VIEW_SPEC:
        if key in cells and key != 'landmarks':
            c = cells[key]
            total = int(c['nan'].sum())
            if key in allowed:
                out[key] = _share(total, int(c['views'] * c['nan'].size))
            else:
                add(key, 'all', 'nan', total)
            add(key, 'all', 'inf', int(c['inf'].sum()))
    out['outside_allowed'] = outside
    out['ok'] = not outside
    return out


# ---- the raw-match measurements (only with records) --------------------------------------------------------------------------------------
def _level_lag_counts(rec, refs: np.ndarray, stored: np.ndarray) -> Counter:
    """The frame-clock cells of one match whose stored level differs from ``traits.LevelIndex.at(ref)``: all of them, those on the ms of
    a LEVEL_UP of that player, and those that are the S2-8 case (the frame shows one level less than the clock while a LEVEL_UP of the
    player falls on the ms of the frame). ``refs`` the reference times of the stored frame-clock views, ``stored`` their (k, 10) levels.

    (REP2-13)의 계산·입력 항을 맡는다.
    """
    index = traits.LevelIndex(rec)
    ups = {(e['participantId'], e['timestamp']) for e in rec.events if e.get('type') == 'LEVEL_UP'}
    c = Counter()
    for ref, row in zip(refs.tolist(), stored):
        clock = index.at(ref)
        c['frame_clock_cells'] += len(row)
        for p in np.flatnonzero(~(clock == row)).tolist():
            on_ms = (p + 1, ref) in ups
            c['differ'] += 1
            c['differ_on_same_ms_level_up'] += on_ms
            c['differ_lag_by_one_at_level_up_ms'] += bool(on_ms and row[p] == int(clock[p]) - 1)
    return c


def _measure(s1: Store, table: dict, players, records, smoke: np.ndarray) -> dict:
    """The level lag and the building-kill counters over the smoke matches, one raw match at a time.

    The E6 check comes first (``run_s1.StrataRunning`` while a model server runs): ``report(records=...)`` reads raw matches, and only the
    command line asked for the check, so a library caller was not protected. A match that cannot be read or has a broken format (the
    reader's ``MatchRejected``, ``MatchFormatError`` and ``MatchReadError``, and a ``ValueError`` of the record format) is counted by the
    class of its error and the place in the code, never by its message (which can hold the key), and is left out of both. Any other error
    is a bug or a broken machine and stops the report.


    (REP2-13)의 계산·입력 항을 맡는다.
    """
    from c_ept.workflows.run_s1 import require_no_strata         # here, not at module level: a report that reads files only needs no runner chain
    require_no_strata()
    keys, patches = np.asarray(s1.array('matches/key')), np.asarray(s1.array('matches/patch'))
    frame_views = np.flatnonzero(table['src'] == 0)
    order = np.argsort(table['match'][frame_views], kind='stable')
    sorted_match = table['match'][frame_views][order]
    lo, hi = np.searchsorted(sorted_match, smoke, side='left'), np.searchsorted(sorted_match, smoke, side='right')
    level_col = tokens.PLAYER_NUM.index('level')
    lag, sweep, errors, measured = Counter(), Counter({k: 0 for k in sweep_state.COUNT_KEYS}), Counter(), 0
    for m, a, b in zip(smoke.tolist(), lo.tolist(), hi.tolist()):
        patch = C.PATCHES[int(patches[m])]
        try:
            rec = records(keys[m].decode('ascii'), patch)
            idx = frame_views[order[a:b]]
            part_lag = _level_lag_counts(rec, table['ref'][idx], np.asarray(players[idx])[:, :, level_col])
            part_sweep = sweep_state.count_record(rec)
        except (reader.MatchRejected, reader.MatchFormatError, reader.MatchReadError, ValueError) as e:   # the measurement goes on and reports it
            where = traceback.extract_tb(e.__traceback__)[-1]
            errors[f'{type(e).__name__} at line {where.lineno} in {where.name} of {Path(where.filename).name} (patch {patch})'] += 1
            continue
        lag.update(part_lag)
        sweep.update(part_sweep)
        measured += 1
    nan_cells = {f: sweep.pop(sweep_state.NAN_PREFIX + f, 0) for f in state.NAN_FEATURES}
    for k in [k for k in sweep if k.startswith(sweep_state.NAN_PREFIX)]:
        del sweep[k]
    cells, differ = lag['frame_clock_cells'], lag['differ']
    return {
        'raw_measurement': {'measured': True, 'matches_requested': len(smoke), 'matches_measured': measured, 'errors': dict(errors)},
        'level_lag': {
            'measured': True, 'matches': measured, 'frame_clock_cells': cells, 'differ': differ, 'share_differ': _share(differ, cells),
            'differ_on_same_ms_level_up': lag['differ_on_same_ms_level_up'],
            'differ_lag_by_one_at_level_up_ms': lag['differ_lag_by_one_at_level_up_ms'],
            'differ_unexplained': differ - lag['differ_lag_by_one_at_level_up_ms'],
            'note': 'frame-clock cells: the level stored in a frame-clock view against traits.LevelIndex.at(ref); the S2-8 case is a '
                    'LEVEL_UP of the player on the ms of the frame while the frame still shows one level less'},
        'building_kill_matching': {
            'measured': True, 'matches': measured, 'counts': {k: sweep[k] for k in sweep_state.COUNT_KEYS},
            'objective_nan_cells_by_feature': nan_cells,
            'note': 'the counters of c_ept.checks.sweep_state (state.building_kills and the state sweep) over the smoke matches'},
    }


def _unmeasured(matches: int) -> dict:
    reason = ('not measured: no records function was given, so no raw match was read (the command line measures them unless '
              '--no-matches is given)')
    return {'raw_measurement': {'measured': False, 'matches_requested': matches, 'matches_measured': 0, 'errors': {}},
            'level_lag': {'measured': False, 'reason': reason}, 'building_kill_matching': {'measured': False, 'reason': reason}}


# ---- the geodesic path between the bases -------------------------------------------------------------------------------------------------
def base_anchors() -> tuple:
    """((x, y) of the blue base, (x, y) of the red base): the midpoint of the two nexus turrets of each team in ``state.LANDMARKS``.

    수식
    ----
    (REP2-2)  anchor_team = (turret_xy_{team,0} + turret_xy_{team,1}) / 2

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      team | 진영 코드 | — | 100, 200 | team
      turret_xy, anchor_team | 넥서스 포탑 좌표와 그 중점 | 지도 단위 | 각 진영 포탑 두 개를 검사 | pts, out

    해석
    ----
    S1–S2 R14-d의 두 기지 경로를 표시할 앵커를 포탑 중점으로 정한다. 포탑 수를 확인하지만 좌표의 유한성은 지형 입력 계약에 맡긴다.

    근거
    ----
    S1–S2 판정 R14-d(지형 그림); state.LANDMARKS(앵커 원천).
    """
    out = []
    for team in (100, 200):
        pts = [(lm['x'], lm['y']) for lm in state.LANDMARKS if lm['tower'] == 'NEXUS_TURRET' and lm['team'] == team]
        if len(pts) != 2:
            raise ReportInputError(f'state.LANDMARKS has {len(pts)} nexus turrets for team {team}, not 2')
        out.append((sum(p[0] for p in pts) / 2.0, sum(p[1] for p in pts) / 2.0))
    return tuple(out)


def base_path(terr) -> dict:
    """A shortest path between the two bases on the walkable grid, walked from the geodesic table.

    ``cells`` the (ix, iy) of every cell from the blue to the red base anchor (each snapped to the walkable grid: ``Terrain.snap``),
    ``start_xy`` and ``end_xy`` the anchors, ``length_u`` the sum of the steps (200 u, 200 sqrt 2 u on a diagonal), ``table_distance_u``
    the table's distance between the two cells (whole units), ``straight_u`` the straight line between the two cell centres, ``ratio``
    their quotient, ``n_cells``. From each cell the step goes to the neighbour (four neighbours; a diagonal only when both orthogonal
    cells are walkable, as in the graph of the table) with the smallest step plus table distance to the target; a tie goes to the
    smaller cell number. The table is rounded to whole units, so this is a shortest path up to that rounding: ``length_u`` is within 1 u
    of ``table_distance_u`` (a RuntimeError otherwise).


    수식
    ----
    (REP2-3)  next_cell = argmin_{legal_cell} (step_length + distance_{legal_cell,target}, legal_cell)
             path_length = Σ_step step_length; centre(cell) = 200·(cell mod 75 + 0.5, cell // 75 + 0.5)
             straight_length = ‖centre(start_cell) − centre(target)‖₂
             path_ratio = path_length / straight_length (straight_length > 0), 아니면 None

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      cell, legal_cell, start_cell, target | 평탄화 칸 번호와 합법 이웃 | — | 0 기반, 0..5624 | here, there, a, b
      distance | 반올림된 표 거리 | 지도 단위 | uint16, 65535는 도달 불가 | table
      step_length | 직교 또는 대각 보폭 | 지도 단위 | 200 또는 200·√2 | step
      path_length, straight_length, path_ratio | 누적 경로·직선 길이·비 | 지도 단위; — | 길이 비음수, 비 None 가능 | length, straight, ratio
      centre | 격자 칸의 중심 좌표 | 지도 단위 | ix=cell mod 75, iy=cell // 75 | centre

    해석
    ----
    설계서 §11 지형 그래프와 일치하는 표를 전제로 하며 모서리를 가르는 대각 이동은 제외한다. 동점은 평탄화 칸 번호로 풀고 다음 칸의 잔여 표 거리가 엄격히 감소해야 한다. 도달 불가 표 값은 거부하고 누적 보폭과 직접 표 거리의 차이는 1 지도 단위 이하여야 한다. 이 검사는 임의의 잘못된 표에도 최단 경로임을 보증하지 않는다. 두 중심이 같으면 비는 None이다.

    근거
    ----
    설계서 §11(지형); S1–S2 판정 R11-b(그래프), S2-4(거리 저장), R14-d(경로 그림).
    """
    (xa, ya), (xb, yb) = base_anchors()
    a, b = int(terr.snap(xa, ya)), int(terr.snap(xb, yb))
    table, walk, grid = terr.geodesic, terr.walk, terrain.GRID
    # (REP2-3): 저장 거리표의 반올림·도달 불가 계약을 경로 검산에 사용한다.
    direct = int(table[a, b])
    if direct == terrain.UNREACHABLE:
        raise ReportInputError('the two base anchors lie in different components of the walkable grid')
    diagonal = terrain.CELL * math.sqrt(2.0)
    cells, here, length = [a], a, 0.0
    # (REP2-3)의 다음 칸: 보폭과 잔여 거리의 합을 줄이고 동점은 칸 번호로 푼다.
    while here != b:
        iy, ix = divmod(here, grid)
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = ix + dx, iy + dy
                if (dx == 0 and dy == 0) or not (0 <= nx < grid and 0 <= ny < grid) or not walk[ny, nx]:
                    continue
                if dx and dy and not (walk[iy, nx] and walk[ny, ix]):    # no cutting of corners
                    continue
                there = grid * ny + nx
                rest = int(table[there, b])
                if rest == terrain.UNREACHABLE:
                    continue
                step = diagonal if dx and dy else terrain.CELL
                if best is None or (step + rest, there) < best[:2]:
                    best = (step + rest, there, step, rest)
        if best is None or best[3] >= int(table[here, b]):
            raise RuntimeError('the geodesic table has no step that gets closer to the target: it does not belong to the walkable grid')
        here, length = best[1], length + best[2]
        cells.append(here)
    if abs(length - direct) > 1.0:
        raise RuntimeError(f'the walked path is {length:.2f} u long, the table says {direct} u: the path is not a shortest path')
    centre = lambda c: (terrain.CELL * (c % grid + 0.5), terrain.CELL * (c // grid + 0.5))          # noqa: E731
    (ca, cb) = centre(a), centre(b)
    straight = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    return {'cells': [(c % grid, c // grid) for c in cells], 'start_xy': (xa, ya), 'end_xy': (xb, yb), 'length_u': length,
            'table_distance_u': direct, 'straight_u': straight, 'ratio': length / straight if straight else None, 'n_cells': len(cells)}


# ---- the figures ---------------------------------------------------------------------------------------------------------------------
def _figure(width: float, height: float):
    """A matplotlib Figure on the Agg canvas: no pyplot, no global state."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), layout='constrained')
    FigureCanvasAgg(fig)
    return fig


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=130)


ZONE_LABELS = {'base_blue': (0.12, 0.12), 'base_red': (0.88, 0.88), 'top': (0.065, 0.55), 'bot': (0.55, 0.065), 'mid': (0.35, 0.35),
               'river': (0.30, 0.70), 'jg_blue_top': (0.22, 0.42), 'jg_blue_bot': (0.42, 0.22), 'jg_red_top': (0.58, 0.78),
               'jg_red_bot': (0.78, 0.58)}               # (x_fraction, y_fraction) of a point inside each zone (checked against the rules when drawn)


def fig_occupancy(terr, path: Path) -> dict:
    """Figure 1: the FIT position occupancy (frames and kills, 100 u cells), the outline of the walkable 200 u grid (white), its largest
    component (green outline; walkable cells outside it filled white) and the zone boundaries with their names.

    수식
    ----
    (REP2-4)  display_{iy,ix} = log10(1 + frames100_{ix,iy} + kills100_{ix,iy})
             boundary_{iy,ix} = [축 이웃 중 zone이 다른 곳이 있음]
             grid_position_slot = slot·map_side/375; grid_edge = grid_position ± (map_side/375)/2
             outside_cells = walk_cells − main_cells; label_xy = map_side·(x_fraction, y_fraction)

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ix, iy, slot | 가로·세로 칸과 경계 표본 위치 | — | 0 기반; slot=0..375 | occupancy, g
      frames100, kills100, display | 프레임·킬 위치 수와 표시값 | 개; — | 유효 비음수 빈도 계약 | terr.frames100, terr.kills100, shown
      zone, boundary | 구역 코드와 양쪽 경계 표시 | — | 이웃의 양 끝을 표시 | zone, boundary
      map_side, grid_position, grid_edge | 구역 지도 폭·표본·반 칸 바깥 경계 | 지도 단위 | 양의 지도 폭 | side, g, grid_edges
      walk_cells, main_cells, outside_cells | 보행·최대 성분·그 밖 칸 수 | 칸 | main⊆walk 계약 | n_walk, n_main, 반환 필드
      x_fraction, y_fraction, label_xy | 구역 이름의 정규화 위치와 지도 좌표 | —; 지도 단위 | 미리 정한 위치, 구역 재확인 | ZONE_LABELS, u, v

    해석
    ----
    S1–S2 R14-d의 점유 그림은 빈 칸도 표현하도록 로그 변환하고 저장 빈도축을 영상축으로 전치한다. 위치 빈도의 칸은 100 지도 단위이고 보행 격자의 칸은 200 지도 단위이다. 구역은 양 끝을 포함한 격자에서 평가하며 코드가 달라지는 양쪽 점을 경계로 칠한다. 최대 성분이 보행 격자의 부분집합이라는 지형 계약 아래 바깥 칸 수를 뺄셈으로 표시한다.

    근거
    ----
    S1–S2 판정 R14-d(그림), R11-a(구역), S2-9(위치); terrain.zone_of(구역 규칙).
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = _figure(7.6, 7.2)
    ax = fig.subplots()
    limit, grid = terrain.MAP_LIMIT, terrain.GRID
    # (REP2-4): 저장 가로·세로 축을 영상 세로·가로 축으로 바꿔 빈 칸도 로그로 표시한다.
    occupancy = (terr.frames100 + terr.kills100).T                       # the histograms are [ix, iy]
    shown = ax.imshow(np.log10(1.0 + occupancy), origin='lower', extent=(0, limit, 0, limit), cmap='magma', interpolation='nearest')
    fig.colorbar(shown, ax=ax, shrink=0.76, label='log10 (1 + positions per 100 u cell)')
    edges = np.arange(grid + 1) * terrain.CELL
    centres = edges[:-1] + terrain.CELL / 2.0
    walk, main = terr.walk, terr.main
    islands = walk & ~main
    if islands.any():
        ax.pcolormesh(edges, edges, np.ma.masked_where(~islands, islands), cmap=ListedColormap(['#f5f5f5']), alpha=0.95)
    if walk.any() and not walk.all():                                    # contour needs both walkable and not walkable cells
        ax.contour(centres, centres, walk.astype(float), levels=[0.5], colors='white', linewidths=0.6, alpha=0.85)
    if main.any() and not main.all():
        ax.contour(centres, centres, main.astype(float), levels=[0.5], colors='#7CFC00', linewidths=1.5)
    side = float(C.W)
    g = np.linspace(0.0, side, 376)
    gx, gy = np.meshgrid(g, g)
    zone = terrain.zone_of(gx, gy)                                       # [iy, ix]
    # (REP2-4)의 경계는 서로 다른 두 구역의 양쪽 표본에 표시한다.
    boundary = np.zeros(zone.shape, dtype=bool)
    differs_x, differs_y = zone[:, :-1] != zone[:, 1:], zone[:-1, :] != zone[1:, :]
    boundary[:, :-1] |= differs_x
    boundary[:, 1:] |= differs_x
    boundary[:-1, :] |= differs_y
    boundary[1:, :] |= differs_y
    step = g[1] - g[0]
    grid_edges = np.concatenate([g - step / 2.0, [g[-1] + step / 2.0]])
    ax.pcolormesh(grid_edges, grid_edges, np.ma.masked_where(~boundary, boundary), cmap=ListedColormap(['#00e5ff']), alpha=0.85)
    for name, (u, v) in ZONE_LABELS.items():
        if int(terrain.zone_of(u * side, v * side)) == terrain.ZONES.index(name):
            ax.text(u * side, v * side, name, color='white', fontsize=7, ha='center', va='center',
                    bbox={'facecolor': 'black', 'alpha': 0.45, 'pad': 1.5, 'linewidth': 0})
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal')
    ax.set_xlabel('x (map units)')
    ax.set_ylabel('y (map units)')
    n_walk, n_main = int(walk.sum()), int(main.sum())
    ax.set_title(f'FIT positions, walkable grid (75 x 75, 200 u) and zones\n{n_walk:,} walkable cells; largest component {n_main:,}, '
                 f'outside it {n_walk - n_main:,}', fontsize=10)
    min_count = _need(terr.manifest, 'grid', 'min_count', source='TERRAIN.json')        # the threshold of the terrain that was built
    ax.legend(handles=[Line2D([], [], color='white', lw=0.8, label=f'walkable area (at least {min_count} positions per 200 u cell)'),
                       Line2D([], [], color='#7CFC00', lw=1.5, label='largest component'),
                       Patch(facecolor='#f5f5f5', edgecolor='black', label='walkable cells outside it'),
                       Patch(facecolor='#00e5ff', label='zone boundary')], loc='upper center', fontsize=7, ncol=2,
              bbox_to_anchor=(0.5, -0.09), frameon=True, facecolor='#444444', labelcolor='white')
    _save(fig, path)
    return {'walkable_cells': n_walk, 'largest_component_cells': n_main, 'cells_outside_largest': n_walk - n_main,
            'zones': list(terrain.ZONES)}


def fig_path(terr, info: dict, path: Path) -> dict:
    """Figure 2: the walkable grid with one shortest path between the two bases and the straight line, with the lengths.

    수식
    ----
    (REP2-5)  path_xy = 200·(ix + 0.5, iy + 0.5)
             grid_kind = 2 (main), 1 (walk이지만 main 아님), 0 (그 밖)
             ratio_text = format(path_ratio, 소수 셋째 자리) (truthy), 아니면 'n/a'

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ix, iy, path_xy | 경로 칸과 그 중심 | —; 지도 단위 | 0 기반, 비어 있지 않은 경로 계약 | info['cells'], xs, ys
      main, walk, grid_kind | 최대 성분·보행 표시와 그림 코드 | — | bool; 0,1,2 | terr.main, terr.walk, kind
      path_ratio, ratio_text | 경로 대 직선 비와 표시 문자열 | — | None 또는 수, 0도 n/a | info['ratio'], ratio

    해석
    ----
    S1–S2 R14-d의 경로와 직선은 같은 칸 중심을 잇는다. 길이와 표 거리는 정수 자리로 표시하지만 원래 수치는 반환 사전에 유지한다. 그림의 비 문자열은 소수 셋째 자리로 반올림하고 거짓으로 평가되는 값은 결측 표시를 쓴다.

    근거
    ----
    S1–S2 판정 R14-d(경로 그림); base_path(경로 계약).
    """
    from matplotlib.colors import ListedColormap
    fig = _figure(7.2, 7.2)
    ax = fig.subplots()
    grid = terrain.GRID
    edges = np.arange(grid + 1) * terrain.CELL
    kind = np.where(terr.main, 2, np.where(terr.walk, 1, 0))
    ax.pcolormesh(edges, edges, kind, cmap=ListedColormap(['#2b2b2b', '#6fa8dc', '#ececec']), vmin=0, vmax=2)
    xs = [terrain.CELL * (ix + 0.5) for ix, _ in info['cells']]
    ys = [terrain.CELL * (iy + 0.5) for _, iy in info['cells']]
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color='black', ls='--', lw=1.0, label=f'straight line {info["straight_u"]:,.0f} u')
    ax.plot(xs, ys, color='#d62728', lw=2.2, label=f'shortest path {info["length_u"]:,.0f} u ({info["n_cells"]} cells)')
    ax.scatter([xs[0]], [ys[0]], s=70, color='#1f77b4', edgecolor='white', zorder=5, label='blue base')
    ax.scatter([xs[-1]], [ys[-1]], s=70, color='#ff7f0e', edgecolor='white', zorder=5, label='red base')
    ax.set_xlim(0, terrain.MAP_LIMIT)
    ax.set_ylim(0, terrain.MAP_LIMIT)
    ax.set_aspect('equal')
    ax.set_xlabel('x (map units)')
    ax.set_ylabel('y (map units)')
    ratio = f'{info["ratio"]:.3f}' if info['ratio'] else 'n/a'
    ax.set_title(f'Geodesic path between the two bases: {info["length_u"]:,.0f} u\ntable distance {info["table_distance_u"]:,} u; '
                 f'path / straight line {ratio}', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    _save(fig, path)
    return {k: v for k, v in info.items() if k != 'cells'}


def fig_counts(ev_n: np.ndarray, ev_dropped: np.ndarray, ed_n: np.ndarray, ed_dropped: np.ndarray, src: np.ndarray, path: Path) -> dict:
    """Figure 3: the number of event tokens and of edges per view, by source, with the caps.

    The histograms hold the window totals, stored + dropped (``ev_n + ev_dropped``, ``ed_n + ed_dropped``): the stored counts stop at the
    cap, so a histogram of them could never show how far the windows run past it. The range reaches the longest window; the cap is the
    dashed line, and what lies right of it is what the cap drops.

    수식
    ----
    (REP2-6)  window_count = int64(stored_count) + dropped_count
             hist_{source,count_bin} = #{view : src=source, window_count∈count_bin}
             dropped_share = #{view : dropped_count>0} / views

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      stored_count, dropped_count, window_count | 저장·탈락·전체 개수 | 개/보기 | 비음수 정수 계약 | ev_n, ed_n, ev_dropped, ed_dropped, totals
      source, count_bin, hist | 보기 종류·정수 중심 빈·빈도 | —; 보기 | src=0,1,2, 빈 경계는 반정수 | src, bins, ax.hist
      views, dropped_share | 전체 보기 수와 탈락 보기 비율 | 보기; — | views>0 가정, 전체 빈 배열은 실패 | n, dropped[key]/n

    해석
    ----
    상한 전 꼬리를 보존하기 위해 저장된 개수와 버린 개수를 합쳐 그린다(S1–S2 R14-d). 히스토그램은 보기 종류별로 나누지만 제목의 넘침 비율은 전체 보기를 분모로 쓴다. 입력이 비면 최댓값 계산부터 실패하며 그림 함수가 별도 빈 입력 처리를 하지 않는다.

    근거
    ----
    S1–S2 판정 R14-d(상한 그림), R14-e(넘침 보고).
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    caps = {'events': C.MAX_EVENTS, 'edges': C.MAX_EDGES}
    # (REP2-6): 상한에 잘린 저장값만으로는 꼬리를 볼 수 없어 탈락 개수를 복구한다.
    totals = {'events': ev_n.astype(np.int64) + ev_dropped, 'edges': ed_n.astype(np.int64) + ed_dropped}
    dropped = {'events': int((ev_dropped > 0).sum()), 'edges': int((ed_dropped > 0).sum())}
    fig = _figure(10.4, 4.4)
    axes = fig.subplots(1, 2)
    for ax, key, label in ((axes[0], 'events', 'event tokens per view (window total: stored + dropped)'),
                           (axes[1], 'edges', 'co-participation edges per view (window total: stored + dropped)')):
        cap, values = caps[key], totals[key]
        bins = np.arange(-0.5, max(cap, int(values.max())) + 1.5, 1.0)
        for code, name in enumerate(SRC_NAMES):
            v = values[src == code]
            if len(v):
                ax.hist(v, bins=bins, histtype='step', linewidth=1.3, label=f'{name} ({len(v):,} views)')
        ax.axvline(cap, color='#d62728', ls='--', lw=1.2)
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.annotate(f'cap {cap}', xy=(cap, 0.97), xycoords=('data', 'axes fraction'), ha='right', va='top', color='#d62728', fontsize=9,
                    rotation=90, xytext=(-3, 0), textcoords='offset points')
        ax.set_xlabel(label)
        ax.set_ylabel('views (log scale)')
        n = len(values)
        ax.set_title(f'{dropped[key]:,} of {n:,} views ({dropped[key] / n:.1%}) drop {key}', fontsize=10)
        ax.legend(loc='best', fontsize=8)
    _save(fig, path)
    return {'caps': caps, 'views_with_dropped': dropped, 'max_stored': {'events': int(ev_n.max()), 'edges': int(ed_n.max())},
            'max_window': {'events': int(totals['events'].max()), 'edges': int(totals['edges'].max())}}


def fig_age(ages: np.ndarray, src: np.ndarray, path: Path) -> dict:
    """Figure 4: the age of the frame of a view (reference time minus the time of the frame it takes its values from), by source.

    수식
    ----
    (REP2-7)  age_s = (ref_ms − frame_ms)/1000
             histogram_top_s = max(60, ceil(max(age_s)) + 2); bins_s = arange(0, histogram_top_s + 2, 2)
             source_median = median(age_s[src=source]); source_p99 = Pct99(age_s[src=source])

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      age_s, ref_ms, frame_ms | 나이·기준 시각·프레임 시각 | s; ms; ms | 유한·비음수 나이 계약 | ages; 호출자의 ref,t0
      histogram_top_s, bins_s | 그림 상단과 빈 경계 | s | 빈 폭 2 | top, bins
      source, source_median, source_p99, Pct | 종류와 요약·numpy 선형 백분위 | —; s | 비어 있지 않은 종류만 요약 | src, a, np.median, np.percentile

    해석
    ----
    두 시계의 차이를 초 단위로 그려 S1–S2 R14-d의 보기별 나이를 보여 준다. 그림의 주석 수치는 소수 첫째 자리이고 반환값은 원래 중앙값과 최댓값이다. 빈 종류는 보기 수만 기록하고 중앙값·최댓값 키를 생략하며 전체 배열이 비면 먼저 실패한다.

    근거
    ----
    설계서 §4(두 시계); S1–S2 판정 R14-d(나이 그림).
    """
    top = max(60.0, math.ceil(float(ages.max())) + 2.0)
    bins = np.arange(0.0, top + 2.0, 2.0)
    fig = _figure(10.6, 3.9)
    axes = fig.subplots(1, len(SRC_NAMES), sharey=True)
    numbers = {'views': {}, 'median_s': {}, 'max_s': {}}
    for ax, (code, name) in zip(axes, enumerate(SRC_NAMES)):
        a = ages[src == code]
        numbers['views'][name] = int(len(a))
        if len(a):
            ax.hist(a, bins=bins, color='#4c78a8')
            numbers['median_s'][name], numbers['max_s'][name] = float(np.median(a)), float(a.max())
            ax.annotate(f'median {np.median(a):.1f} s\np99 {np.percentile(a, 99):.1f} s\nmax {a.max():.1f} s', xy=(0.97, 0.95),
                        xycoords='axes fraction', ha='right', va='top', fontsize=9)
        ax.set_title(f'{name}: {len(a):,} views', fontsize=10)
        ax.set_xlabel('age of the frame (s)')
    axes[0].set_ylabel('views')
    _save(fig, path)
    return numbers


# ---- the report --------------------------------------------------------------------------------------------------------------------------
def _quantiles(values: np.ndarray) -> dict:
    """p50, p90, p95, p99 (the next higher value: whole numbers for counts) and the maximum.

    (REP2-8)의 계산·입력 항을 맡는다.
    """
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99], method='higher')
    return {'p50': int(p50), 'p90': int(p90), 'p95': int(p95), 'p99': int(p99), 'max': int(values.max())}


def _overflow(stored: np.ndarray, dropped: np.ndarray, src: np.ndarray, cap: int) -> dict:
    """Events (or edges) that did not fit: the views that drop some, the total dropped, the window before the cap (stored + dropped), the
    views by source, and the smallest cap with which at most 5 % of the views would drop any (the largest window total that the allowed
    share of views may exceed: the (n // 20 + 1)-th largest).

    ``cap_for_5_percent`` is written only where the share of views that drop is over 5 % (the flag fires). Where it is not, the cap now
    already keeps the share at 5 % or less, a number below it would be a proposal for nothing, and the entry is None with a note.

    (REP2-8)의 계산·입력 항을 맡는다.
    """
    n = len(stored)
    over = dropped > 0
    window = stored.astype(np.int64) + dropped
    # (REP2-8): 허용 초과 수의 다음 순서통계량으로 동점까지 보수적으로 포함한다.
    allowed = n * OVERFLOW_LIMIT.numerator // OVERFLOW_LIMIT.denominator
    flagged = over_share(int(over.sum()), n)
    note = ('the smallest cap that keeps the share of views that drop at 5 % or less, from the window totals (the (n // 20 + 1)-th '
            'largest); a proposal, no cap is changed here' if flagged else
            'None: at most 5 % of the views drop at the cap now, so no other cap is proposed')
    return {'views_with_dropped': int(over.sum()), 'dropped_total': int(dropped.astype(np.int64).sum()), 'max_stored': int(stored.max()),
            'window_total': _quantiles(window), 'by_src': _by_src(over, src),
            'cap_for_5_percent': int(np.sort(window)[::-1][allowed]) if flagged else None, 'cap_for_5_percent_note': note, 'cap_now': cap}


def _flag(flags: list, code: str, message: str) -> None:
    flags.append({'code': code, 'message': message})


def _check_record(n: int, by_src: dict, with_views: int, parts: dict, fields: dict) -> None:
    """ReportInputError unless the ``s2_result`` of RUN.json says what the store holds: views, views by source, bytes per view (the total
    and its two parts), and no more matches with views than matches listed (a listed match without a view, a very short one, leaves no
    row in the table, so the table can hold fewer matches than were listed); and unless the x arrays take the bytes that
    ``tokens.bytes_per_view()`` says."""
    total = parts['x'] + parts['views_table']
    for what, ours, theirs in (('views', n, fields['views']), ('views_by_src', by_src, fields['views_by_src']),
                               ('bytes_per_view', total, fields['bytes_per_view'])):
        if ours != theirs:
            raise ReportInputError(f'RUN.json s2_result.{what} is {theirs}, the store has {ours}: the record and the store disagree')
    if with_views > fields['matches']:
        raise ReportInputError(f'RUN.json s2_result.matches lists {fields["matches"]} matches, the views table holds views of '
                               f'{with_views}: the record and the store disagree')
    for what, ours in (('x_bytes_per_view', parts['x']), ('views_bytes_per_view', parts['views_table'])):
        if fields[what] is not None and fields[what] != ours:
            raise ReportInputError(f'RUN.json s2_result.{what} is {fields[what]}, the store has {ours}: the record and the store disagree')
    if parts['x'] != tokens.bytes_per_view():
        raise ReportInputError(f'the x arrays of the layout take {parts["x"]} bytes per view, tokens.bytes_per_view() says '
                               f'{tokens.bytes_per_view()}')


def _level_formula(p_scan: dict, t0: np.ndarray, src: np.ndarray) -> dict:
    """The ``level_formula_check`` section (R14-b) from the differences ``lvl + item - cs`` that ``_scan_players`` collected.

    (REP2-11)의 계산·입력 항을 맡는다.
    """
    diffs, frame = p_scan['diffs'], src == 0
    offset = np.abs(t0[frame] - np.rint(t0[frame] / 60000.0) * 60000.0)             # how far each frame is from a whole minute
    if len(diffs):
        p10, median, p90 = np.percentile(diffs, [10, 50, 90])
        spread = {'median': float(median), 'p10': float(p10), 'p90': float(p90), 'share_within_1': float(np.mean(np.abs(diffs) <= 1.0))}
    else:
        spread = dict.fromkeys(('median', 'p10', 'p90', 'share_within_1'))
    return {
        'formula': 'lvl_attackDamage + item_attackDamage - cs_attackDamage',
        'fields_used': {'lvl_attackDamage': 'players_num trait column: the champion statistic at the level of the frame '
                                            '(traits.stat_at_level)',
                        'item_attackDamage': 'players_num item column: the attack damage of the items owned (LEGACY item replay)',
                        'cs_attackDamage': "players_num frame column: the frame's championStats.attackDamage"},
        'views': p_scan['frame_views'], 'player_frames': int(len(diffs)), 'nan_excluded': p_scan['unknown'].get('nan_excluded', 0),
        'unknown_item_excluded': p_scan['unknown'].get('unknown_item_excluded', 0),
        **spread,
        't0_offset_from_whole_minute_ms': {'median': float(np.median(offset)) if len(offset) else None,
                                           'max': int(offset.max()) if len(offset) else None},
        'note': 'a sanity check, not a gate: frame-clock views (reference time = frame time), per player; NaN rows (unknown champions) '
                'and rows with an unknown item (its attack damage is missing from the item term) left out and counted '
                '(player_frames + nan_excluded + unknown_item_excluded = 10 x views); runes and skills explain the remainder; '
                'percentiles by linear interpolation'}


def _extrapolation(s1: Store, run: Path, n: int, n_matches: int, bytes_per_view: int, raw_fit_matches: int) -> dict:
    """The ``extrapolated_gib`` section (R14-c): bytes per view x the exact frame-clock and event rows of the S1 store per macro of the full
    build, and x the estimated views of the raw FIT cohort (``raw_fit_matches`` x the views per match of this run), each with its
    arithmetic; the same estimate with the mean views per match of all FIT matches of the S1 store is shown as a cross-check.

    (REP2-14)의 계산·입력 항을 맡는다.
    """
    role, row_off, event_off = (np.asarray(s1.array(f'matches/{name}')) for name in ('role', 'row_off', 'event_off'))
    frame_rows, event_rows = np.diff(row_off), np.diff(event_off)
    by_macro = {}
    for macro in FULL_BUILD_MACROS:
        sel = role == C.ROLES.index(macro)
        f, e = int(frame_rows[sel].sum()), int(event_rows[sel].sum())
        by_macro[macro] = {'matches': int(sel.sum()), 'frame_rows': f, 'event_rows': e, 'views': f + e,
                           'gib': (f + e) * bytes_per_view / GIB}
    full_views = sum(v['views'] for v in by_macro.values())
    full_gib = full_views * bytes_per_view / GIB
    views_per_match = n / n_matches
    raw_views = raw_fit_matches * views_per_match
    raw_gib = raw_views * bytes_per_view / GIB
    full = {'macros': list(FULL_BUILD_MACROS), 'by_macro': by_macro, 'views': full_views, 'gib': full_gib,
            'arithmetic': f'{full_views:,} views x {bytes_per_view:,} bytes / 2^30 = {full_gib:.3f} GiB '
                          '(the exact frame-clock and event rows of the S1 store)'}
    other_roles = {C.ROLES[r]: int(frame_rows[role == r].sum() + event_rows[role == r].sum()) for r in np.unique(role).tolist()
                   if C.ROLES[r] not in FULL_BUILD_MACROS}
    if other_roles:
        full['views_of_other_roles_not_in_the_gate_command'] = other_roles
    raw = {'is_estimate': True, 'matches': raw_fit_matches, 'matches_source': RAW_FIT_SOURCE, 'views_per_match': views_per_match,
           'views_per_match_source': f'this smoke run: {n:,} views / {n_matches:,} matches listed in RUN.json', 'views': raw_views,
           'gib': raw_gib,
           'arithmetic': f'estimate: {raw_fit_matches:,} matches x {views_per_match:.2f} views/match x {bytes_per_view:,} bytes / 2^30 '
                         f'= {raw_gib:.1f} GiB'}
    fit = by_macro['FIT']
    if fit['matches']:
        mean = fit['views'] / fit['matches']
        raw['cross_check_mean_of_all_fit_matches_of_s1'] = {
            'views_per_match': mean, 'views': raw_fit_matches * mean, 'gib': raw_fit_matches * mean * bytes_per_view / GIB,
            'arithmetic': f'estimate: {raw_fit_matches:,} matches x {mean:.2f} views/match ({fit["views"]:,} views / '
                          f'{fit["matches"]:,} FIT matches of the S1 store) x {bytes_per_view:,} bytes / 2^30 = '
                          f'{raw_fit_matches * mean * bytes_per_view / GIB:.1f} GiB'}
    return {'bytes_per_view': bytes_per_view, 'cache_full_build': full_gib, 'raw_fit_estimate': raw_gib,
            'detail': {'gib_is': '2^30 bytes', 'cache_full_build': full, 'raw_fit_estimate': raw,
                       'drive_free_gib_at_report_time': shutil.disk_usage(run).free / GIB}}


def _suspect_positions(terr, p_scan: dict, rows: int) -> dict:
    """The ``suspect_positions`` section (S2-9): the census of TERRAIN.json by source, and the positions stored in the views of this run.

    (REP2-10)의 계산·입력 항을 맡는다.
    """
    pos = p_scan['positions']
    keys = ('nan_rows', 'zero_zero_rows', 'off_map_rows', 'views_with_zero_zero', 'views_with_off_map', 'views_with_any')
    return {'terrain_fit': {k: _need(terr.manifest, 'positions', k, source='TERRAIN.json') for k in CENSUS_SOURCES},
            'smoke_views': {'player_rows': rows, **{k: pos[k] for k in keys}, 'views_with_any_by_src': p_scan['positions_by_src'],
                            'map_side': float(C.W)},
            'note': 'S2-9: such positions are snapped like any other and never dropped; terrain_fit counts the positions of the FIT '
                    'matches that made the terrain, smoke_views the player positions stored in the views of this run'}


def _frame_age(ages: np.ndarray, src: np.ndarray) -> dict:
    """The age of the frame of a view in s, by source: views, minimum, median, 99th percentile, maximum.

    (REP2-15)의 계산·입력 항을 맡는다.
    """
    out = {}
    for code, name in enumerate(SRC_NAMES):
        a = ages[src == code]
        out[name] = ({'views': int(len(a)), 'min': float(a.min()), 'median': float(np.median(a)), 'p99': float(np.percentile(a, 99)),
                      'max': float(a.max())} if len(a) else {'views': 0})
    return out


def report(run, *, s1=None, terrain_dir=None, report_dir=None, json_path=None, records=None, verify: bool = True,
           raw_fit_matches: int = RAW_FIT_MATCHES) -> Path:
    r"""Write ``S2_REPORT.json`` (``json_path``, default ``<run>\S2_REPORT.json``) and the four figures (``report_dir``, default
    ``<run>\report``) of a finished S2 run folder and return the path of the JSON. The module text lists what the report holds.

    ``s1`` the S1 run folder or its store (default: the folder named in RUN.json next to ``run``), ``terrain_dir`` the terrain folder
    (default ``<run>\terrain``), ``records`` a function ``(key, patch) -> MatchRecord`` for the two measurements on the raw matches (None:
    no raw match is read; given: the measurement raises ``run_s1.StrataRunning`` while a model server runs, before anything is
    written), ``verify`` hashes every file of the S2 store against STORE.json before it is read, ``raw_fit_matches`` the
    size of the raw FIT cohort for the estimate (an int of at least 1). Refusals are listed in the module text; nothing is written for
    them. The outputs may lie neither in the S2 store nor in the S1 run.


    수식
    ----
    (REP2-8)  window_count = int64(stored_count) + dropped_count; overflow_view = [dropped_count>0]
             allowed_rows = floor(views/20); proposed_cap = sort_desc(window_count)[allowed_rows] (flag), 아니면 None
             window_Pct_p = int(Pct_p(window_count, method='higher')), p∈{50,90,95,99}
             union_share = Σ_view [event_dropped>0 ∨ edge_dropped>0] / views

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      views, stored_count, dropped_count, window_count | 보기 수와 창 개수 | 보기; 개 | views>0, 유효 정수 계약 | n, stored, dropped, window
      overflow_view, flag, allowed_rows | 탈락 표시·엄격 초과·허용 수 | —; 보기 | flag는 (REP2-1), 정렬 위치는 0 기반 | over, flagged, allowed
      proposed_cap, window_Pct_p, Pct | 상한 제안·백분위·numpy 연산 | 개 | 허용 수+1번째로 큰 값, 동점 허용 | cap_for_5_percent, _quantiles
      union_share, event_dropped, edge_dropped | 합집합 비율과 두 탈락 개수 | —; 개 | 전체 보기 분모 | any_dropped, ints

    경계: views>0 및 비음수 정수 개수 계약; 빈 입력의 정렬·최댓값은 별도 처리 없음. 동점은 cap보다 엄격히 큰 창만 초과로 셈. 합집합은 각 종류와 별도 검사.

    (REP2-9)  nan_count_col = Σ_row [isnan(value_{row,col})]; inf_count_col = Σ_row [isinf(value_{row,col})]
             col_min/col_max = NaN을 무시한 최소/최대
             constant_col = None (모든 행 NaN), col_min (NaN 없음 ∧ col_min=col_max), 아니면 키 생략
             nan_share_cell = nan_count_cell / denominator_cell (분모>0), 아니면 None

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, col, cell, value | 관측 행·열·뒷축 칸과 값 | —; 원 값 단위 | 0 기반; 플레이어 행은 보기×10 | flat, block, arr
      nan_count, inf_count, col_min, col_max | NaN·무한대 수와 극값 | 개; 원 값 단위 | 무한대는 극값에 남음 | acc, c
      constant_col | 상수 열 기록 | 원 값 단위 | 혼합 NaN 열은 미기록 | _constant_columns
      denominator_cell | 플레이어 열·요약 열·랜드마크·전체 텐서 분모 | 관측 칸 | 각각 10·views, views, views, views·뒷축 칸 수 | rows, n, c['views']*c['nan'].size
      nan_share_cell | 허용 칸의 NaN 비율 | — | 분모 영이면 None | _nan_section, _share

    경계: 무한대는 허용 NaN 칸에서도 위반; 위반 목록은 개수 양수인 항목만, ok=목록이 빔. 혼합 NaN 상수 열은 생략; 모든 NaN이면 None. 직렬화 때 비유한 값을 None으로 바꾸는 처리는 유한성 검증이 아님.

    (REP2-10)  missing_pos = isnan(pos_x) ∨ isnan(pos_y)
             zero_pos = ¬missing_pos ∧ pos_x=0 ∧ pos_y=0
             offmap_pos = ¬missing_pos ∧ [pos_x 또는 pos_y가 [0,map_side] 밖]
             any_suspect_view = any_player(zero_pos) ∨ any_player(offmap_pos)
             unknown_share = unknown_player_rows/(10·views); unknown_item_total = Σ_block round(Σ_{item_count>0} float64(item_count))

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      pos_x, pos_y, map_side | 플레이어 위치와 지도 폭 | 지도 단위 | 경계 포함, 무한대는 지도 밖 | x,y,side
      missing_pos, zero_pos, offmap_pos, any_suspect_view | 위치 표시와 보기 표시 | — | bool; NaN만 있는 보기는 any에서 제외 | missing,zero,off,vz,vo
      unknown_player_rows, unknown_share | champ_unknown=1 또는 item_count>0 행 수와 비율 | 관측 행; — | 같은 플레이어의 반복 관측 포함 | unknown, 반환 필드
      item_count, unknown_item_total, views | 미지 아이템 수와 합·보기 수 | 개; 보기 | 양수 수만 블록별 반올림 | item, unknown_items_total, n

    경계: NaN-only 보기는 any_suspect_view에서 제외; 무한 좌표는 offmap. 분모는 고유 인원이 아닌 반복 관측 행; 아이템 총합은 블록별 float64 합 후 반올림.

    (REP2-11)  damage_diff = float64(level_damage) + item_damage − frame_damage (src=0)
             valid_damage = isfinite(damage_diff) ∧ ¬(item_count>0)
             within_share = mean([abs(damage_diff)≤1] on valid_damage)
             minute_offset_ms = abs(frame_ms − rint(frame_ms/60000)·60000)
             damage_summary = Pct_{10,50,90}(valid damage_diff), 선형 보간

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      level_damage, item_damage, frame_damage, damage_diff | 레벨·아이템·프레임 공격력과 잔차 | 공격력 | 프레임 시계 관측만 | lvl_attackDamage,item_attackDamage,cs_attackDamage,d
      valid_damage, item_count | 점검 포함 표시와 미지 아이템 수 | —; 개 | 비유한 잔차 먼저 제외, 항목 제외와 겹치지 않음 | finite,short_item
      within_share, damage_summary, Pct | 허용 폭 내 비율과 잔차 백분위 | —; 공격력 | 잔차 없음이면 모두 None | spread,np.percentile
      frame_ms, minute_offset_ms, rint | 프레임 시각·정분 편차·최근접 정수 | ms | 반정수 동점은 짝수 방향 | t0,offset,np.rint

    경계: nan_excluded는 모든 비유한 잔차를 셈; 유한한 미지 아이템 제외와 중복 없음. 유효 잔차 없으면 잔차 요약 None; 프레임 없으면 시각 편차 None.

    (REP2-12)  age_s = (ref_ms − frame_ms)/1000; time_s = ref_ms/1000
             age_deviation = max_blocks(previous, max(abs(float64(stored_age_s) − age_s)))
             time_deviation = max_blocks(previous, max(abs(float64(stored_time_s) − time_s)))
             age_ok = [age_deviation≤0.0005]; time_ok = [time_deviation≤0.0005]; frame_ok = all(ref_ms[src=0]=frame_ms[src=0])

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ref_ms, frame_ms, age_s, time_s | 기준·프레임 시각과 초 단위 값 | ms; s | 유한 저장 시계 계약 | ref,t0,ages
      stored_age_s, stored_time_s | summary의 저장 시계 | s | float32 저장, 비교는 float64 | block[:,age_col],block[:,time_col]
      previous, age_deviation, time_deviation | 이전 최댓값과 최대 절대 차이 | s | 초기값 0 | age_dev,time_dev
      age_ok, time_ok, frame_ok | 시계 대조 표시 | — | bool | checks

    경계: 허용차는 float32 저장을 위한 수치 규약. NaN 블록 최댓값은 Python max에서 이전 최댓값을 남길 수 있어 이 검사만으로 유한성 보증 없음.

    (REP2-13)  different_{view,player_row} = [clock_level ≠ stored_level]
             same_ms = [(player_row+1,ref_ms)∈LEVEL_UP]
             lag_one = different ∧ same_ms ∧ [stored_level=clock_level−1]
             different_share = Σ different / measured_cells; unexplained = Σ different − Σ lag_one

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      view, player_row, ref_ms | 보기·플레이어 배열 행과 기준 시각 | —; ms | 0 기반 행 0..9, 사건 참가자 ID는 행+1 | p,ref
      clock_level, stored_level, LEVEL_UP | 레벨 시계·저장값·레벨업 사건 집합 | 레벨; — | NaN 저장값도 불일치 | clock,row,ups
      different, same_ms, lag_one | 불일치·동시 사건·한 레벨 지연 | — | bool | lag 카운터
      measured_cells, different_share, unexplained | 성공 기록의 관측 칸·비율·잔여 수 | 칸; —; 칸 | 분모 영이면 None | cells,_share,differ_unexplained

    경계: 두 원시 측정이 모두 성공한 경기만 합산; 지정한 읽기·형식 오류는 양쪽에서 제외. NaN 저장 레벨은 불일치; 분모 영은 None; unexplained는 산술 잔여. 원시 기록 측정 요청이 없으면 미측정이며 요청이 있으면 측정의 첫 단계에서 E6 검사.

    (REP2-14)  bytes_per_view = Σ_x prod(shape_x)·itemsize_x + Σ_view_column itemsize_column
             cache_views_role = Σ_{match in role}(frame_rows_match + event_rows_match)
             cache_gib = Σ_{role∈FULL_BUILD_MACROS} cache_views_role·bytes_per_view / 2^30
             raw_views_estimate = raw_matches·views/listed_matches; raw_gib = raw_views_estimate·bytes_per_view/2^30
             fit_crosscheck_gib = raw_matches·(cache_views_FIT/cache_matches_FIT)·bytes_per_view/2^30

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      shape_x, itemsize, bytes_per_view | 보기 배열 모양·dtype 바이트·합 | —; byte | 뷰 표 27 byte도 포함 | _layout(1),parts
      role, frame_rows_match, event_rows_match, cache_views_role | 합산 역할과 그 경기별 프레임·사건·보기 수 | —; 행 | role∈FULL_BUILD_MACROS; 나머지 역할은 별도 보고 | role,np.diff(row_off),np.diff(event_off),by_macro
      FULL_BUILD_MACROS | 전체 배열 크기에 합산하는 역할 집합 | — | 정확히 FIT,STOP,DEV,CAL,AUDIT | FULL_BUILD_MACROS
      cache_matches_FIT, cache_views_FIT | S1 FIT 캐시의 경기 수와 보기 수 | 경기; 보기 | 경기 수가 양수인 분기에서만 교차 외삽 | fit['matches'],fit['views']
      views, listed_matches, raw_matches, raw_views_estimate | 현재 선택 실행의 보기·명단 경기·원시 추정 경기·추정 보기 | 보기; 경기 | listed_matches>0 계약, raw_matches 양의 정수 검사 | n,n_matches,raw_fit_matches,raw_views
      cache_gib, raw_gib, fit_crosscheck_gib | 배열 크기와 두 외삽 | GiB | FIT 경기 없으면 crosscheck 키 생략 | full_gib,raw_gib,raw cross_check

    경계: listed_matches>0 계약; raw_matches 양의 정수 검사. FIT 캐시 경기 없으면 교차 확인 키 생략; 기존 역할 라벨만 읽고 재분할하지 않음. 배열 payload 크기이며 상주·피크 메모리나 디스크 사용량·실행 비용 측정이 아님.

    (REP2-15)  memory_ok = [recorded_peak≤48] (recorded_peak 있음), 아니면 None
             views_per_match = views/listed_matches
             source_views = Σ_view [src=source]; source_age = min/median/Pct99/max(age_s[src=source])

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      recorded_peak, memory_ok | 저장된 트리 메모리 표본 최댓값과 표시 | GiB; — | None 가능 | fields['peak_private_gib'],memory_gate
      views, listed_matches, views_per_match | 보기·명단 경기 수와 비 | 보기; 경기; 보기/경기 | 양의 분모 계약, 보기 없는 경기도 명단에 포함 | n,fields['matches'],doc
      src, source, source_views, source_age, age_s, Pct | 원천 종류와 종류별 수·나이 요약 | —; 보기; s | 0=frame_clock,1=M3,2=M4; 빈 종류는 views만 | src,by_src,_frame_age

    경계: 명단의 보기 없는 경기도 경기 분모에 포함; 빈 원천 종류는 views 키만 기록. 저장 피크 없음이면 memory_ok=None; flags만으로 보고를 중지하거나 다음 실행을 승인하지 않음.

    해석
    ----
    S1–S2 R14-a·R14-e의 보고는 현재 선택된 완료 실행의 보기·플레이어 관측 칸·명단 경기를 각 통계의 분모로 구별하고 상한을 바꾸지 않는다.
    넘침은 꼬리를 잃지 않도록 상한 전 창 개수로 요약하며 허용 초과 수 다음의 순서통계량을 제안한다.
    결측·위치·저장 시계·레벨 지연은 각 원래 결정의 관측 계약과 비교하는 진단이고 검사 범위 밖의 입력 보증은 하지 않는다.
    공격력 잔차는 S1–S2 R14-b의 sanity check이며 잔차만으로 룬·스킬의 원인을 식별하지 않는다.
    S1–S2 R14-c의 캐시 행 수와 현재 선택 실행을 이용한 원시 경기 외삽을 구별하며 저장된 메모리와 주의 표시는 사람의 관문 승인과 별개이다.

    근거
    ----
    S1–S2 판정 R14-a(분모·결측·상수), R14-b(공격력), R14-c(외삽), R14-e(넘침), R13-o(바이트), R13-f 개정(메모리), S2-8(레벨 지연), S2-9(위치).
    설계서 §4(두 시계); tokens.NAN_ALLOWED(허용 계약); _scan_summary(수치 대조); 지침 E6(원시 측정); S3 판정 R69(공개 소유).
    """
    if isinstance(raw_fit_matches, bool) or not isinstance(raw_fit_matches, int) or raw_fit_matches < 1:
        raise ValueError(f'raw_fit_matches must be an int of at least 1, got {raw_fit_matches!r}')
    run = Path(run)
    run_json = _read_run_json(run)
    fields = _run_fields(run_json)
    s2 = _open_s2(run, verify)
    s1_store, s1_folder = _open_s1(s1, run, fields)
    terr, terrain_folder = _open_terrain(terrain_dir, run, fields)
    json_path = _check_output(run / 'S2_REPORT.json' if json_path is None else json_path, s2.root, 'the report file', s1_folder)
    report_dir = _check_output(run / 'report' if report_dir is None else report_dir, s2.root, 'the figure folder', s1_folder)

    # -- the table, the sizes, and RUN.json against the store
    table = {c: np.asarray(s2.array(f'views/{c}')) for c, _ in VIEWS_COLUMNS}
    src, ref, t0 = table['src'], table['ref'], table['t0']
    n = len(src)
    if int(src.min()) < 0 or int(src.max()) >= len(SRC_NAMES):
        raise ReportInputError('views/src holds a value other than 0, 1 or 2')
    smoke = np.unique(table['match'])                                   # the S1 matches of this run
    s1_role, s1_patch = np.asarray(s1_store.array('matches/role')), np.asarray(s1_store.array('matches/patch'))
    if int(smoke.min()) < 0 or int(smoke.max()) >= len(s1_role):
        raise ReportInputError(f'views/match holds S1 match rows outside 0 .. {len(s1_role) - 1}: the S1 store is not the one of this run')
    counts = np.bincount(src, minlength=len(SRC_NAMES))
    by_src = {name: int(counts[i]) for i, name in enumerate(SRC_NAMES)}
    parts = {'x': sum(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize for name, (shape, dtype) in _layout(1).items()
                      if name.startswith('x/')),
             'views_table': sum(np.dtype(d).itemsize for _, d in VIEWS_COLUMNS)}
    bytes_per_view = parts['x'] + parts['views_table']
    _check_record(n, by_src, len(smoke), parts, fields)

    # -- one pass over every array
    players = s2.array('x/players_num')
    p_scan = _scan_players(players, table)
    s_scan = _scan_summary(s2.array('x/summary'), table)
    cells = {key: _scan_cells(s2.array(f'x/{key}')) for key, (_, dtype) in tokens.VIEW_SPEC.items()
             if dtype.kind == 'f' and key not in ('players_num', 'summary')}
    ints = {key: np.asarray(s2.array(f'x/{key}')) for key in ('ev_n', 'ev_dropped', 'ed_n', 'ed_dropped')}
    # (REP2-12): 정수 ms 차를 s로 바꾼 뒤 float32 저장 시계와 대조한다.
    ages = (ref - t0) / 1000.0
    checks = {'views_agree_with_run_json': True,                        # _check_record raised otherwise
              'frame_clock_rows_on_their_frame': bool((ref[src == 0] == t0[src == 0]).all()),
              'frame_age_matches_summary': bool(s_scan['age_deviation_s'] <= TIME_TOLERANCE_S),
              'game_time_matches_ref': bool(s_scan['time_deviation_s'] <= TIME_TOLERANCE_S)}

    # -- the findings
    flags, peak = [], fields['peak_private_gib']
    memory_gate = {'limit_gib': LIMIT_PEAK_GIB, 'peak_private_gib': peak, 'within_limit': None if peak is None else peak <= LIMIT_PEAK_GIB}
    if peak is None:
        _flag(flags, 'PEAK_MEMORY_NOT_RECORDED', 'RUN.json records no peak private memory (no psutil?): the 48 GiB gate cannot be checked')
    elif peak > LIMIT_PEAK_GIB:
        _flag(flags, 'PEAK_MEMORY_OVER_LIMIT', f'peak private memory {peak} GiB (the whole process tree) is above the '
                                               f'{LIMIT_PEAK_GIB:g} GiB limit: stop and report')
    if run_json.get('strata_running') is True:
        _flag(flags, 'STRATA_WAS_RUNNING_AT_BUILD', 'RUN.json says a model server ran when the build started (E6 should have stopped it)')
    if run_json['terrain'].get('built_from_this_s1') is False:
        _flag(flags, 'TERRAIN_NOT_FROM_THIS_S1', 'RUN.json says the terrain was not built from the S1 run this S2 run consumed '
                                                 '(terrain.built_from_this_s1 is false)')
    if fields['warnings']:
        _flag(flags, 'BUILD_WARNINGS', f'the build recorded {len(fields["warnings"])} warning(s): {fields["warnings"][:3]}')
    if fields['recovered'] is True:
        _flag(flags, 'RUN_RECORD_RECOVERED', 'the run record (s2_result) was repaired after a crash from what the store remembers: its '
                                             'times and memory peaks are those of the calls that the store knows')
    # (REP2-8)의 합집합: 종류별 초과와 독립적으로 전체 보기 분모에 대조한다.
    any_dropped = int(((ints['ev_dropped'] > 0) | (ints['ed_dropped'] > 0)).sum())
    overflow = {'caps': {'events': C.MAX_EVENTS, 'edges': C.MAX_EDGES}, 'limit_share': float(OVERFLOW_LIMIT),
                'events': _overflow(ints['ev_n'], ints['ev_dropped'], src, C.MAX_EVENTS),
                'edges': _overflow(ints['ed_n'], ints['ed_dropped'], src, C.MAX_EDGES),
                'events_or_edges': {'views_with_dropped': any_dropped, 'share': any_dropped / n}}
    for key, code in (('events', 'EVENT_OVERFLOW_OVER_5_PERCENT'), ('edges', 'EDGE_OVERFLOW_OVER_5_PERCENT')):
        o = overflow[key]
        if over_share(o['views_with_dropped'], n):                                     # cap_for_5_percent is written exactly then
            _flag(flags, code, f'{o["views_with_dropped"] / n:.1%} of the views drop {key} (more than 5 %): a cap of '
                               f'{o["cap_for_5_percent"]} would keep it at 5 % or less (now {o["cap_now"]}); the cap is not changed here')
    if over_share(any_dropped, n):                  # R14-e: more than 5 % of the views drop events OR edges (each kind alone may be under 5 %)
        _flag(flags, 'EVENTS_OR_EDGES_OVERFLOW_OVER_5_PERCENT',
              f'{any_dropped / n:.1%} of the views ({any_dropped} of {n}) drop events or edges (more than 5 %): '
              f'{overflow["events"]["views_with_dropped"]} drop events, {overflow["edges"]["views_with_dropped"]} drop edges; a cap is '
              'proposed above only for a kind that is over 5 % on its own; no cap is changed here')
    nan_share = _nan_section(n, p_scan, s_scan, cells)
    if not nan_share['ok']:
        first = '; '.join(f'{e["array"]}/{e["cell"]} {e["kind"]} x{e["count"]}' for e in nan_share['outside_allowed'][:5])
        _flag(flags, 'NAN_OUTSIDE_ALLOWED', f'{len(nan_share["outside_allowed"])} cell group(s) hold NaN or infinity where '
                                            f'tokens.NAN_ALLOWED allows none: {first}')
    if not all(checks.values()):
        _flag(flags, 'VIEW_TABLE_INCONSISTENT', f'the views table and the arrays disagree: {[k for k, v in checks.items() if not v]}')

    # -- the measurements on the raw matches: before the first file is written, because the E6 check of the measurement can refuse the whole report
    raw = _measure(s1_store, table, players, records, smoke) if records is not None else _unmeasured(len(smoke))

    # -- the figures
    path_info = base_path(terr)
    report_dir.mkdir(parents=True, exist_ok=True)
    over_events, over_edges = overflow['events']['views_with_dropped'], overflow['edges']['views_with_dropped']
    numbers = {
        FIG_NAMES[0]: fig_occupancy(terr, report_dir / f'{FIG_NAMES[0]}.png'),
        FIG_NAMES[1]: fig_path(terr, path_info, report_dir / f'{FIG_NAMES[1]}.png'),
        FIG_NAMES[2]: fig_counts(ints['ev_n'], ints['ev_dropped'], ints['ed_n'], ints['ed_dropped'], src, report_dir / f'{FIG_NAMES[2]}.png'),
        FIG_NAMES[3]: fig_age(ages, src, report_dir / f'{FIG_NAMES[3]}.png')}
    figures = {name: {'file': _relative(report_dir / f'{name}.png', Path(json_path).parent), 'numbers': numbers[name]}
               for name in FIG_NAMES}

    # -- the findings of the raw measurements
    errors = raw['raw_measurement']['errors']
    if errors:
        _flag(flags, 'RAW_MEASUREMENT_ERRORS', f'{sum(errors.values())} match(es) could not be measured: {sorted(errors)[:3]}')
    if raw['level_lag']['measured'] and raw['level_lag']['differ_unexplained']:
        _flag(flags, 'LEVEL_LAG_UNEXPLAINED', f'{raw["level_lag"]["differ_unexplained"]} frame-clock cell(s) differ from the level clock '
                                              'without a LEVEL_UP of the player on the ms of the frame that explains them (S2-8): '
                                              'stop and report')

    # -- the document
    # (REP2-10): 분모는 고유 인원이 아니라 플레이어 관측 칸 전체이다.
    rows, u = 10 * n, p_scan['unknown']
    unknown_keys = ('rows_with_unknown_champion', 'rows_with_unknown_item', 'unknown_items_total', 'views_with_unknown_champion',
                    'views_with_unknown_item')
    unknown = {'player_rows': rows, **{k: u.get(k, 0) for k in unknown_keys}}
    patches = sorted({C.PATCHES[int(p)] for p in s1_patch[smoke].tolist()}, key=C.PATCHES.index)
    manifest = terr.manifest
    doc = {
        'report': 'S2_REPORT', 'flags': flags,
        'run': {'name': run.name, 'stage': run_json.get('stage'), 'created': run_json.get('created'),
                'code_commit': run_json.get('code_commit'), 'code_dirty': run_json.get('code_dirty'), 'gpu_at_start': run_json.get('gpu'),
                'strata_running': run_json.get('strata_running'), 'args': fields['args'], 'calls': fields['calls'],
                'recovered': fields['recovered'], 'warnings': fields['warnings'], 'per_call': fields['per_call'],
                'code_state_at_end': fields['code_state_at_end']},
        's1_run': s1_folder.name, 'report_code': {**runlog.code_state(), 'store_verified': bool(verify)},
        'views': n, 'matches': fields['matches'], 'matches_with_views': len(smoke), 'views_per_match': n / fields['matches'],
        'views_by_src': by_src,
        'bytes_per_view': bytes_per_view, 'bytes_per_view_parts': parts,
        'seconds_per_match': fields['seconds_per_match'], 'wall_s': fields['wall_s'], 'workers': fields['workers'],
        'peak_private_gib': peak, 'peak_private_main_gib': fields['peak_private_main_gib'],
        'peak_private_worker_gib': fields['peak_private_worker_gib'], 'memory_gate': memory_gate,
        'event_overflow_share': over_events / n, 'edge_overflow_share': over_edges / n, 'overflow': overflow,
        'unknown_item_share': unknown['rows_with_unknown_item'] / rows,
        'unknown_champion_share': unknown['rows_with_unknown_champion'] / rows,
        'unknown': unknown, 'level_formula_check': _level_formula(p_scan, t0, src),
        'extrapolated_gib': _extrapolation(s1_store, run, n, fields['matches'], bytes_per_view, raw_fit_matches),
        'suspect_positions': _suspect_positions(terr, p_scan, rows), 'nan_share': nan_share,
        'constant_columns': {'views': n, 'summary': _constant_columns(s_scan['columns'], tokens.SUMMARY),
                             'players_num': _constant_columns(p_scan['columns'], tokens.PLAYER_NUM), 'patches_in_smoke': patches,
                             'note': f'constant over the views of this smoke run only (patches {", ".join(patches)}); a column that '
                                     'varies by patch (atakhan_in_patch, the rule_uncertain_* flags) can vary in the full build'},
        'frame_age_s': _frame_age(ages, src),
        'level_lag': raw['level_lag'], 'building_kill_matching': raw['building_kill_matching'], 'raw_measurement': raw['raw_measurement'],
        'smoke': {'matches_by_patch': {p: int((s1_patch[smoke] == C.PATCHES.index(p)).sum()) for p in patches},
                  'matches_by_macro': {C.ROLES[r]: int((s1_role[smoke] == r).sum()) for r in np.unique(s1_role[smoke]).tolist()}},
        'terrain': {'folder': terrain_folder.name, 'grid': manifest['grid'], 'walkable_cells': manifest.get('walkable_cells'),
                    'components': manifest.get('components'), 'fit_matches': manifest.get('fit_matches'), 'hashes_match_run_json': True,
                    'built_from_this_s1': run_json['terrain'].get('built_from_this_s1'),
                    'geodesic_path': {k: v for k, v in path_info.items() if k != 'cells'}},
        'store': {'complete': True, 'verified': bool(verify), 'files': len(s2.stored_files()),
                  'bytes': sum(p.stat().st_size for p in s2.stored_files())},
        'checks': checks, 'figures': figures,
    }
    text = json.dumps(_clean(doc), indent=1, allow_nan=False) + '\n'
    assert_no_keys(s1_store, text)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    store_core.write_atomic(json_path, text)
    for store in (s1_store, s2):
        store.maps.clear()
    del terr, players
    gc.collect()                                                     # Windows does not delete a folder that holds a mapped file
    return Path(json_path)


# ---- the command line --------------------------------------------------------------------------------------------------------------------
def _summary(doc: dict, path: Path) -> list:
    """The headline numbers of a finished report as lines of text (the gate's list, R14-h) and its flags; counts and names only."""
    by = doc['views_by_src']
    gate = doc['memory_gate']
    ex = doc['extrapolated_gib']
    either = doc['overflow']['events_or_edges']
    nan = doc['nan_share']
    shares = sorted(((s, f'{where}/{name}') for where in ('players_num', 'summary', 'landmark_seconds')
                     for name, s in nan[where].items() if s),
                    reverse=True)
    constants = doc['constant_columns']
    lines = [f'S2 report written: {path}',
             f'views {doc["views"]:,} (frame-clock {by["frame_clock"]:,}, M3 {by["M3"]:,}, M4 {by["M4"]:,}) of {doc["matches"]:,} matches; '
             f'{doc["bytes_per_view"]:,} bytes per view (x {doc["bytes_per_view_parts"]["x"]:,} + table '
             f'{doc["bytes_per_view_parts"]["views_table"]})',
             f'seconds per match {doc["seconds_per_match"]}; peak private {gate["peak_private_gib"]} GiB '
             f'(main {doc["peak_private_main_gib"]}, largest worker {doc["peak_private_worker_gib"]}); limit {gate["limit_gib"]:g} GiB',
             f'event overflow {doc["event_overflow_share"]:.2%}, edge overflow {doc["edge_overflow_share"]:.2%}, events or edges '
             f'{either["share"]:.2%} ({either["views_with_dropped"]:,} of {doc["views"]:,} views); '
             f'unknown item {doc["unknown_item_share"]:.2%}, unknown champion {doc["unknown_champion_share"]:.2%}',
             f'extrapolated: full cache build {ex["cache_full_build"]:.2f} GiB; raw FIT (estimate) {ex["raw_fit_estimate"]:.1f} GiB',
             f'NaN: {len(nan["outside_allowed"])} cell group(s) outside tokens.NAN_ALLOWED; largest shares '
             + (', '.join(f'{name} {s:.1%}' for s, name in shares[:5]) or 'none'),
             f'constant columns: summary {len(constants["summary"])}, players_num {len(constants["players_num"])} '
             '(over this smoke run only)']
    lines += [f'figure: {path.parent / f["file"]}' for f in doc['figures'].values()]
    lines += [f'FLAG {f["code"]}: {f["message"]}' for f in doc['flags']]
    return lines


def main(argv=None) -> int:
    """명령행 인수를 보고 입력에 연결하고 거부·성공 상태를 반환한다. 원시 기록 측정을 요청한 경우에만 지정된 E6 검사 경로를 따른다."""
    ap = argparse.ArgumentParser(prog='python -m c_ept.report_s2',
                                 description='S2 smoke report: S2_REPORT.json and four figures of an S2 run.')
    ap.add_argument('run', type=Path, help='the S2 run folder')
    ap.add_argument('--s1', type=Path, help='the S1 run folder (default: the one named in RUN.json, next to the S2 run folder); read only')
    ap.add_argument('--terrain', type=Path, dest='terrain_dir', help='the terrain folder (default <run>/terrain); read only')
    ap.add_argument('--report-dir', type=Path, help='folder of the figures (default <run>/report)')
    ap.add_argument('--json', type=Path, dest='json_path', help='path of the report (default <run>/S2_REPORT.json)')
    ap.add_argument('--no-matches', action='store_true',
                    help='read files only: skip level_lag and building_kill_matching, which read the raw matches of the smoke run')
    ap.add_argument('--no-verify', action='store_true', help='do not hash the store files before reading them')
    ap.add_argument('--raw-fit-matches', type=int, default=RAW_FIT_MATCHES,
                    help='size of the raw FIT cohort for the estimate (default 162,000)')
    a = ap.parse_args(argv)
    if a.raw_fit_matches < 1:
        ap.error(f'--raw-fit-matches must be at least 1, got {a.raw_fit_matches}')
    strata_error = ()                               # an empty tuple in an except clause catches nothing
    if not a.no_matches:
        from c_ept.workflows.run_s1 import StrataRunning, require_no_strata     # here: a report that reads files only never needs the runner chain
        strata_error = StrataRunning
        try:
            require_no_strata()                                     # E6: before any raw match is read (the measurement checks it too)
        except StrataRunning as e:
            print(f'{e} (The raw-match measurements need it; --no-matches reads files only.)', file=sys.stderr, flush=True)
            return 2
    try:
        path = report(a.run, s1=a.s1, terrain_dir=a.terrain_dir, report_dir=a.report_dir, json_path=a.json_path,
                      records=None if a.no_matches else reader.read_match, verify=not a.no_verify, raw_fit_matches=a.raw_fit_matches)
    except (ReportInputError, FileNotFoundError) as e:                 # the inputs do not fit: say why, write nothing
        print(f'report refused: {e}', file=sys.stderr, flush=True)
        return 2
    except strata_error as e:                                         # a model server started between the check above and the measurement
        print(f'{e} (The raw-match measurements need it; --no-matches reads files only.)', file=sys.stderr, flush=True)
        return 2
    for line in _summary(json.loads(path.read_text(encoding='utf-8')), path):
        print(line, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
