"""Shared parts of the measurement scripts of ``c_ept.checks``: the S1 run folder, the match lists, the level comparison and the output.

Nothing runs at import. The match lists are read with numpy from the arrays ``matches/key``, ``matches/patch`` and ``matches/role`` of an S1
store (the store of the run, or the golden lock, which has the same layout); the keys stay in memory and are never printed or written. Raw
matches are read only through ``reader.read_match`` with its default root. A QA check (``sample_matches``, ``sweep_s1``, ``sweep_state``)
reads only the matches of the roles in ``QA_ROLES`` = FIT, STOP, DEV (decision B21 of the S4b preparation): the matches of TEST, CAL,
AUDIT, EXTERNAL and UNUSED are neither listed nor read, and ``qa_roles`` refuses a request for any of them.


방법에서의 위치. 이 모듈은 진단에 쓰는 S1 목록 선택과 레벨 시계 대조·출력을 공통화한다.
레벨 시계는 기준 시각까지의 레벨 상승 사건을 반영한 값이고, 프레임 레벨은 정기 기록의 값이다(Task 9).
입력은 S1 목록·경기 기록이고, 출력은 수치 집계와 선택적 불일치 셀 목록이다.
식 목록.
    (CHK-1) 프레임별 레벨 불일치와 같은 ms의 한 레벨 차이 분류·집계.
공통 가정. 프레임 k와 선수 행 p는 0부터 시작하며 참가자 ID는 p + 1이다.
frame_num의 레벨은 0부터 시작하는 열 2이고, 이 함수는 판독 기록의 원래 float64 값과 시계 값을 먼저 비교한다.
반환 목록의 정수 변환은 비교 뒤에 수행하므로 원래 비교를 정수 비교로 바꾸지 않는다.
목록 선택은 파일 존재를 확인하지만 여기서 저장소 전체 해시 검증을 수행하지는 않는다.
진단의 분류는 정책 결정의 근거 자료이고, 다른 원인의 중단 여부를 자동 집행하는 관문은 아니다(Task 9 fix1 M4).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept.data import reader
from c_ept.preprocessing import traits

STORE_DIR = 's1'          # the store folder under a run folder (= run_s1.STORE_DIR); the golden lock has the same layout under golden/
QA_ROLES = ('FIT', 'STOP', 'DEV')   # the only roles a QA check reads (B21); TEST, CAL, AUDIT, EXTERNAL and UNUSED are never read
ALL_ROLES = 'all'                   # the named argument of s1_matches that lists every role (not for a QA check)
LEVEL_KEYS = ('matches', 'frames', 'cells', 'level_ups', 'level_ups_on_frame_ms', 'mismatch_cells', 'mismatch_lag_by_one_at_frame_ms',
              'mismatch_other')
LAG, OTHER = 'lag_by_one_at_frame_ms', 'other'      # the two causes of a difference between the level clock and a frame


# ---- the S1 run and its match lists --------------------------------------------------------------------------------------------------
def run_folder(path) -> Path:
    """The S1 run folder; ``path`` may name the ``s1`` store inside it (a folder that holds STORE.json).

    S1–S2 판정 R37의 진단 입력에서 저장소 폴더와 실행 폴더 표기를 공통 실행 경로로 맞춘다.
    """
    path = Path(path)
    return path.parent if (path / 'STORE.json').is_file() else path


def qa_roles(roles=None) -> tuple:
    """``roles`` (default ``QA_ROLES``) as a tuple, after a check that every one is in ``QA_ROLES``: a QA check never asks for TEST, CAL,
    AUDIT, EXTERNAL or UNUSED. A bare string, an empty choice or a role outside ``QA_ROLES`` is an error that names the role and never a key.

    수식
    ----
    (QA-1) 허용 역할 R_qa = {FIT, STOP, DEV}; 요청 R_req는 R_req ≠ ∅ 이고 R_req ⊆ R_qa 일 때만 받는다.

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    R_qa | 품질 점검(QA)이 읽을 수 있는 역할의 집합 | — | 세 역할 | QA_ROLES
    R_req | 호출자가 요청한 역할의 집합 | — | 비어 있지 않은 부분집합이어야 함 | roles

    해석
    ----
    식 (QA-1)은 결정 B21의 문턱이다. 점검 도구가 교정(CAL)·감사(AUDIT)·시험(TEST) 표본의 원시 기록을 열어 보면 그 표본이
    더는 봉인 상태가 아니므로, 도구가 실수로 그런 역할을 요청하는 일 자체를 입구에서 막는다. 문자열 하나를 넘기면 글자마다 역할로
    읽히는 사고가 나므로 따로 거부한다.

    근거
    ----
    S4b 준비 계획 B21(QA 점검이 CAL·AUDIT 원시 기록을 읽는다, A4 Q6), Task M15.
    """
    if roles is None:
        return QA_ROLES
    if isinstance(roles, (str, bytes)) or not hasattr(roles, '__iter__'):
        raise TypeError(f'roles must be a sequence of role names taken from QA_ROLES {QA_ROLES}, not a single string')
    roles = tuple(roles)
    outside = [r for r in roles if not isinstance(r, str) or r not in QA_ROLES]
    if not roles or outside:
        raise ValueError(f'a QA check reads only the roles of QA_ROLES {QA_ROLES}; refused: '
                         f'{[str(r) for r in outside] if outside else "no role at all"}')
    return roles


def _matches(d: Path, roles=None) -> list:
    """``(key, patch, macro role)`` for each row of the arrays key, patch and role in ``d``, the ``matches`` folder of a store; with
    ``roles`` (a tuple of role names), only the rows of those roles. The role array is read first; the keys of the other rows are never
    decoded and never copied into Python objects (the key array is memory-mapped and only the selected rows are copied). The bytes of
    the other keys are in the file, and the operating system may page some of them in with their neighbours; nothing reads them."""
    names = ('key', 'patch', 'role')
    if not all((d / f'{n}.npy').is_file() for n in names):
        raise FileNotFoundError(f'no S1 store arrays (key, patch, role) under {d}')
    if roles is None:
        key, patch, role = (np.load(d / f'{n}.npy').tolist() for n in names)
        return [(k.decode('ascii'), C.PATCHES[p], C.ROLES[r]) for k, p, r in zip(key, patch, role)]
    patch, role = np.load(d / 'patch.npy'), np.load(d / 'role.npy')
    mask = np.isin(role, [C.ROLES.index(r) for r in roles])
    key = np.load(d / 'key.npy', mmap_mode='r')
    picked = np.array(key[mask])                                   # a copy of the rows asked for; the mapping is dropped below
    del key
    return [(k.decode('ascii'), C.PATCHES[p], C.ROLES[r]) for k, p, r in zip(picked.tolist(), patch[mask].tolist(), role[mask].tolist())]


def s1_matches(run, roles=QA_ROLES) -> list:
    """The matches of the S1 store of ``run`` as ``(key, patch, macro role)``, in the store's order, of the roles in ``roles``: by default
    ``QA_ROLES`` (FIT, STOP, DEV), and any subset of them (checked by ``qa_roles``; another role is a ValueError). The matches of every role
    are listed only for ``roles=ALL_ROLES`` (the string ``'all'``), a named argument for a builder that must see all of them and is not
    a QA check; no QA script passes it.

    S1–S2 판정 R37의 진단 대상 목록을 저장 순서대로 읽으며, 이 함수는 역할을 다시 계산하지 않는다.
    결정 B21: 기본값은 QA 역할이다. 다른 역할의 경기는 키를 해독하거나 Python 객체로 옮기지 않는다(키 배열의 바이트는 파일과 함께
    메모리에 대응되어 올 수 있으나 읽거나 복사하지 않는다). 모든 역할은 roles='all'로만 얻고, QA_ROLES 밖의 역할 이름은 오류이다.
    """
    d = run_folder(run) / STORE_DIR / 'matches'
    return _matches(d, None) if isinstance(roles, str) and roles == ALL_ROLES else _matches(d, qa_roles(roles))


def golden_matches(run, golden_dir=None) -> list:
    """The matches of the golden lock of ``run`` (``<run>/golden``, or ``golden_dir``), in the lock's order. A lock row whose role is not
    in ``QA_ROLES`` is a ValueError that names the role and never a key: ``measure_levels`` reads every match of the lock, and ``--golden-dir``
    accepts any lock. The default lock (the first 20 FIT matches of 15.18) is returned unchanged.

    S1–S2 판정 R37의 기준값 진단 목록을 선택한 잠금 폴더에서 읽는다.
    결정 B21(검토 I2): 잠금에 QA 역할(FIT·STOP·DEV) 밖의 경기가 있으면 원시 기록을 열기 전에 오류로 멈춘다. 기본 잠금은 FIT뿐이므로 결과가 같다.
    """
    base = Path(golden_dir) if golden_dir is not None else run_folder(run) / 'golden'
    rows = _matches(base / STORE_DIR / 'matches', None)
    outside = sorted({role for _, _, role in rows if role not in QA_ROLES})
    if outside:
        raise ValueError(f'the golden lock lists matches of the roles {outside}; a QA check reads only the roles of QA_ROLES {QA_ROLES}')
    return rows


def sample_matches(run, fit: int = 40, other: int = 20, roles=None) -> list:
    """The first ``fit`` FIT matches and the first ``other`` matches of every other role of ``roles``, per patch, by sorted key; the
    patches in the order of ``config.PATCHES`` and the roles in the order of ``config.ROLES``. ``roles`` is ``QA_ROLES`` (FIT, STOP,
    DEV) by default and may name fewer of them; a role outside ``QA_ROLES`` (TEST, CAL, AUDIT, EXTERNAL, UNUSED) is a ``ValueError``
    (``qa_roles``), and the keys of those matches are not decoded.

    S1–S2 판정 R37의 표본 진단용으로 패치·역할별 정렬 목록 앞부분을 고르며 무작위 표본은 아니다.
    결정 B21: 표본은 QA 역할(FIT·STOP·DEV)에서만 뽑는다. 이전에는 CAL·AUDIT 등 다른 모든 역할이 `other`개씩 들어 있었다.
    """
    roles = qa_roles(roles)
    groups: dict = {}
    for key, patch, macro in s1_matches(run, roles):
        groups.setdefault((patch, macro), []).append(key)
    return [(key, patch, macro) for patch in C.PATCHES for macro in C.ROLES if macro in roles
            for key in sorted(groups.get((patch, macro), ()))[:fit if macro == 'FIT' else other]]


def records(matches, limit=None):
    """The ``MatchRecord`` of the first ``limit`` of ``matches`` (all if None), read with ``reader.read_match`` and its default root;
    the reader refuses a holdout patch.

    No handler here, and none in the three scripts that loop over these records (``measure_levels``, ``measure_levels_wide``,
    ``explain_mismatch``): a match that cannot be used ends the script. That is safe because every error ``read_match`` raises names a
    code, or the kind of file and the patch, and never the key or the path (``MatchRejected``, ``MatchFormatError``, ``MatchReadError``),
    so the traceback of the script shows none. ``sweep_s1`` and ``sweep_state`` count their errors by class and go on.

    S1–S2 판정 R37의 진단 기록을 판독기에 위임하며, limit가 None이면 전부이고 0이면 읽는 기록이 없다.
    오류를 이 반복기에서 집계하거나 무시하지 않으며, 음수 limit는 일반 목록 절단 의미를 그대로 갖는다.
    """
    for key, patch, _ in matches[:limit]:
        yield reader.read_match(key, patch)


# ---- the level clock against the frames ----------------------------------------------------------------------------------------------
def new_level_counts() -> Counter:
    """A ``Counter`` with every key of ``LEVEL_KEYS`` at 0 (``update`` keeps the zeros, so the output always has every key).

    S1–S2 판정 R37의 공통 레벨 집계 키를 빈 진단에서도 보이도록 초기화한다.
    """
    return Counter({key: 0 for key in LEVEL_KEYS})


def compare_levels(rec, cells: list | None = None) -> Counter:
    """``traits.level_at`` against the frame level at every regular frame of one match.

    Counts (``LEVEL_KEYS``): the match, its regular frames, its player-frames (``cells``: 10 per frame), its LEVEL_UPs, those that fall on
    exactly a regular frame's ms, and the player-frames that differ, split by cause: ``lag_by_one_at_frame_ms`` (the frame shows one level
    less than the clock and a LEVEL_UP of that player falls on exactly the frame's ms) or ``other``. If ``cells`` is a list, every
    differing player-frame is appended to it as a dict of numbers (frame index, ms, participant id, both levels, the cause).


    수식
    ----
    (CHK-1) diff_{frame_row,player_row} = 1[clock_level ≠ frame_level];
            lag_{frame_row,player_row} = diff · 1[frame_ms ∈ up_times_{participant_id}
                                                  그리고 int(clock_level) = int(frame_level) + 1];
            mismatch_cells = Σ diff, mismatch_lag = Σ lag, mismatch_other = Σ(diff − lag).
            frames = len(frame_ts), cells = 10·frames;
            level_ups = Σ len(up_times), level_ups_on_frame_ms = Σ_{up_entry} 1[up_ms ∈ set(frame_ts)].

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    frame_row, player_row | 프레임·선수의 0부터 시작하는 행 | — | 프레임 범위; 선수 0–9 | k, p
    participant_id | 선수의 참가자 ID | — | 1–10, player_row + 1 | pid
    frame_ms, up_ms | 프레임·레벨 상승 시각 | ms | 기록 값 | ms, t
    frame_ts | 정기 프레임의 시각 배열 | ms | 기록 순서 | rec.frame_ts
    clock_level, frame_level | 시계·프레임의 레벨 | 레벨 | 원본/LevelIndex 계약 | got[p], want[p]
    up_times, up_entry | 참가자별 상승 시각 목록·사건 항목 | ms 목록; — | 없으면 빈 목록 | ups, e
    diff, lag | 원값 차이·동시각 한 레벨 차이 표시 | — | 0 또는 1 | got != want, cause
    mismatch_cells, mismatch_lag, mismatch_other | 전체·동시각 분류·나머지 불일치 수 | 셀 | 0 이상 | c['mismatch_cells'], c['mismatch_lag_by_one_at_frame_ms'], c['mismatch_other']
    frames, cells | 프레임 수·열 선수 셀 수 | 프레임; 셀 | 0 이상 | c['frames'], c['cells']
    level_ups, level_ups_on_frame_ms | 상승 사건 수·프레임 시각과 같은 사건 수 | 사건 | 0 이상 | c['level_ups'], c['level_ups_on_frame_ms']
    1[·], int | 조건의 표시·Python 정수 변환 | — | 0/1; 변환 가능 값 | 조건식, int

    해석
    ----
    식 (CHK-1)은 레벨 시계와 프레임 기록의 차이를 먼저 찾고 같은 ms의 한 레벨 차이만 따로 센다(Task 9 fix1 M4).
    원값이 다른 경우에만 정수 변환으로 분류하므로, 비정상 소수 레벨을 사전에 정수로 보정하는 검사는 아니다(Task 9 대조 계약).
    다른 원인은 나머지 분류에 남기며 이 함수 자체가 실행을 중단하지 않는다(Task 9 fix1 M4).
    프레임이 없으면 프레임·셀·불일치 수는 초기값이고, 상승 사건 수는 여전히 기록에서 센다(Task 9 진단 범위).
    셀별 목록 인수가 None이면 셀별 목록은 만들지 않고, 빈 목록을 넘기면 발견한 셀을 그 목록에 추가한다(Task 9 fix1 M4).

    근거
    ----
    S1–S2 계획 Task 9(레벨 대조), S1–S2 판정 R37 및 Task 9 fix1 M4·M5(원인 기록).
    """
    index = traits.LevelIndex(rec)
    ups: dict = {}
    for e in rec.events:
        if e['type'] == 'LEVEL_UP':
            ups.setdefault(e['participantId'], []).append(e['timestamp'])
    frame_ms = set(rec.frame_ts.tolist())
    c = new_level_counts()
    c['matches'] = 1
    c['level_ups'] = sum(len(v) for v in ups.values())
    c['level_ups_on_frame_ms'] = sum(1 for v in ups.values() for t in v if t in frame_ms)
    for k, ms in enumerate(rec.frame_ts.tolist()):
        got, want = index.at(ms), rec.frame_num[k, :, 2]
        c['frames'] += 1
        c['cells'] += 10
        # (CHK-1)의 원값 차이를 먼저 찾고, 같은 ms의 정수 한 레벨 차이는 원인 분류에만 쓴다(Task 9 fix1 M4).
        for p in np.flatnonzero(got != want).tolist():
            pid = p + 1
            on_ms = ms in ups.get(pid, ())
            cause = LAG if on_ms and int(got[p]) == int(want[p]) + 1 else OTHER
            c['mismatch_cells'] += 1
            c['mismatch_lag_by_one_at_frame_ms' if cause == LAG else 'mismatch_other'] += 1
            if cells is not None:
                cells.append({'frame': k, 'ms': ms, 'pid': pid, 'level_at': int(got[p]), 'frame_level': int(want[p]),
                              'level_up_on_frame_ms': on_ms, 'cause': cause})
    return c


# ---- the command line and the output -------------------------------------------------------------------------------------------------
def arguments(description: str, *, golden: bool = False, limit: bool = False) -> argparse.ArgumentParser:
    """The parser every script starts from: the S1 run folder and ``--out``; ``golden`` adds ``--golden-dir``, ``limit`` ``--limit``.

    S1–S2 판정 R37의 공통 진단 인자를 정의하며, 파서 생성 자체로 E6 검사를 수행하지 않는다.
    """
    ap = argparse.ArgumentParser(description=description.split('\n\n')[0],
                                 epilog='Prints counts or per-cell numbers only (never a match key or an identifier). '
                                        'Refuses to start while a Strata server runs (E6).')
    ap.add_argument('run', help='the S1 run folder (or its s1 store): it names the matches to read; nothing is written into it')
    ap.add_argument('--out', type=Path, help='also write the counts to this file as JSON')
    if golden:
        ap.add_argument('--golden-dir', type=Path, help='the golden folder to read the lock from (default: <run>/golden)')
    if limit:
        ap.add_argument('--limit', type=int, help='read only the first N matches (a smoke test)')
    return ap


def parse(ap: argparse.ArgumentParser, argv) -> argparse.Namespace:
    """``ap.parse_args(argv)``, and an argument error (exit code 2, before any work) if ``--out`` would be a file inside the S1 run
    folder: the run is read only.

    S1–S2 판정 R37의 읽기 전용 실행 폴더를 보호하도록 출력 경로의 포함 관계를 검사한다.
    """
    args = ap.parse_args(argv)
    if args.out is not None and args.out.resolve().is_relative_to(run_folder(args.run).resolve()):
        ap.error('--out may not be a file inside the S1 run folder: the run is read only')
    return args


def refuse(message: str) -> int:
    """Say why the script does not run (on stderr) and return its exit code 2, as ``run_s1.main`` does.

    S1–S2 판정 R37의 진단 거부 사유와 종료 상태를 반환하는 출력 경로이다.
    """
    print(message, file=sys.stderr, flush=True)
    return 2


def write_json(path, data) -> None:
    """``data`` as sorted, indented JSON (UTF-8, Unix line ends) at ``path``. Counts or per-cell numbers without identifiers are the
    caller's responsibility; this serializer does not sanitize keys or values.

    S1–S2 판정 R37의 수치 결과를 직렬화하며, 전달받은 객체에서 식별자를 자동 제거하는 검사는 없다.
    """
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')


def finish(args, script: str, title: str, counts, **extra) -> int:
    """Print ``counts`` (and the dicts in ``extra``) and, with ``--out``, write them as JSON with the script name and the name of the run
    folder (not its path). Returns the exit code 0.

    S1–S2 판정 R37의 집계를 표시하고 요청된 경우 JSON으로 저장하며, 집계 내용의 연구 관문을 판정하지 않는다.
    """
    print(title)
    for key in sorted(counts):
        print(f'  {key:44s} {counts[key]:>12,}')
    for name, value in extra.items():
        if isinstance(value, dict):
            print(f'{name}: {value or "none"}')
    if args.out is not None:
        write_json(args.out, {'script': script, 'run': run_folder(args.run).name, 'counts': dict(counts), **extra})
    return 0
