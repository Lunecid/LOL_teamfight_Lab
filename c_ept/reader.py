"""Raw Match-V5 timeline and detail files -> one ``MatchRecord`` per match (input of stages S1 and S2).

Only what the model needs is read: from the detail file the participant ids, teams, champion ids and the rune/spell entries that the
item replay needs; from the timeline file the frames (``participantFrames``) and the events. Player identifiers (riot id, summoner name)
are never read, and the puuids of the two files are only compared with each other in memory (``_same_match``), never copied into a
record, a message or a log, so no record can carry an identifier. The cache files (featV7) are not used here; they only serve as a check.

Four kinds of failure, kept apart so that a counted rejection is never a disguised format problem:
``MatchRejected(code)`` is a real match that cannot be used (a file is missing, not 10 participants, a wrong pid/team layout, no
GAME_END, a timeline and a detail file that are not of one match, a regular frame without all ten players or with a player whose entry
holds no value); ``MatchFormatError`` (a ValueError) is a file that is not a Match-V5 file as expected (not UTF-8 JSON, say), or whose
clock breaks what every real match shows (see ``read_match``); ``MatchReadError`` (an OSError) is a file that is there and cannot be
read (permission denied, locked by another process, an I/O error); ``HoldoutRefused`` (a ValueError) is a call for a holdout patch
(15.22, 16.13) without ``allow_holdout=True``. None of them, and nothing else that this module raises for a file it cannot read, carries
the path or any text of the file in its message, ``repr``, ``args``, cause or context: the name of a raw file is the match key and its
text holds the players' identifiers (``_read_json``). One attribute holds a key: ``MatchFormatError.key``, by that class's contract (the
caller must know which match is malformed; it travels in the pickle to a worker pool's parent) and never in any text of the error;
``MatchRejected``, ``HoldoutRefused`` and ``MatchReadError`` keep no key at all.

Clock: the last frame of a timeline is the game-end frame; ``read_match`` checks that it holds GAME_END. It is not a regular frame:
only its positions are kept, as ``end_xy``. ``T_ms`` is the time of GAME_END, which is the last event, and every regular frame is
before it (``frame_ts[-1] < T_ms``); the first frame is at 0 ms and the frame times strictly increase. All of this is checked.

방법에서의 위치. 이 모듈은 원시 Match-V5 timeline·detail 파일 한 쌍을 경기 하나의 입력 기록(MatchRecord)으로 읽는다(S1·S2의 입력; 설계서 11절 항목 1·4·5·9, S1–S2 계획 Task 2). 기록에는
모형이 쓸 값만 담는다: 참가자 번호별 팀과 챔피언 번호, 정규 프레임마다 참가자 10명의 프레임 값, 경기 끝 프레임의 위치, 시각 순으로 늘어놓은 모든 사건이다. 두 파일이 한 경기의 것인지,
시계가 실제 경기에서 보이는 꼴인지, 프레임마다 참가자 10명의 값이 있는지를 읽는 자리에서 검사하여, 어긋난 경기는 이유 코드로 거부하거나 형식 오류로 멈춘다.

용어. 정규 프레임: 1분마다 찍은 경기 상태 프레임(분 프레임)이며 마지막 경기 끝 프레임은 정규 프레임이 아니다. 사건: timeline이 기록한 킬, 오브젝트 따위의 이벤트이다. 경기 끝 T: 마지막
사건인 GAME_END의 시각이다.

식 목록. 식 번호는 (모듈 약어-번호) 꼴이며 식과 기호는 docs/superpowers/notes/equation-index.md의 식 색인과 기호 등록부에 올라 있다.
    (RD-1)  챔피언 이름 번호  id(name) = 1 + (hash32(norm(name)) mod max(1, vocab − 1)),   norm(name)이 빈 문자열이면 id = 0
    (RD-2)  경기 시계         ts_0 = 0,   ts_k > ts_{k−1}   (k = 1, …, F),   ts_{F−1} < T   (F ≥ 1),   T = 마지막 사건인 GAME_END의 시각
    (RD-3)  완전한 프레임     정규 프레임마다 참가자 1–10 모두에게 숫자로 읽히는 값이 하나 이상 있다

공통 가정.
    (가) 식별자(riot id, 소환사 이름, puuid)는 기록에 담기지 않는다. 두 파일의 puuid는 같은 경기인지 메모리에서 맞대는 데만 쓰고 반환하거나 글에 넣지 않는다(S1–S2 계획 전역 제약).
    (나) 읽을 수 없는 파일에 대해 이 모듈이 내는 오류는 경기를 이름으로 말하지 않는다: 글, repr, args, 원인 어디에도 경기 키, 파일 경로, 파일 내용이 없다. 키는 MatchFormatError의 속성에만
         있다(S1–S2 판정 14, 33, 41과 재잠금 전 커밋 aa72786; S1–S2 progress.md의 R41 기록).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 11, items 1, 4, 5 and 9 (raw data, missing positions, game end T,
#      frame features).
# Ref: LEGACY data/cache_io.py::_extract_static_meta_from_detail (the format of runes_by_pid and the hash of champion_name_id), so that
#      the record's runes_by_pid equals the cache meta static_meta.runes_by_pid and can be handed to the legacy item replay.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from c_ept import config as C

NAME_VOCAB = 4096          # LEGACY CHAMPION_NAME_VOCAB: champion_name_id lies in [1, NAME_VOCAB - 1]
HOLDOUT_ROLES = {'15.22': 'TEST', '16.13': 'EXTERNAL'}     # the patches whose raw files are opened once, at S8 (macro roles TEST, EXTERNAL)
HOLDOUT_PATCHES = tuple(HOLDOUT_ROLES)
_NAN = float('nan')


class MatchRejected(Exception):
    """A match that cannot enter the data set. ``code`` says why: missing_file, participants_not_10, pid_team_layout, no_game_end,
    timeline_detail_mismatch, incomplete_frame (later stages add their own, e.g. same_team_kill).

    쓸 수 없는 실제 경기를 이유 코드와 함께 거부하는 신호이다. 이유별로 세어 보고하는 대상이며 형식 문제를 가리는 데 쓰지 않는다(모듈 머리).
    """

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class HoldoutRefused(ValueError):
    """``read_match`` was asked for a match of a holdout patch (15.22 TEST, 16.13 EXTERNAL) without ``allow_holdout=True``.

    Not a rejection and not a format error: the call itself is wrong, and no file was looked for. The message names the patch and the rule
    and never the match key. Only the constant ``True`` opens the gate: 1, "yes" or any other truthy value is refused too.

    보류 패치를 읽으려는 호출이 잘못이라는 신호이다. 파일을 찾기 전에 거부하고 상수 True만 문을 연다(S1–S2 판정 36; 보류 패치의 원시 파일은 S8에서 한 번 열기로 정해져 있다).
    """

    def __init__(self, patch: str):
        role = HOLDOUT_ROLES.get(patch, 'holdout')
        super().__init__(f'patch {patch} ({role}) is a holdout patch: its raw files are opened once at S8, so read_match refuses it '
                         f'unless allow_holdout=True is passed')
        self.patch = patch

    def __reduce__(self):                  # the default would call __init__(message): keep it picklable for worker pools
        return type(self), (self.patch,)


class MatchFormatError(ValueError):
    """A raw file that is not a Match-V5 file as expected (an API error body saved as a file, a file that is not UTF-8 JSON, a shifted
    clock, a missing champion id, ...). Not a rejection: nothing is counted and skipped, the caller sees the error.

    ``key`` and ``patch`` say which match; ``reason`` says what is wrong. The message holds the reason and the patch only, never the
    match key, so that logs and tracebacks do not carry match ids.

    Match-V5 형식이 기대와 다른 파일을 알린다. 거부와 달리 세어서 건너뛰지 않고 호출한 쪽이 보도록 멈춘다.
    """

    def __init__(self, key: str, patch: str, reason: str):
        super().__init__(f'{reason} (patch {patch})')
        self.key, self.patch, self.reason = key, patch, reason

    def __reduce__(self):                  # the default would call __init__(message) alone: keep it picklable for worker pools
        return type(self), (self.key, self.patch, self.reason)


class MatchReadError(OSError):
    """A raw file that is there but cannot be read: permission denied, locked by another process, an I/O error, ...

    An error and not a rejection: nothing is counted and skipped, the run stops, as it did on the native OSError. The native error names
    the file in its message, in ``filename`` and in a pickle, and the name of a raw file is the match key; so the reader replaces it with
    this one, raised ``from None`` outside the ``except`` clause (no cause, no context). The message holds the kind of file (``timeline``
    or ``detail``), the class name and the errno of the native error and the patch, and nothing else: the object carries neither the
    path nor the key. ``errno`` is the native one.

    있는데 읽을 수 없는 파일을 알린다. 원래의 OSError는 파일 이름, 곧 경기 키를 담으므로 키가 없는 이 오류로 바꾸고, 거부가 아니라 오류이므로 실행이 멈춘다.
    """

    def __init__(self, kind: str, error: str, errno_: int | None, patch: str):
        super().__init__(f'{kind} file cannot be read: {error}, errno {errno_} (patch {patch})')
        self.errno = errno_
        self.kind, self.error, self.patch = kind, error, patch

    def __reduce__(self):                  # the default would call __init__(message) alone: keep it picklable for worker pools
        return type(self), (self.kind, self.error, self.errno, self.patch)


@dataclass(frozen=True, eq=False)
class MatchRecord:
    """Everything the later stages take from one raw match. Arrays are indexed by participant id - 1 (1-5 blue 100, 6-10 red 200).

    ``frame_num`` holds ``config.FRAME_FIELDS`` per regular frame (NaN where the file has no value). ``events`` are all timeline
    events as in the file (dicts), ordered by time; events of the same ms keep the file order. Records compare by content
    (NaN equals NaN) and are not hashable.

    Every numpy array of a record is read-only (``setflags(write=False)``, also after pickling or copying): take a copy to change
    values. An array handed to the constructor (or to ``dataclasses.replace``) is made read-only in place. The dicts are not frozen:
    ``events`` is a tuple of plain dicts as parsed from the file and ``runes_by_pid`` a plain dict. They are only not to be modified.

    경기 하나의 입력 기록이다. 플레이어 식별자를 담지 않고(경기 키는 key 필드에만 있으며 repr에서는 가려진다) 배열은 읽기 전용이어서 읽은 뒤 값이 바뀌지 않는다(공통 가정 (가)).
    """
    key: str
    patch: str
    T_ms: int
    team: np.ndarray            # (10,) int16, 100 / 200
    champ: np.ndarray           # (10,) int32, Data Dragon champion key
    runes_by_pid: dict          # {'1': {...14 fields...}, ..., '10': ...} as in the cache meta static_meta.runes_by_pid
    frame_ts: np.ndarray        # (F,) int64, regular frames (the game-end frame is not among them)
    frame_num: np.ndarray       # (F, 10, 33) float64
    end_xy: np.ndarray          # (10, 2) float64, positions in the game-end frame
    events: tuple               # tuple of dict

    def __post_init__(self):
        self._freeze_arrays()

    def __setstate__(self, state):         # unpickling and copy.deepcopy do not run __init__: freeze here as well
        self.__dict__.update(state)
        self._freeze_arrays()

    def _freeze_arrays(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v.setflags(write=False)

    def __eq__(self, other):
        if not isinstance(other, MatchRecord):
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                    return False
                if a.dtype != b.dtype or a.shape != b.shape:
                    return False
                if not np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'):
                    return False
            elif a != b:
                return False
        return True

    def __repr__(self):
        # The match key is left out on purpose: a record turns up in assertion explanations ("where x = MatchRecord(...).patch"), in
        # tracebacks and in logs, and a match id must not.
        return (f'MatchRecord(key=<redacted>, patch={self.patch!r}, T_ms={self.T_ms}, frame_num={tuple(np.shape(self.frame_num))}, '
                f'events={len(self.events)})')


# ---- small helpers ---------------------------------------------------------------------------------------------------------
def _int(x, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError):
        return default


def _is_int(x) -> bool:
    """A JSON integer (not a float, a string or a bool)."""
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def champion_name_id(name) -> int:
    """Stable id of a champion name: blake2b of the lower-case stripped name, folded into [1, NAME_VOCAB - 1]; 0 for no name.

    수식
    ----
    (RD-1)  id(name) = 1 + (hash32(norm(name)) mod max(1, vocab − 1)),    id(name) = 0   (norm(name)이 빈 문자열일 때)
            norm(name) = str(name or '').strip().lower()이다. hash32는 이 문자열의 UTF-8 바이트에 대한 BLAKE2b 4바이트 요약을 big-endian 부호 없는 정수로 읽은 값이다.

      기호       | 뜻                                    | 단위 | 범위               | 코드 변수
      name       | 챔피언 이름(없으면 빈 이름)           | —    | 문자열 또는 없음   | name
      norm(name) | str(name or '')의 공백을 떼고 소문자로 바꾼 이름 | —    | 문자열(없음이면 빈 문자열) | s
      hash32     | BLAKE2b 4바이트 요약을 정수로 읽은 값 | —    | 0, …, 2^32 − 1     | int(hexdigest, 16)
      vocab      | 번호의 어휘 크기                      | 개   | 정수(내장 4096)    | NAME_VOCAB
      id(name)   | 챔피언 이름의 번호                    | —    | 0, 1, …, vocab − 1 | 반환값

    해석
    ----
    식 (RD-1)은 챔피언 이름을 해시로 접어 1 이상 vocab − 1 이하의 번호로 바꾸고 이름이 없으면 0으로 둔다. 같은 이름은 언제나 같은 번호를 받으며 번호는 이름의 목록이나 읽는 순서에
    의존하지 않는다. 서로 다른 이름이 같은 번호가 되는 충돌은 코드가 검사하지 않는다. 이 번호는 룬 기록의 한 필드로, 캐시의 형식과 같게 만들어 옛 아이템 재생에 넘기기 위한
    것이다(모듈 머리의 Ref).

    근거
    ----
    S1–S2 계획 Task 2(룬 기록을 캐시 meta와 같은 형식으로), 설계서 11절 항목 1(캐시는 맞대보기에만 쓴다).
    Ref: Saarinen, M-J. and Aumasson, J-P. (2015), "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", RFC 7693;
         doi:10.17487/RFC7693 (BLAKE2b).
    """
    s = str(name or '').strip().lower()
    if not s:
        return 0
    return int(hashlib.blake2b(s.encode('utf-8'), digest_size=4).hexdigest(), 16) % max(1, NAME_VOCAB - 1) + 1


def _runes_entry(p: dict) -> dict:
    """The 14 values of one participant in the cache's runes_by_pid format (0 where the file has none)."""
    perks = p.get('perks') or {}
    styles = perks.get('styles') or []
    stat = perks.get('statPerks') or {}

    def style_at(i: int) -> int:
        try:
            return _int(styles[i].get('style'))
        except (LookupError, AttributeError, TypeError):
            return 0

    def perk_at(i: int, j: int) -> int:
        try:
            return _int(styles[i]['selections'][j]['perk'])
        except (LookupError, AttributeError, TypeError):
            return 0

    return {
        'champion_name_id': champion_name_id(p.get('championName')),
        'summoner_spell_1_id': _int(p.get('summoner1Id')),
        'summoner_spell_2_id': _int(p.get('summoner2Id')),
        'primary_style_id': style_at(0),
        'sub_style_id': style_at(1),
        'primary_rune_1': perk_at(0, 0),
        'primary_rune_2': perk_at(0, 1),
        'primary_rune_3': perk_at(0, 2),
        'primary_rune_4': perk_at(0, 3),
        'sub_rune_1': perk_at(1, 0),
        'sub_rune_2': perk_at(1, 1),
        'stat_perk_offense': _int(stat.get('offense')),
        'stat_perk_flex': _int(stat.get('flex')),
        'stat_perk_defense': _int(stat.get('defense')),
    }


def _read_json(path: Path, kind: str, key: str, patch: str):
    """The parsed JSON of one raw file (``kind``: ``'timeline'`` or ``'detail'``), and never an exception that holds its path or its text.

    The native exceptions carry both. An OSError names the path in its message, its traceback, ``filename`` and a pickle; a
    UnicodeDecodeError shows the whole file in its ``repr`` and ``args``; a JSONDecodeError keeps it in ``doc``; and the path is the match
    key while the text holds the key, the puuids, riot ids and summoner names. A pool of workers sends its exceptions to the parent as
    pickles. So they are replaced here, after the ``except`` clauses and ``from None`` (no cause, no context): an OSError by a
    ``MatchReadError`` (kind, class and errno of the native error, patch), a ValueError - a JSONDecodeError or a UnicodeDecodeError: the
    file is not UTF-8 JSON - by a ``MatchFormatError`` (kind, class and offset of the native error, patch). Anything else passes as it is.
    """
    problem = None
    try:
        with open(path, 'rb') as f:
            return json.load(f)
    except OSError as err:
        problem = MatchReadError(kind, type(err).__name__, err.errno, patch)
    except ValueError as err:
        offset = err.pos if isinstance(err, json.JSONDecodeError) else getattr(err, 'start', None)      # a character or a byte offset
        where = f' at offset {offset}' if isinstance(offset, int) else ''
        problem = MatchFormatError(key, patch, f'the {kind} file is not valid UTF-8 JSON: {type(err).__name__}{where}')
    raise problem from None


def _is_file(path: Path, kind: str, patch: str) -> bool:
    """``path.is_file()``, with the error of ``_read_json`` for an OSError that pathlib does not swallow (a permission error, say): it
    names the path, which is the match key."""
    problem = None
    try:
        return path.is_file()
    except OSError as err:
        problem = MatchReadError(kind, type(err).__name__, err.errno, patch)
    raise problem from None


def _participants(detail, fail) -> tuple[np.ndarray, np.ndarray, dict]:
    """(team, champ, runes_by_pid) of the 10 participants. ``fail(reason)`` builds the MatchFormatError to raise.

    MatchFormatError: no ``info.participants`` list, entries that are not objects, a participant without an integer ``championId``.
    MatchRejected: not 10 participants (participants_not_10); participant ids not 1-10 with 1-5 on team 100 and 6-10 on team 200
    (pid_team_layout).
    """
    info = detail.get('info') if isinstance(detail, dict) else None
    parts = info.get('participants') if isinstance(info, dict) else None
    if not isinstance(parts, list):
        raise fail('detail has no info.participants list')
    if len(parts) != 10:
        raise MatchRejected('participants_not_10')
    if not all(isinstance(p, dict) for p in parts):
        raise fail('info.participants holds entries that are not objects')
    by_pid = {_int(p.get('participantId')): p for p in parts}
    if sorted(by_pid) != list(range(1, 11)) or any(_int(p.get('teamId')) != (100 if pid <= 5 else 200) for pid, p in by_pid.items()):
        raise MatchRejected('pid_team_layout')
    order = range(1, 11)
    for i in order:
        if not _is_int(by_pid[i].get('championId')):
            raise fail(f'participant {i} has no integer championId')
    team = np.array([_int(by_pid[i].get('teamId')) for i in order], dtype=np.int16)
    champ = np.array([by_pid[i]['championId'] for i in order], dtype=np.int32)
    return team, champ, {str(i): _runes_entry(by_pid[i]) for i in order}


# frame value sources, in the order of config.FRAME_FIELDS: (section of participantFrames[pid] or None, key)
def _source(name: str) -> tuple:
    if name in ('x', 'y'):
        return 'position', name
    if name.startswith('cs_'):
        return 'championStats', name[3:]
    return None, name


_SOURCES = tuple(_source(n) for n in C.FRAME_FIELDS)
_COL_X, _COL_Y = C.FRAME_FIELDS.index('x'), C.FRAME_FIELDS.index('y')


def _fill_frame(out: np.ndarray, participant_frames) -> bool:
    """Write the values of one frame into ``out`` (10, 33); whatever the file does not have stays as it is (NaN).

    Returns whether every participant id 1-10 has an entry that gives at least one usable value (a position or a numeric field): False
    when ``participantFrames`` is not an object, a player has no entry (or one that is not an object), or an entry holds no number at
    all (``{}``, ``{"participantId": k}``, only null or non-numeric values), whose row then stays NaN. ``read_match`` rejects a regular
    frame like that (incomplete_frame); the game-end frame is exempt. An entry with only some of the values (no position, say) is
    complete: what it lacks is NaN.

    (RD-3)의 완전함을 판정하고 한 프레임의 값을 채운다.
    """
    if not isinstance(participant_frames, dict):
        return False
    complete = True
    for i in range(10):
        pf = participant_frames.get(str(i + 1))
        if not isinstance(pf, dict):
            complete = False
            continue
        sections = {None: pf, 'position': pf.get('position'), 'championStats': pf.get('championStats')}
        row = out[i]
        usable = False
        for j, (section, name) in enumerate(_SOURCES):
            d = sections[section]
            if isinstance(d, dict):
                value = _num(d.get(name))
                row[j] = value
                usable = usable or value == value                  # NaN is the one value that differs from itself
        complete = complete and usable
    return complete


def _same_match(timeline: dict, detail: dict) -> bool:
    """Whether a timeline and a detail file are the two files of one match. Both have been through ``_frames`` / ``_participants``: each
    is an object with an ``info`` object, and the detail lists the participant ids 1-10 once.

    Three comparisons, and a failed one is a difference: ``metadata.matchId`` (a value in one file only counts as a difference, a value
    in neither as none); ``info.gameId``, only when both files have one; for every participant id 1-10 the puuid of the detail, which the
    timeline's ``info.participants`` must list once, with the same puuid (so a timeline that lists the ids in another order is fine, one
    that swaps two puuids or leaves an id out or lists it twice is not). The identifiers live only inside this function: they are compared
    in memory and are neither returned, kept nor put in a message, so the answer is a bool and nothing else leaves.
    """
    def match_id(doc):
        meta = doc.get('metadata')
        return meta.get('matchId') if isinstance(meta, dict) else None

    if match_id(timeline) != match_id(detail):
        return False
    game_t, game_d = timeline['info'].get('gameId'), detail['info'].get('gameId')
    if game_t is not None and game_d is not None and game_t != game_d:
        return False
    entries = timeline['info'].get('participants')
    listed: dict = {}                                                  # participant id -> the puuids the timeline lists for it
    for p in entries if isinstance(entries, list) else ():
        if isinstance(p, dict):
            listed.setdefault(_int(p.get('participantId')), []).append(p.get('puuid'))
    return all(listed.get(_int(p.get('participantId'))) == [p.get('puuid')] for p in detail['info']['participants'])


def _frames(timeline, fail) -> tuple[list, list]:
    """(frames, events per frame) of a timeline; MatchFormatError unless ``info.frames`` is a list of objects with an integer
    ``timestamp`` whose ``events`` (absent or null: no events) are a list of objects with an integer ``timestamp``."""
    info = timeline.get('info') if isinstance(timeline, dict) else None
    frames = info.get('frames') if isinstance(info, dict) else None
    if not isinstance(frames, list):
        raise fail('timeline has no info.frames list')
    per_frame = []
    for k, fr in enumerate(frames):
        if not (isinstance(fr, dict) and _is_int(fr.get('timestamp'))):
            raise fail(f'frame {k} is not an object with an integer timestamp')
        evs = fr.get('events')
        evs = [] if evs is None else evs
        if not (isinstance(evs, list) and all(isinstance(e, dict) and _is_int(e.get('timestamp')) for e in evs)):
            raise fail(f'frame {k} has events that are not a list of objects with an integer timestamp')
        per_frame.append(evs)
    return frames, per_frame


def read_match(key: str, patch: str, root: Path = C.RAW_ROOT, *, allow_holdout: bool = False) -> MatchRecord:
    """Read one match from ``<root>/<patch>/raw/<region>/{timeline,detail}/<key>.json``.

    Raises HoldoutRefused (a ValueError) for the patches 15.22 (TEST) and 16.13 (EXTERNAL) unless ``allow_holdout`` is the constant
    ``True`` (``is True``: 1, "yes" or another truthy value stays refused); the refusal is the first thing the function does, before any
    file or folder is looked at. Their raw files are opened once, at S8; nothing in c_ept passes the switch yet.
    Raises MatchRejected for a real match that cannot be used, with the code
      missing_file         the timeline or the detail file does not exist;
      participants_not_10  ``info.participants`` is a list of another length than 10;
      pid_team_layout      the participant ids are not 1-10 with 1-5 on team 100 and 6-10 on team 200;
      no_game_end          ``info.frames`` is a list without a GAME_END event;
      timeline_detail_mismatch  the two files are not of one match: ``metadata.matchId`` differs, ``info.gameId`` differs (only when both
                           files have one) or the puuid of a participant id 1-10 differs (see ``_same_match``);
      incomplete_frame     a regular frame (not the game-end frame) has no ``participantFrames`` entry for some participant id 1-10, or an
                           entry that gives no usable value (no position and no numeric field, e.g. ``{}``; see ``_fill_frame``).
    The last two are checked after everything else, in this order, so a file that is rejected or raises an error for another reason is
    treated as before; the puuids are compared in memory and are never kept, shown or put in a message.
    Raises MatchFormatError (a ValueError; the message has the reason and the patch, not the key) for a file that is not a Match-V5
    file as expected: a file that is not UTF-8 JSON (the reason names the kind of file and the class and the offset of the native
    error, and nothing of the text); in the detail, no ``info.participants`` list (an API error body saved as a file, for example),
    entries that are not objects, a participant without an integer ``championId``; in the timeline, no ``info.frames`` list, a frame
    that is not an object with an integer ``timestamp``, ``events`` that are not a list of objects with an integer ``timestamp``; or a
    clock that breaks what every real match shows: the first frame is not at 0 ms, the frame timestamps do not strictly increase,
    GAME_END is not in the last frame, an event comes after GAME_END (``T_ms`` is the time of GAME_END, the last event), the last
    regular frame is not before the game end.
    Raises MatchReadError (an OSError; the message has the kind of file, the class and the errno of the native error and the patch,
    never the path) for a file that is there and cannot be read or looked up: permission denied, locked by another process, an I/O
    error. An error, not a rejection: the run stops.
    Other structure trouble in a file that parsed raises the native exception (KeyError, ...); nothing is hidden behind a rejection.

    수식
    ----
    (RD-2)  ts_0 = 0,   ts_k > ts_{k−1}   (k = 1, …, F),   ts_{F−1} < T   (F ≥ 1),   T = 시각 순으로 늘어놓은 사건의 마지막, 곧 GAME_END의 시각
            GAME_END는 마지막 프레임의 사건에도 있다. 정규 프레임은 k = 0, …, F − 1이고 ts_F는 경기 끝 프레임의 시각이다.
    (RD-3)  정규 프레임 k는 완전하다 ⇔ 모든 참가자 번호 pid = 1, …, 10에 대해 frame_num[k, pid − 1, ·]에 NaN이 아닌 값이 하나 이상 있다
            경기 끝 프레임(k = F)은 완전하지 않아도 된다. 값이 없는 칸은 NaN이다.

      기호                     | 뜻                                          | 단위 | 범위                  | 코드 변수
      ts_k                     | 프레임 k의 시각                             | ms   | ts_0 = 0, 엄격히 증가 | frame_times[k]
      F                        | 정규 프레임의 수(경기 끝 프레임은 뺀다)     | 개   | ≥ 0                   | len(frames) − 1
      T                        | 경기 끝: 마지막 사건 GAME_END의 시각        | ms   | > ts_{F−1}            | T_ms
      pid                      | 참가자 번호                                 | —    | 1, …, 10              | 배열의 위치 pid − 1
      frame_num[k, pid − 1, ·] | 정규 프레임 k에서 참가자 pid의 프레임 값 열 | 혼합 | 숫자 또는 NaN         | frame_num

    해석
    ----
    식 (RD-2)는 한 경기의 시계가 지킬 조건이다: 프레임은 0 ms에서 시작해 시각이 엄격히 늘고, 경기 끝 T는 마지막 사건인 GAME_END의 시각이며 모든 정규 프레임이 T보다 앞선다(설계서 11절
    항목 5). 이 조건은 모듈 머리가 실제 경기의 시계로 적은 것이어서, 어긴 파일은 보정하지 않고 형식 오류로 멈추고, 쓸 수 없는 실제 경기는 이유 코드로 거부하여 센다. 식 (RD-3)은 정규
    프레임마다 참가자 10명이 모두 값을 가질 것을 요구하여, 값이 하나도 없는 참가자의 행을 조용히 NaN 행으로 두지 않고 그 경기를 거부한다(S1–S2 판정 36). 보류 패치는 파일을 찾기도 전에
    거부하는데, 그 원시 파일은 S8에서 한 번만 열기로 정해져 있기 때문이다(S1–S2 계획 전역 제약; S1–S2 판정 36).

    근거
    ----
    설계서 11절 항목 4·5·9(위치 없음, 경기 끝 T, 프레임 특징), S1–S2 계획 Task 2와 검토 초점 1(어긋난 경기는 이유 코드로 거부),
    S1–S2 판정 36(빈 항목은 incomplete_frame, 보류 패치 문은 상수 True만).
    """
    if patch in HOLDOUT_ROLES and allow_holdout is not True:                   # the constant True only: 1, 'yes', [x] stay refused
        raise HoldoutRefused(patch)                                            # before anything else: no path is built, no file touched
    region = C.RAW_REGION.get(patch)
    if region is None:
        raise ValueError(f'unknown patch {patch!r}: not in config.RAW_REGION')
    base = Path(root) / patch / 'raw' / region
    timeline_path, detail_path = base / 'timeline' / f'{key}.json', base / 'detail' / f'{key}.json'
    if not (_is_file(timeline_path, 'timeline', patch) and _is_file(detail_path, 'detail', patch)):
        raise MatchRejected('missing_file')

    def fail(reason: str) -> MatchFormatError:
        return MatchFormatError(key, patch, reason)

    detail = _read_json(detail_path, 'detail', key, patch)
    team, champ, runes_by_pid = _participants(detail, fail)                          # the small file first: cheap rejections first

    timeline = _read_json(timeline_path, 'timeline', key, patch)
    frames, per_frame = _frames(timeline, fail)
    events = [e for evs in per_frame for e in evs]
    events.sort(key=lambda e: e['timestamp'])                                  # list.sort is stable: same ms keeps the file order
    if not any(e.get('type') == 'GAME_END' for e in events):
        raise MatchRejected('no_game_end')
    if events[-1].get('type') != 'GAME_END':
        raise fail('an event comes after GAME_END')

    # 식 (RD-2)의 시계: ts_k와 T를 정하고 아래 조건을 차례로 확인한다. 어긋난 파일은 보정하지 않고 형식 오류로 멈춘다.
    frame_times = [fr['timestamp'] for fr in frames]
    T_ms = int(events[-1]['timestamp'])                                        # the time of GAME_END, the last (= latest) event
    if frame_times[0] != 0:
        raise fail(f'first frame is at {frame_times[0]} ms, not at 0')
    for k in range(1, len(frame_times)):
        if frame_times[k] <= frame_times[k - 1]:
            raise fail(f'frame timestamps do not increase at frame {k}')
    if not any(e.get('type') == 'GAME_END' for e in per_frame[-1]):
        raise fail('GAME_END is not in the last frame')
    if len(frames) > 1 and frame_times[-2] >= T_ms:
        raise fail(f'last regular frame at {frame_times[-2]} ms is not before the game end at {T_ms} ms')

    # The two checks below come after every older one on purpose: a file that was rejected or raised an error before still does so
    # in the same way, so only a match that was accepted can be lost to them. (Pre-S2 guards; final review of S1, Minor 5.)
    if not _same_match(timeline, detail):
        raise MatchRejected('timeline_detail_mismatch')

    # 식 (RD-3): 정규 프레임마다 참가자 10명의 값을 채우고 완전하지 않은 프레임이 있으면 경기를 거부한다. 경기 끝 프레임은 이 요구에서 뺀다.
    regular, last = frames[:-1], frames[-1]
    frame_ts = np.array(frame_times[:-1], dtype=np.int64)
    frame_num = np.full((len(regular), 10, len(C.FRAME_FIELDS)), _NAN)
    for k, fr in enumerate(regular):
        if not _fill_frame(frame_num[k], fr.get('participantFrames')):
            raise MatchRejected('incomplete_frame')                                # no silent all-NaN row for a player without a value
    end_frame = np.full((10, len(C.FRAME_FIELDS)), _NAN)
    _fill_frame(end_frame, last.get('participantFrames'))                          # the game-end frame may miss players (positions only)

    return MatchRecord(key=key, patch=patch, T_ms=T_ms, team=team, champ=champ, runes_by_pid=runes_by_pid,
                       frame_ts=frame_ts, frame_num=frame_num, end_xy=np.ascontiguousarray(end_frame[:, [_COL_X, _COL_Y]]),
                       events=tuple(events))
