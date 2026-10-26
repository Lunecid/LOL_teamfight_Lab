"""Run ledger of the run kernel: stages with input and output hashes, idempotent finalisation, tamper refusal (S4b-prep K5a, decision D11, A5-F7).

Design reference: review EPT-T010 A5-F7 ("S4a stage finalisation has crash windows that force a brand-new run"): the S4a stage wrote the stage JSON,
then ``RUN.json``, then ``ledger.finish``; a crash between those writes made the run permanently unresumable, although every file was intact and
verifiable by hash, and none of the refusal paths was tested. Decision D11: "Finalisation is idempotent: when a stage JSON exists on resume, its
input and output hashes are re-verified and the ledger is completed; any mismatch stays fail-closed", and "the supervisor and ledger reuse the
components proven in T009".

What the earlier stacks have, and what this module keeps (those modules are frozen and unchanged):

* ``workflows.run_s3.Ledger`` (S3/S4a): ``begin(stage, inputs)`` then ``finish(stage, outputs)`` with the hashes of the inputs; a stage that was
  begun on other inputs is not begun again ("a run folder never mixes two sets of inputs"). Kept, with one stage at a time and the hashes taken here
  (an agreement test runs the S3 class next to this one).
* ``comparison.runtime._receipt`` / ``_write_receipt`` (T009): the stage receipt lists the artifact hashes by path relative to the run folder, and
  every hash is checked again before the stage counts as done; the path rule is ``comparison.protocol.inside``. Kept: the stage JSON is that
  receipt, with the input hashes added, the run header bound in and the previous stage file chained in.
* ``comparison.infrastructure.CampaignLedger`` (T009): the ledger belongs to one owner; a file that is not the bytes this object read or wrote is
  refused (``_verify``). Kept as ``_unchanged``. T009 reopens read-only and never resumes; the kernel does resume.
* All JSON is read and written through ``run.io`` (strict, read once; atomic and fsynced; canonical hash). This module imports no ``json``.

Files under the run folder R
----------------------------
    HEADER.json        written once at creation: the K2 identity, the decision-record fingerprint (SHA-256 of the bytes, SHA-256 of every
                       top-level member of the record outside ``decisions``, SHA-256 of every decision block), the SHA-256 of the profile, and a
                       free mapping of runner constants (K6: batch size, device, effective M fields)
    LEDGER.json        the stages in the order in which they began, each ``begun`` or ``finished``; it names the header by its hash; and, only
                       once there is one, ``notes`` (K5b: the stop reasons and E6 records of the supervisor, see "Notes" below)
    stages/<s>.json    the stage file of a stage s: its input and output hashes, bound to the header and chained to the previous stage file

Symbols
-------
    R, H     the run folder and the header file R/HEADER.json; h = SHA-256(bytes of H), the run's binding (the writer's returned hash)
    L, l     the ledger file R/LEDGER.json and l = SHA-256(bytes of L) as this object last read or wrote it
    s        a stage name, ``[a-z][a-z0-9_]{0,47}`` and not a Windows device name (it is a file name)
    E_s      the entry of s in L: ``state`` begun or finished; ``inputs`` I_s and ``outputs`` O_s (path -> SHA-256); once finished also
             ``stage_sha256`` f_s and ``previous_sha256`` q_s
    F_s      the stage file R/stages/s.json = {schema, stage: s, header_sha256: h, previous_sha256: q_s, inputs: I_s, outputs: O_s}
    q_s      the chain: q_s = h for the first finished stage, else f_t of the finished stage t just before s (f_t = SHA-256(bytes of F_t))
    sha(p)   ``run.io.sha_file(p)``, the SHA-256 of the file p; a path is relative to R (forward slashes) or, for an input, absolute
    X, Y     the paths that begin(s, inputs=X) and finish(s, Y) are given
    n, m     counts: n of the m files of a stage differ from their record
    D, D_k   the decision record (one JSON object) and its member ``decisions``, an object of named blocks k
    D_t      a top-level member t of D other than ``decisions`` (``operating_point``, ``data``, a section such as ``G2_time_base``)
    C(v)     the canonical hash of a JSON value v: SHA-256 of json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
             allow_nan=False) encoded as UTF-8 (``run.io.canonical_sha256``)
    T, B     the header's ``spec_top`` = { t -> C(D_t) : t a top-level key of D other than ``decisions`` } and ``spec_blocks`` = { k -> C(D_k) },
             both taken at the start
    N_i      note i of L: {stage: s or null, kind: ``stop`` or ``e6``, record: a JSON object}; at most ``MAX_NOTES`` of them

Equations
---------
    begin(s, X)   :  I_s := { x -> sha(x) : x in X };  require sha(x) = O_t(x) whenever x is an output of a finished stage t (a stage never starts
                     from a file that differs from the output that was recorded); E_s := (begun, I_s);  one write of L.
                     Begun again on equal I_s: no write; on another I_s: refused
    finish(s, Y)  :  require sha(x) = I_s(x) for every x (the inputs did not change while s ran);  O_s := { y -> sha(y) : y in Y }  (Y inside R);
                     write F_s;  E_s := (finished, I_s, O_s, f_s = SHA-256(bytes of F_s), q_s);  write L.            two writes, in this order
    resume()      :  for every finished E_s:  SHA-256(F_s) = f_s,  F_s = {s, h, q_s, I_s, O_s},  sha(p) = hash for every (p, hash) in I_s and O_s,
                     and the chain q_s holds from the header through every finished stage
                     for the begun E_s, if F_s exists (the stop between the two writes of finish):  F_s = {s, h, q_s, I_s, O_s'} with every hash true,
                     then E_s := (finished, I_s, O_s', SHA-256(F_s), q_s) and write L          (idempotent: the second resume writes nothing)
                     a stage folder item that is not s.json for an entry s (a temporary file aside)  =>  unknown stage
                     anything else  =>  Tampered;  resume returns the finished stages in order
    opened ledger :  begin and finish are refused until resume() has run to its end in that object (a created ledger counts as resumed)
    crash points  :  begin (L) and finish (F_s, then L): a stop before or after any one write leaves a state that the next start either completes
                     (stop after F_s) or runs again (stop before F_s), and the finished run is the run that was never stopped.
                     Creation (H, then L) is the exception: a stop between its two writes leaves a header without a ledger, refused as
                     "creation incomplete"; nothing was computed yet, and the answer is a new run folder
    spec growth   :  the decision record D' (read at a resume) is accepted iff  D' is a JSON object with an object ``decisions`` (or none),
                     for every t in T:  t is in D' and C(D'_t) = T(t),  and for every k in B:  k is in D'.decisions and C(D'.decisions[k]) = B(k).
                     New top-level members (K5b ruling R1: the real record grew by whole sections such as ``G2_time_base``), new blocks, other
                     white space and another member order are accepted; an edited or removed top-level member, an edited or removed block are
                     ``GateRefused('spec')``
    notes         :  note(s, kind, record) appends N = (s, kind, record) to L (one write, owner bytes checked, only on a resumed ledger); notes
                     are not part of the chain and carry no hash of files. A ledger without notes has no ``notes`` member (the K5a bytes).
    check         :  ``Ledger.check`` is an optional function without arguments that is passed to ``run.io.sha_file(..., check=)`` for every
                     file the ledger hashes (the owner check, begin, finish, resume); whatever it raises stops the operation before any write
                     (a wall-clock or memory budget, a stop request)

해석
----
원장은 "어느 단계까지 어떤 입력으로 끝났는가"를 해시로 증명하는 기록이다. 단계 하나는 ``begin`` 한 번(원장에 ``begun``과 입력 해시), ``finish`` 한 번(단계
파일을 쓰고, 그다음 원장에 ``finished``)으로 이루어진다. 쓰는 순서가 중요하다. 단계 파일을 먼저 쓰므로, 단계 파일이 있으면 그 단계의 출력은 이미 해시된 뒤
이다. 단계 파일을 쓴 직후 멈추면 원장은 ``begun``인데 단계 파일은 있는 상태가 되고, 이어 하기(``resume``)는 단계 파일의 입력·출력 해시를 실제 파일과 다시 대조해
하나도 틀리지 않을 때에만 원장을 마저 완성한다(A5-F7). 틀리면 거부한다. S4a는 이 경우를 거부했고 새 실행을 강제했다. 단계 파일을 쓰기 전에 멈추면 ``begun``만 남고,
같은 입력으로 ``begin``을 다시 하면 된다(출력이 덮어 쓰일 뿐이다). 쓰기 하나하나가 원자적이므로 멈춘 자리는 "썼다" 아니면 "안 썼다"뿐이다. 예외는 만들 때(머리글, 그다음 원장)다.
두 쓰기 사이에서 멈추면 머리글만 남고, 그 폴더는 "creation incomplete"로 거부한다. 아직 계산한 것이 없으므로 새 실행 폴더를 쓰면 된다.

무엇이 거부되는가. 출력이나 입력 파일의 바이트가 바뀌거나 지워지면 해시가 맞지 않는다. 단계 파일이 바뀌면 원장에 적힌 단계 파일 해시나 원장의 입력·출력 기록과 달라진다.
원장 항목이 바뀌면 단계 파일과 달라진다. 단계 폴더에 원장이 모르는 파일이 있으면 "모르는 단계"이다. 머리글(``HEADER.json``)이 바뀌면 원장과 단계 파일이 적어 둔 머리글 해시와
달라진다. 끝난 단계의 기록(원장 항목과 단계 파일) 하나를 통째로 지우면 다음 단계의 ``previous_sha256``이 이어지지 않아 거부된다(마지막 단계는 뒤따르는 것이 없으므로 지워도
깨지는 것이 없고, 그 단계를 다시 하면 된다). 단계를 시작할 때 입력이 앞선 단계가 기록한 출력과 같은 파일이면 그 해시와 같아야 한다. 이 모두는 ``Tampered``이다.

무엇이 거부되지 않는가. 바깥에 기준(앵커, external anchor)이 없으므로 변조자가 출력 파일을 바꾸고, 그 새 해시를 단계 파일, 원장 항목, ``stage_sha256``(그리고 뒤따르는 단계의 사슬)에 모두 맞게
고쳐 쓰면 이 원장은 알아내지 못한다. 시작한 단계(``begun``)에 대해서는 단계 파일 하나만 고치면 되고(``begun`` 항목에는 출력이 없다), 출력의 일부만 적은 단계 파일도 받아들인다. 그래서
K6는 끝난 단계가 기대한 출력을 기록했는지 ``Ledger.stage``로 확인해야 한다. 일관된 위조를 막는 것은 이 원장이 아니라 실행 정체성과 환경 잠금, 그리고 기록을 따로 보관하는 일이다.

원장은 소유자가 한 명이다. 이 객체가 마지막으로 읽거나 쓴 바이트와 디스크의 파일이 다르면 쓰기 전에 거부한다(T009 ``CampaignLedger``와 같음). 다시 연 원장은 ``resume``이 끝까지
돌기 전에는 ``begin``과 ``finish``를 거부한다(검증하지 않은 상태에 쓰지 않는다). 한 번에 한 단계만 ``begun``일 수 있고, 단계 이름은 파일 이름이므로 짧은 소문자 낱말로 제한한다.
메시지에는 단계 이름과 개수만 싣는다. 경로, 파일 내용, 해시는 싣지 않는다(경로에 경기 키가 들어 있을 수 있다). 해시를 읽다가 나는 운영체제 오류(권한, 공유 위반)는 변조의 증거가
아니므로 ``Tampered``가 아니라 그대로 올라간다. 아무것도 쓰지 않았으므로 다시 시도해도 안전하다.

결정 기록(spec)은 JSON 객체 하나이고 ``decisions`` 아래에 결정 블록이 이름을 달고 들어 있다. 새 결정은 그 안에 새 멤버로 끼어들므로 바이트 접두사 규칙으로는 받아들일 수 없다. 그래서
머리글에 시작 때 기록의 최상위 멤버(``decisions``를 뺀 것)마다의 정규 해시와 블록마다의 정규 해시를 남긴다. 이어 하기에서는 저장된 최상위 멤버와 블록이 모두 그대로 있을 때만 받아들이며
새 최상위 멤버(예: ``G2_time_base`` 같은 새 구획)와 새 블록은 허용한다(K5b 판정 R1: 실제 기록은 최상위 구획으로도 자랐다). 정규 JSON은 키를 정렬하고 공백을 없앤 UTF-8 문자열이므로
공백이나 키 순서만 달라진 기록은 같다. 고쳐졌거나 지워진 최상위 멤버와 블록은 ``GateRefused('spec')``이다.

메모(notes, K5b). 감독자는 단계가 멈춘 이유(예산, 실패, 거부)와 E6 확인 기록을 남길 자리가 필요하다. 원장 파일에 ``notes`` 목록을 두고 ``note(단계, 종류, 기록)``로 덧붙인다. 메모는
사슬에 들어가지 않고 파일 해시도 갖지 않는다. 사슬이 증명하는 것은 단계의 입력·출력이지 감독자의 사정이 아니다. 메모가 없는 원장에는 ``notes`` 멤버가 없으므로 K5a와 바이트가 같다.
``check``는 원장이 파일을 해시하는 동안 불리는 멈춤 함수이다. 예산을 넘기면 여기서 예외를 올려 큰 파일의 해시를 중간에 끊는다. 쓰기 전에 끊기므로 반쯤 쓴 기록이 남지 않는다. 만들 때는 기록 바이트를 한 번 읽고, ``guards.spec_check(경로, expected_sha256=그 바이트의 해시)``로 같은 바이트를 확인한 뒤, 같은 바이트로
``spec_fingerprint``를 부른다(K5b의 몫). 이어 할 때는 ``Ledger.check_spec_file(경로)``가 한 번 읽은 바이트에 ``spec_check``(상수 대조)와 ``check_spec``(블록 규칙)을 모두 적용한다.

내구성의 한계(power loss). 쓰기는 파일을 fsync한 뒤 이름을 바꾸지만 폴더는 fsync하지 않는다(Windows에서는 폴더 fsync를 쓸 수 없다). 프로세스가 멈추는 경우는 위에서 다룬 대로이지만, 전원이
나가면 이름 바꾸기 순서가 보장되지 않아 ``finished`` 항목은 남았는데 그 단계 파일은 없는 상태가 될 수 있다. 그 상태는 ``Tampered``로 거부된다. 안전하게 멈추지만 끝난 학습을 잃을 수 있다.
이것은 K1 쓰기와 이 원장이 함께 가진 잔여 위험이다.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType

from c_ept.run import guards
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io

SCHEMA_HEADER = 'ept-run-header-v1'
SCHEMA_LEDGER = 'ept-run-ledger-v1'
SCHEMA_STAGE = 'ept-run-stage-v1'
HEADER_FILE = 'HEADER.json'
LEDGER_FILE = 'LEDGER.json'
STAGE_FOLDER = 'stages'
STATES = ('begun', 'finished')
MAX_SPEC_BYTES = 16 << 20                                           # the largest decision record that ``check_spec_file`` reads (the bound of ``run.io.read_json``)
STAGE_NAME = re.compile(r'[a-z][a-z0-9_]{0,47}')
# A file name that Windows reads as a device, whatever the extension ("con.json" is the console). The stage name is a file name.
_DEVICE_NAMES = frozenset({'con', 'prn', 'aux', 'nul'} | {f'com{i}' for i in range(1, 10)} | {f'lpt{i}' for i in range(1, 10)})
_HASH = re.compile(r'[0-9a-f]{64}')
_HEADER_FIELDS = frozenset({'schema', 'identity', 'spec_sha256', 'spec_top', 'spec_blocks', 'profile_sha256', 'runner'})
_LEDGER_FIELDS = frozenset({'schema', 'header_sha256', 'stages'})
NOTE_KINDS = ('stop', 'e6')                                         # K5b: a stage's stop reason, an E6 check record
MAX_NOTES = 10_000                                                  # the notes of one run (a record per stage and a few per stop: far below this)
_NOTE_FIELDS = frozenset({'stage', 'kind', 'record'})
_STAGE_FIELDS = frozenset({'schema', 'stage', 'header_sha256', 'previous_sha256', 'inputs', 'outputs'})
_ENTRY_FIELDS = {'begun': frozenset({'stage', 'state', 'inputs'}),
                 'finished': frozenset({'stage', 'state', 'inputs', 'outputs', 'stage_sha256', 'previous_sha256'})}
_OWN_FILES = (HEADER_FILE.casefold(), LEDGER_FILE.casefold())       # keys that name the ledger's own files (compared case-folded: Windows is case-blind)


class LedgerRefused(run_io.IORefused):
    """A ledger operation that is not acceptable: a ledger that exists or is missing, a stage that is begun or finished out of order, a path or a
    stage name that is not acceptable, an input or output that is missing, an opened ledger that was not resumed. An ``IORefused`` (so a
    ``ValueError``): K5b maps it to exit code 2.

    The message names the stage and counts; it never holds a path, a file content or a hash.
    """


class Tampered(LedgerRefused):
    """A record does not match the files or another record: a hash that differs, a file that is missing, a record that is malformed or belongs
    to another run, an unknown stage file, a broken chain, a ledger that changed behind its owner. Fail-closed: the run is not resumed. Nothing was written."""


# ---- shapes ------------------------------------------------------------------------------------------------------------------------------
def _is_hash(value) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _valid_stage_name(name) -> bool:
    return isinstance(name, str) and STAGE_NAME.fullmatch(name) is not None and name not in _DEVICE_NAMES


valid_stage_name = _valid_stage_name                                # public: the supervisor checks its plan with the ledger's own rule before it reserves anything


def _valid_relative(text) -> bool:
    """A path relative to the run folder, with forward slashes, that cannot leave it.

    Ref: ``comparison.protocol.inside`` (T009 artifact path: not absolute, no ``..``, no ``:``). Stricter: no backslash, every segment non-empty
    (so no ``.``, no ``//``, no trailing ``/``) and not ending in ``.`` or a space (Windows strips them, so ``.. `` would be ``..``).
    """
    if not isinstance(text, str) or '\\' in text or ':' in text or any(ord(c) < 32 or ord(c) == 127 for c in text):
        return False
    return all(part != '' and part == part.rstrip(' .') for part in text.split('/'))      # the empty text has one empty part: refused


def _valid_absolute(text) -> bool:
    """An absolute path in its normal form (forward slashes), without ``..`` or control characters. Only an input may be outside the run folder."""
    if not isinstance(text, str) or any(ord(c) < 32 or ord(c) == 127 for c in text):
        return False
    path = Path(text)                                                       # the empty text is the folder "." : not absolute, refused below
    return path.is_absolute() and path.as_posix() == text and '..' not in path.parts


def _valid_digests(value, *, external: bool) -> bool:
    return isinstance(value, dict) and all((_valid_relative(k) or (external and _valid_absolute(k))) and _is_hash(v) for k, v in value.items())


def _is_own_file(key: str) -> bool:
    """True when the relative key names ``HEADER.json``, ``LEDGER.json`` or the stage folder (or anything in it): files that the ledger itself rewrites."""
    low = key.casefold()
    return low in _OWN_FILES or low == STAGE_FOLDER or low.startswith(STAGE_FOLDER + '/')


def _freeze(value):
    """A read-only copy at every depth: a mapping becomes a ``MappingProxyType``, a list or tuple a tuple."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    """The plain JSON form of a frozen value: a mapping becomes a dict, a tuple a list (a new object at every depth)."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


# ---- the header ---------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Header:
    """The run header, written once at creation (``HEADER.json``).

    ``identity``: the K2 ``Identity`` (commit and imported-file hashes). The decision-record fingerprint (``spec_fingerprint`` gives the three
    values): ``spec_sha256``, the SHA-256 of the record bytes (what ``guards.spec_check`` returns); ``spec_top``, the canonical hash of every
    top-level member of the record other than ``decisions``, by name; ``spec_blocks``, the canonical hash of every member of ``decisions`` by name.
    A resume accepts a record that holds every top-level member and every block unchanged (``Ledger.check_spec``); it may hold more of both
    (K5b ruling R1). ``profile_sha256``: the lock of the
    profile file. ``runner``: a free mapping of runner constants that K6 fills (batch size, device, the effective M fields); keys are text and the
    values must be strict JSON (the writer refuses anything else when the ledger is created).

    Validated on construction (``IORefused``, naming the field and never its value). ``runner``, ``spec_top`` and ``spec_blocks`` are read-only at every depth: a
    nested mapping is a ``MappingProxyType``, a list is a tuple, and a later change of the caller's objects changes nothing. ``to_record`` gives
    plain dicts and lists again.
    """

    identity: run_identity.Identity
    spec_sha256: str
    spec_top: Mapping = field(hash=False)
    spec_blocks: Mapping = field(hash=False)
    profile_sha256: str
    runner: Mapping = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not isinstance(self.identity, run_identity.Identity):
            raise run_io.IORefused('header identity must be an Identity')
        if not (_is_hash(self.spec_sha256) and _is_hash(self.profile_sha256)):
            raise run_io.IORefused('header spec_sha256 and profile_sha256 must be 64 lowercase hex digits')
        if not (isinstance(self.spec_top, Mapping) and all(isinstance(k, str) and k != 'decisions' and _is_hash(v) for k, v in self.spec_top.items())):
            raise run_io.IORefused('header spec_top must map top-level member names (not decisions) to 64 lowercase hex digits')
        if not (isinstance(self.spec_blocks, Mapping) and all(isinstance(k, str) and _is_hash(v) for k, v in self.spec_blocks.items())):
            raise run_io.IORefused('header spec_blocks must map block names to 64 lowercase hex digits')
        if not isinstance(self.runner, Mapping) or not all(isinstance(key, str) for key in self.runner):
            raise run_io.IORefused('header runner must be a mapping with text keys')
        refused = False
        try:
            runner = _freeze(dict(self.runner))                              # a new read-only copy at every depth, whatever the caller holds
        except RecursionError:                                              # a cycle or an absurdly deep mapping
            refused = True
        if refused:
            raise run_io.IORefused('header runner is nested too deeply')
        object.__setattr__(self, 'runner', runner)
        object.__setattr__(self, 'spec_top', MappingProxyType(dict(sorted(self.spec_top.items()))))
        object.__setattr__(self, 'spec_blocks', MappingProxyType(dict(sorted(self.spec_blocks.items()))))

    def to_record(self) -> dict:
        """The plain-JSON form that ``run.io.write_json_atomic`` writes (a new object at every depth)."""
        return {'schema': SCHEMA_HEADER, 'identity': self.identity.to_record(), 'spec_sha256': self.spec_sha256,
                'spec_top': dict(self.spec_top), 'spec_blocks': dict(self.spec_blocks), 'profile_sha256': self.profile_sha256,
                'runner': _thaw(self.runner)}

    @classmethod
    def from_record(cls, record) -> Header:
        """The header of a record read by ``run.io``; ``IORefused`` for a missing or extra field, another schema or a bad value."""
        if not isinstance(record, Mapping) or set(record) != _HEADER_FIELDS:
            raise run_io.IORefused('header record fields mismatch')
        if record['schema'] != SCHEMA_HEADER:
            raise run_io.IORefused('header record schema mismatch')
        return cls(identity=run_identity.Identity.from_record(record['identity']), spec_sha256=record['spec_sha256'],
                   spec_top=record['spec_top'], spec_blocks=record['spec_blocks'], profile_sha256=record['profile_sha256'],
                   runner=record['runner'])


# ---- the decision record only grows, block by block (K5a review I1, ruling (a)) ----------------------------------------------------------------
def _spec_parts(record) -> tuple[dict, dict]:
    """``({top-level name: C(member)}, {block name: C(block)})`` of the bytes of a decision record, the top-level members being those other than
    ``decisions``; ``GateRefused('spec', <fixed text>)`` if the bytes are not a strict JSON object (``run.io.parse_json_object``) or ``decisions`` is
    not an object. A record without ``decisions`` has no blocks."""
    refusal = None
    try:
        doc = run_io.parse_json_object(record)
        decisions = doc.get('decisions', {})
        if not isinstance(decisions, dict):
            refusal = 'malformed: decisions in the decision record is not an object'
        else:
            top = {key: run_io.canonical_sha256(value) for key, value in doc.items() if key != 'decisions'}
            blocks = {name: run_io.canonical_sha256(block) for name, block in decisions.items()}
    except run_io.IORefused:
        refusal = 'unreadable: the decision record is not a JSON object'
    if refusal is not None:
        raise guards.GateRefused('spec', refusal)                           # outside the except block: no exception is chained, so nothing is printed
    return top, blocks


def spec_fingerprint(record: bytes) -> dict:
    """The header fields of a decision record: ``{'spec_sha256', 'spec_top', 'spec_blocks'}`` (``Header(..., **spec_fingerprint(bytes))``).

    ``spec_sha256`` is the SHA-256 of the bytes, ``spec_top`` the canonical hash of each top-level member other than ``decisions`` by name and
    ``spec_blocks`` the canonical hash of each member of ``decisions`` by name (canonical JSON: ``json.dumps(value, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False, allow_nan=False)`` as UTF-8). Take the bytes from one read and bind them to the gate with
    ``guards.spec_check(path, expected_sha256=sha256(bytes))``. ``TypeError`` if ``record`` is not bytes; ``GateRefused('spec')`` if it is not a JSON
    object whose ``decisions`` (if any) is an object.
    """
    if not isinstance(record, (bytes, bytearray)):
        raise TypeError('the decision record must be bytes')
    top, blocks = _spec_parts(record)
    return {'spec_sha256': hashlib.sha256(record).hexdigest(), 'spec_top': top, 'spec_blocks': blocks}


def check_spec_growth(current: bytes, *, top: Mapping, blocks: Mapping) -> None:
    """Accept the current decision record only if it is the stored record with top-level members and decision blocks added, changed nowhere.

    ``top`` and ``blocks`` are the header's ``spec_top`` and ``spec_blocks``. Accepted: the same record in other white space or member order, a record
    with more top-level members (K5b ruling R1: the real record also grew by whole sections such as ``G2_time_base``) and a record with more blocks
    under ``decisions`` (a new decision is a new member there, so a byte-prefix rule could not accept it).
    ``guards.GateRefused('spec', <fixed text>)`` (K5b: exit code 2) for bytes that are not a JSON object or whose ``decisions`` is not an object
    (``unreadable``, ``malformed``), a stored top-level member that is missing (``removed outside decisions``) or differs (``changed outside
    decisions``), a stored block that is missing (``removed``) and a stored block that differs (``changed``). The text holds neither the record nor a
    hash nor a member or block name. This is the structural part of the resume check; the constants part is ``guards.spec_check``, which K5b calls on
    the same bytes. ``TypeError`` for ``current`` that is not bytes, ``ValueError`` for a malformed ``top`` or ``blocks``.
    """
    if not isinstance(current, (bytes, bytearray)):
        raise TypeError('the decision record must be bytes')
    if not (isinstance(top, Mapping) and all(isinstance(k, str) and k != 'decisions' and _is_hash(v) for k, v in top.items())):
        raise ValueError('top must map top-level member names (not decisions) to 64 lowercase hex digits')
    if not (isinstance(blocks, Mapping) and all(isinstance(k, str) and _is_hash(v) for k, v in blocks.items())):
        raise ValueError('blocks must map block names to 64 lowercase hex digits')
    now_top, now = _spec_parts(current)
    refusal = None
    if any(name not in now_top for name in top):
        refusal = 'removed outside decisions: a top-level member of the record the run began with is missing'
    elif any(now_top[name] != digest for name, digest in top.items()):
        refusal = 'changed outside decisions: a top-level member of the record the run began with differs'
    elif any(name not in now for name in blocks):
        refusal = 'removed: a decision block of the record the run began with is missing'
    elif any(now[name] != digest for name, digest in blocks.items()):
        refusal = 'changed: a decision block of the record the run began with differs'
    if refusal is not None:
        raise guards.GateRefused('spec', refusal)


# ---- the ledger ---------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StageRecord:
    """What the ledger recorded for one stage (``Ledger.stage``): its ``state`` (begun or finished), the ``inputs`` and, once finished, the
    ``outputs`` (read-only mappings of path -> SHA-256; a begun stage has no outputs). K6 uses it to check that a finished stage recorded the
    outputs that it expects."""

    stage: str
    state: str
    inputs: Mapping = field(hash=False)
    outputs: Mapping = field(hash=False)


@dataclass(frozen=True)
class Note:
    """One note of the ledger (``Ledger.notes``): ``stage`` (a stage name, or None for the run as a whole), ``kind`` (one of ``NOTE_KINDS``: ``stop``, the
    reason a stage did not finish; ``e6``, the record of an E6 check) and ``record``, a JSON object (read-only at every depth, a list is a tuple)."""

    stage: str | None
    kind: str
    record: Mapping = field(hash=False)

    def __post_init__(self):
        object.__setattr__(self, 'record', _freeze(dict(self.record)))


def _read_record(path: Path, what: str):
    """``(object, sha256)`` of a record file, read once through ``run.io``; ``Tampered`` if it is missing, unreadable or malformed (no cause is chained)."""
    refused = False
    try:
        return run_io.read_json_with_sha(path)
    except run_io.IORefused:
        refused = True
    if refused:
        raise Tampered(f'{what} record is missing, unreadable or malformed')


def _require_stage_name(stage) -> None:
    if not _valid_stage_name(stage):
        raise LedgerRefused('a stage name must be a lowercase word of at most 48 letters, digits and underscores that starts with a letter '
                            'and is not a Windows device name')


# ---- the paths of a stage: the same rule for ``begin``, ``finish`` and the supervisor's check of its whole plan -------------------------------
def _key_of(run_dir: Path, item, *, external: bool):
    """The record key of one path (relative to the run folder, forward slashes; an input outside the run folder keeps its absolute form), or None.

    A path object is converted with ``as_posix()`` first (``Path('a/b')`` is ``a\\b`` as text on Windows); a text is taken as it is, so a text with
    a backslash is refused. Nothing is read from the disk."""
    if isinstance(item, PurePath):
        text = item.as_posix()
    elif isinstance(item, str):
        text = item
    elif isinstance(item, os.PathLike) and isinstance(os.fspath(item), str):
        text = PurePath(os.fspath(item)).as_posix()
    else:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            inside = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(run_dir))).as_posix()
        except ValueError:
            inside = None
        text = inside if inside is not None else path.as_posix()          # an absolute path inside the run folder is keyed relative
    if _valid_relative(text) or (external and _valid_absolute(text)):     # outside it, only an input may be named (by its absolute path)
        return text
    return None


key_of = _key_of                                                    # public (K5b review m4): the supervisor refuses the names it writes itself by the same key


def _keys_of(run_dir: Path, paths, *, what: str, external: bool) -> list:
    if isinstance(paths, (str, bytes, Mapping)) or not isinstance(paths, Iterable):
        raise LedgerRefused(f'{what} must be a list of paths')
    keys = []
    for item in paths:
        key = _key_of(run_dir, item, external=external)
        if key is None:
            raise LedgerRefused(f'{what}: a path is not acceptable (a relative path with forward slashes' +
                                (' inside the run folder, or an absolute path)' if external else ' inside the run folder)'))
        if _is_own_file(key):
            raise LedgerRefused(f'{what}: a path names a file of the ledger itself ({HEADER_FILE}, {LEDGER_FILE} or the {STAGE_FOLDER} folder)')
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise LedgerRefused(f'{what}: a path is given twice')
    return sorted(keys)


def check_paths(run_dir, *, inputs=(), outputs=()) -> None:
    """Refuse (``LedgerRefused``) the inputs or outputs of a stage that ``Ledger.begin`` or ``Ledger.finish`` would refuse for their *form*: a path that
    is not acceptable, one of the ledger's own files, a path given twice, a list that is not a list. Nothing is read, nothing is written, no file has
    to exist (``begin`` and ``finish`` still refuse a file that is missing). The supervisor calls it for its whole plan before it reserves anything."""
    run_dir = Path(run_dir)
    _keys_of(run_dir, inputs, what='inputs', external=True)
    _keys_of(run_dir, outputs, what='outputs', external=False)


class Ledger:
    """The ledger of one run folder: ``Ledger.create(run_dir, header)`` makes it, ``Ledger(run_dir)`` opens it.

    ``begin(stage, inputs=...)``, ``finish(stage, outputs)`` and ``resume()`` (the list of the finished stages, after the re-verification and the
    completion described in the module docstring). One owner: before every operation the files must be the bytes that this object last read or wrote.
    Opening validates every record (``Tampered`` for a malformed one or a broken chain); ``resume`` verifies the files; an opened ledger refuses
    ``begin`` and ``finish`` until ``resume`` has succeeded in that object.

    ``create`` is a check and then a write, so two processes that create the same folder at once are not told apart here; the supervisor's run lock
    (K5b) is what makes the creator single. A stop between the two creation writes leaves a header without a ledger: ``create`` and the open both
    refuse that folder ("creation incomplete"; start a new run folder). Every later stop is recoverable (see the module docstring).
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self._header_path = self.run_dir / HEADER_FILE
        self._ledger_path = self.run_dir / LEDGER_FILE
        self._resumed = False
        self._check = None                                                  # K5b: the stop or budget function that the hashing calls (see ``check``)
        has_header, has_ledger = os.path.lexists(self._header_path), os.path.lexists(self._ledger_path)
        if not has_header and not has_ledger:
            raise LedgerRefused('no ledger in this run folder')
        if not has_header:
            raise Tampered('the header file is missing')
        if not has_ledger:
            if os.path.lexists(self.run_dir / STAGE_FOLDER):
                raise Tampered('the ledger file is missing (the run has a stage folder)')
            raise LedgerRefused('creation incomplete: the run folder has a header but no ledger (a stop during creation); start a new run folder')
        record, header_sha = _read_record(self._header_path, 'header')
        header = self._header_of(record)
        doc, ledger_sha = _read_record(self._ledger_path, 'ledger')
        self._entries = self._entries_of(doc, header_sha)
        self._notes = self._notes_of(doc)
        self._header, self._header_sha, self._ledger_sha = header, header_sha, ledger_sha

    # -- construction and validation ----
    @classmethod
    def create(cls, run_dir, header: Header) -> Ledger:
        """Write the header and an empty ledger into ``run_dir`` (made if missing) and return the opened ledger, which counts as resumed. Two writes,
        in this order.

        ``LedgerRefused`` if ``header`` is not a ``Header`` or if the folder already has a header or a ledger (never overwritten, not even by an
        equal header). ``IORefused`` from the writer, before any file or folder exists, if the runner mapping is not strict JSON.
        """
        if not isinstance(header, Header):
            raise LedgerRefused('create needs a Header')
        run_dir = Path(run_dir)
        if os.path.lexists(run_dir / HEADER_FILE) or os.path.lexists(run_dir / LEDGER_FILE):
            raise LedgerRefused('a ledger already exists in this run folder; it is never overwritten')
        header_sha = run_io.write_json_atomic(run_dir / HEADER_FILE, header.to_record())
        run_io.write_json_atomic(run_dir / LEDGER_FILE, {'schema': SCHEMA_LEDGER, 'header_sha256': header_sha, 'stages': []})
        ledger = cls(run_dir)
        ledger._resumed = True                                              # nothing was there to verify: a new ledger is its own verified state
        return ledger

    @staticmethod
    def _header_of(record) -> Header:
        failed = False
        try:
            header = Header.from_record(record)
        except run_io.IORefused:
            failed = True
        if failed:
            raise Tampered('the header record is malformed')
        return header

    @staticmethod
    def _entries_of(doc, header_sha: str) -> list:
        """The validated entries of a ledger document; ``Tampered`` for any malformed record, a broken chain or a ledger that names another header."""
        if (set(doc) - {'notes'} != _LEDGER_FIELDS or doc['schema'] != SCHEMA_LEDGER or not isinstance(doc['stages'], list)
                or not _is_hash(doc['header_sha256'])):
            raise Tampered('the ledger record is malformed')
        if doc['header_sha256'] != header_sha:
            raise Tampered('the header file does not match the header named by the ledger')
        entries, seen = [], set()
        for index, entry in enumerate(doc['stages']):
            state = entry.get('state') if isinstance(entry, dict) else None
            ok = state in STATES and set(entry) == _ENTRY_FIELDS[state] and _valid_stage_name(entry['stage']) and entry['stage'] not in seen
            ok = ok and _valid_digests(entry['inputs'], external=True)
            if ok and state == 'finished':
                ok = _valid_digests(entry['outputs'], external=False) and _is_hash(entry['stage_sha256'])      # previous_sha256 is compared with the chain below
            if ok and state == 'begun':
                ok = index == len(doc['stages']) - 1                  # one stage at a time: only the last entry may be begun
            if not ok:
                raise Tampered(f'the ledger record is malformed (entry {index})')
            seen.add(entry['stage'])
            entries.append(copy.deepcopy(entry))
        expected = header_sha                                              # the chain: the header, then each finished stage file in turn
        for entry in entries:
            if entry['state'] == 'finished':
                if entry['previous_sha256'] != expected:
                    raise Tampered('the ledger chain is broken (a finished stage record was removed or replaced)')
                expected = entry['stage_sha256']
        return entries

    @staticmethod
    def _notes_of(doc) -> list:
        """The validated notes of a ledger document (none when there is no ``notes`` member); ``Tampered`` for a malformed one."""
        notes = doc.get('notes', [])
        if not isinstance(notes, list) or len(notes) > MAX_NOTES or ('notes' in doc and not notes):      # an empty list is a second byte form of "no notes"
            raise Tampered('the ledger notes are malformed')
        for index, note in enumerate(notes):
            ok = isinstance(note, dict) and set(note) == _NOTE_FIELDS and note['kind'] in NOTE_KINDS and isinstance(note['record'], dict)
            ok = ok and (note['stage'] is None or _valid_stage_name(note['stage']))
            if not ok:
                raise Tampered(f'the ledger notes are malformed (note {index})')
        return copy.deepcopy(notes)

    # -- the state ----
    @property
    def header(self) -> Header:
        return self._header

    @property
    def check(self):
        """The stop function of the ledger, or None: called without arguments while the ledger hashes a file (the owner check, ``begin``, ``finish``,
        ``resume``), through ``run.io.sha_file(..., check=)``. Whatever it raises ends the operation before any write (a wall-clock or memory budget,
        a stop request). It belongs to this object (a ledger that is opened again has none). ``TypeError`` for a value that is neither callable nor None."""
        return self._check

    @check.setter
    def check(self, function) -> None:
        if function is not None and not callable(function):
            raise TypeError('check must be a function without arguments, or None')
        self._check = function

    @property
    def notes(self) -> tuple:
        """The ``Note`` s of the run, in the order in which they were written (stop reasons and E6 records of the supervisor, K5b)."""
        return tuple(Note(stage=note['stage'], kind=note['kind'], record=note['record']) for note in self._notes)

    @property
    def stages(self) -> tuple:
        """``((stage, state), ...)`` in the order in which the stages began; the state is ``begun`` or ``finished``."""
        return tuple((entry['stage'], entry['state']) for entry in self._entries)

    def stage(self, name: str) -> StageRecord:
        """What the ledger recorded for the stage ``name`` (``StageRecord``, read-only); ``LedgerRefused`` if there is no such stage."""
        _require_stage_name(name)
        entry = self._find(name)
        if entry is None:
            raise LedgerRefused(f'stage {name!r} is not in the ledger')
        return StageRecord(stage=entry['stage'], state=entry['state'], inputs=MappingProxyType(dict(entry['inputs'])),
                           outputs=MappingProxyType(dict(entry.get('outputs', {}))))

    def _find(self, stage: str):
        return next((entry for entry in self._entries if entry['stage'] == stage), None)

    def _stage_path(self, stage: str) -> Path:
        return self.run_dir / STAGE_FOLDER / f'{stage}.json'

    def _locate(self, key: str) -> Path:
        path = Path(key)
        return path if path.is_absolute() else self.run_dir / key

    def _chain_before(self, index: int) -> str:
        """The chain value ``q`` of the stage at ``index``: the header hash if no finished stage comes before it, else the stage-file hash of the one before."""
        return self._entries[index - 1]['stage_sha256'] if index > 0 else self._header_sha

    def _unchanged(self) -> None:
        """Ref: ``CampaignLedger._verify``. The ledger and header files must be the bytes this object read or wrote; ``Tampered`` otherwise (a file that is
        gone, or is now a folder or a link, is a change too). Another ``OSError`` (a permission error, a sharing violation) is not a change: it propagates."""
        problem = None
        try:
            for path, expected, what in ((self._ledger_path, self._ledger_sha, 'ledger'), (self._header_path, self._header_sha, 'header')):
                if path.is_symlink() or not path.is_file():
                    problem = f'the {what} file is missing or is no longer a regular file'
                elif run_io.sha_file(path, check=self._check) != expected:
                    problem = f'the {what} file changed since this object read or wrote it'
                if problem is not None:
                    break
        except FileNotFoundError:                                           # removed between the check and the read
            problem = 'the ledger or header file is missing'
        if problem is not None:
            raise Tampered(problem)

    def _require_resumed(self) -> None:
        if not self._resumed:
            raise LedgerRefused('an opened ledger must run resume() before begin() or finish(): nothing is written on a state that was not verified')

    def _commit(self, entries: list, notes=None) -> None:
        """Write the ledger with ``entries`` and ``notes`` (default: the notes it has; one atomic write); this object takes the new state only after
        the write succeeded. A ledger without notes has no ``notes`` member."""
        notes = self._notes if notes is None else notes
        doc = {'schema': SCHEMA_LEDGER, 'header_sha256': self._header_sha, 'stages': copy.deepcopy(entries)}
        if notes:
            doc['notes'] = copy.deepcopy(notes)
        written = run_io.write_json_atomic(self._ledger_path, doc)
        self._entries, self._notes, self._ledger_sha = entries, notes, written

    # -- paths ----
    def _keys(self, paths, *, what: str, external: bool) -> list:
        return _keys_of(self.run_dir, paths, what=what, external=external)

    def _digests(self, keys: list, *, stage: str, what: str, error) -> dict:
        """``{key: SHA-256}`` of the files; ``error`` (with the stage and a count, no path) if one is missing, a link or not a regular file."""
        digests, missing = {}, 0
        for key in keys:
            path = self._locate(key)
            if path.is_symlink() or not path.is_file():
                missing += 1
            else:
                digests[key] = run_io.sha_file(path, check=self._check)
        if missing:
            raise error(f'stage {stage!r}: {missing} of {len(keys)} {what} file(s) missing or not a regular file')
        return digests

    def _verify_files(self, stage: str, what: str, recorded: dict) -> None:
        """Every recorded file exists and has its recorded hash; ``Tampered`` with the stage and a count otherwise."""
        bad = 0
        for key, digest in recorded.items():
            path = self._locate(key)
            if path.is_symlink() or not path.is_file() or run_io.sha_file(path, check=self._check) != digest:
                bad += 1
        if bad:
            raise Tampered(f'stage {stage!r}: {bad} of {len(recorded)} {what} file(s) changed or missing')

    # -- the operations ----
    # Ref: c_ept/workflows/run_s3.py ``Ledger.begin`` / ``Ledger.finish`` (S3/S4a stage ledger, frozen): begin records the inputs and refuses to begin again
    # on other inputs; finish records the outputs. Changed: the hashes are taken here (``run.io.sha_file``), one stage is begun at a time, an input
    # that an earlier stage recorded as an output must still be that file, and the write order and ``resume`` make every crash point recoverable
    # (A5-F7; the S4a order was stage JSON, RUN.json, ledger). Agreement test: ``test_begin_and_finish_agree_with_the_s3_stage_ledger_they_refer_to``.
    def begin(self, stage: str, *, inputs=()) -> None:
        """Begin ``stage``: record the hashes of ``inputs`` (paths relative to the run folder, or absolute) in the ledger. One write.

        Begun again (after a stop) on inputs with the same hashes: no write. ``LedgerRefused`` for an opened ledger that was not resumed, a stage that
        is finished or has a stage file (call ``resume`` first), another stage that is still begun, inputs that differ from those the stage was begun
        on, and an input that is missing, not acceptable or one of the ledger's own files. ``Tampered`` if the ledger file changed behind this
        object, or if an input is the output of a finished stage but no longer has the hash that was recorded for it.
        """
        self._unchanged()
        self._require_resumed()
        _require_stage_name(stage)
        keys = self._keys(inputs, what='inputs', external=True)
        current = self._find(stage)
        if current is not None and current['state'] == 'finished':
            raise LedgerRefused(f'stage {stage!r} is already finished')
        for entry in self._entries:
            if entry['state'] == 'begun' and entry['stage'] != stage:
                raise LedgerRefused(f'stage {entry["stage"]!r} is still begun; finish it before beginning another stage')
        if os.path.lexists(self._stage_path(stage)):
            raise LedgerRefused(f'stage {stage!r} has a stage file that is not recorded; resume() completes it')
        digests = self._digests(keys, stage=stage, what='input', error=LedgerRefused)
        certified = {key: digest for entry in self._entries if entry['state'] == 'finished' for key, digest in entry['outputs'].items()}
        differing = sum(1 for key, digest in digests.items() if key in certified and certified[key] != digest)
        if differing:
            raise Tampered(f'stage {stage!r}: {differing} of {len(digests)} input file(s) differ from the output that an earlier stage recorded')
        if current is not None:
            if current['inputs'] != digests:
                raise LedgerRefused(f'stage {stage!r} was begun on other inputs; a run folder never mixes two sets of inputs')
            return
        self._commit([*self._entries, {'stage': stage, 'state': 'begun', 'inputs': digests}])

    def finish(self, stage: str, outputs) -> str:
        """Finish ``stage``: check that its inputs still have their begin hashes, hash ``outputs`` (paths inside the run folder), write the stage
        file, then the ledger. Two writes, in this order. Returns the SHA-256 of the stage file.

        ``LedgerRefused`` for an opened ledger that was not resumed, a stage that was not begun, is finished, has a stage file (call ``resume``
        first), or an output that is missing, not acceptable or one of the ledger's own files; nothing is written. ``Tampered`` if an input changed
        or vanished since ``begin``; nothing is written.
        """
        self._unchanged()
        self._require_resumed()
        _require_stage_name(stage)
        entry = self._find(stage)
        if entry is None:
            raise LedgerRefused(f'stage {stage!r} was not begun')
        if entry['state'] == 'finished':
            raise LedgerRefused(f'stage {stage!r} is already finished')
        if os.path.lexists(self._stage_path(stage)):
            raise LedgerRefused(f'stage {stage!r} has a stage file that is not recorded; resume() completes it')
        keys = self._keys(outputs, what='outputs', external=False)
        now = self._digests(sorted(entry['inputs']), stage=stage, what='input', error=Tampered)
        changed = sum(1 for key, digest in entry['inputs'].items() if now[key] != digest)
        if changed:
            raise Tampered(f'stage {stage!r}: {changed} of {len(now)} input file(s) changed since begin')
        hashed = self._digests(keys, stage=stage, what='output', error=LedgerRefused)
        previous = self._chain_before(len(self._entries) - 1)                  # the stage being finished is the last entry
        stage_sha = run_io.write_json_atomic(self._stage_path(stage), {'schema': SCHEMA_STAGE, 'stage': stage, 'header_sha256': self._header_sha,
                                                                        'previous_sha256': previous, 'inputs': entry['inputs'], 'outputs': hashed})
        finished = {'stage': stage, 'state': 'finished', 'inputs': entry['inputs'], 'outputs': hashed, 'stage_sha256': stage_sha,
                    'previous_sha256': previous}
        self._commit([finished if other['stage'] == stage else other for other in self._entries])
        return stage_sha

    def note(self, stage, kind: str, record) -> None:
        """Append a note to the ledger (one write): why ``stage`` stopped (``kind='stop'``) or the record of an E6 check (``kind='e6'``).

        ``stage``: a stage name, or None for the run as a whole (the stage need not be in the ledger yet: the E6 check comes before ``begin``).
        ``record``: a JSON object (strict JSON; the writer refuses anything else before the disk is touched). The ledger does not look into the record:
        the caller keeps it free of paths, keys and secrets (the supervisor scrubs its text first). ``LedgerRefused`` for an opened ledger that was
        not resumed, a stage name or a kind that is not acceptable, a record that is not a dict, and more than ``MAX_NOTES`` notes; ``IORefused`` for a
        record that is not strict JSON; ``Tampered`` if the ledger file changed behind this object. Notes are no part of the chain.
        """
        self._unchanged()
        self._require_resumed()
        if stage is not None:
            _require_stage_name(stage)
        if kind not in NOTE_KINDS:
            raise LedgerRefused(f'a note kind must be one of {", ".join(NOTE_KINDS)}')
        if not isinstance(record, dict):
            raise LedgerRefused('a note record must be a JSON object')
        if len(self._notes) >= MAX_NOTES:
            raise LedgerRefused(f'the ledger holds the largest number of notes ({MAX_NOTES})')
        self._commit(self._entries, [*self._notes, {'stage': stage, 'kind': kind, 'record': copy.deepcopy(record)}])

    def resume(self) -> list:
        """Verify the run and complete what a stop left half done; return the finished stages, in order. Idempotent.

        Every finished stage: its stage file has the hash the ledger records and says what the ledger says, the chain holds, and every input and
        output file has its recorded hash. The begun stage, if its stage file exists (a stop between the two writes of ``finish``): the same checks
        against the begun record, and then the ledger is completed (the only write). A stage folder item that no entry explains, a missing or edited
        file, or a record that is malformed or belongs to another run is ``Tampered``; nothing is written then, and the ledger stays unresumed. An
        ``OSError`` while hashing propagates (it is not evidence of a change).
        """
        self._unchanged()
        self._check_stage_folder()
        for entry in self._entries:
            if entry['state'] == 'finished':
                self._verify_stage(entry)
        last = self._entries[-1] if self._entries else None
        if last is not None and last['state'] == 'begun' and os.path.lexists(self._stage_path(last['stage'])):
            doc, stage_sha = self._verify_stage(last)
            finished = {'stage': last['stage'], 'state': 'finished', 'inputs': last['inputs'], 'outputs': doc['outputs'], 'stage_sha256': stage_sha,
                        'previous_sha256': doc['previous_sha256']}
            self._commit([*self._entries[:-1], finished])
        self._resumed = True
        return [entry['stage'] for entry in self._entries if entry['state'] == 'finished']

    def _check_stage_folder(self) -> None:
        """Every item of the stage folder is ``<stage>.json`` of an entry (a leftover ``.tmp`` of an atomic write aside); else an unknown stage."""
        folder = self.run_dir / STAGE_FOLDER
        if not os.path.lexists(folder):
            return
        if folder.is_symlink() or not folder.is_dir():
            raise Tampered('the stage folder is not a folder')
        known = {entry['stage'] for entry in self._entries}
        unexpected = 0
        for item in folder.iterdir():
            if item.name.endswith('.tmp'):
                continue
            if item.name.endswith('.json') and item.name[:-5] in known and item.is_file() and not item.is_symlink():
                continue
            unexpected += 1
        if unexpected:
            raise Tampered(f'the stage folder holds {unexpected} unexpected item(s) (an unknown stage?)')

    # Ref: c_ept/comparison/runtime.py ``_receipt`` / ``_write_receipt`` (T009 stage receipt, frozen): a receipt lists the artifact hashes by path
    # relative to the run folder, is bound to its campaign (there ``config_sha256``, here the header hash) and every hash is checked again before
    # the stage counts. Added: the input hashes, the stage-file hash in the ledger, the chain to the previous stage file, and the completion of a
    # ledger that lags behind the stage file.
    def _verify_stage(self, entry: dict):
        """Check the stage file of ``entry`` against the entry, the header, the chain and the files; return ``(stage file object, its SHA-256)``."""
        stage = entry['stage']
        doc, stage_sha = _read_record(self._stage_path(stage), f'stage {stage!r}')
        ok = set(doc) == _STAGE_FIELDS and doc['schema'] == SCHEMA_STAGE and doc['stage'] == stage and doc['header_sha256'] == self._header_sha
        ok = ok and _valid_digests(doc['inputs'], external=True) and _valid_digests(doc['outputs'], external=False)
        if not ok:
            raise Tampered(f'stage {stage!r}: the stage file is malformed or belongs to another run')
        if doc['previous_sha256'] != self._chain_before(self._entries.index(entry)):
            raise Tampered(f'stage {stage!r}: the stage file is not chained to the stage before it')
        if doc['inputs'] != entry['inputs'] or (entry['state'] == 'finished' and doc['outputs'] != entry['outputs']):
            raise Tampered(f'stage {stage!r}: the stage file and the ledger record different files')
        if entry['state'] == 'finished' and stage_sha != entry['stage_sha256']:
            raise Tampered(f'stage {stage!r}: the stage file changed since it was recorded')
        self._verify_files(stage, 'input', doc['inputs'])
        self._verify_files(stage, 'output', doc['outputs'])
        return doc, stage_sha

    def check_spec(self, current: bytes) -> None:
        """``check_spec_growth`` with the fingerprint in this run's header: the current decision record may have grown by blocks, nothing else.

        It reads only the header, so it needs no ``resume``. Call it on the bytes that ``guards.spec_check`` just checked (see the module docstring).
        """
        check_spec_growth(current, top=self._header.spec_top, blocks=self._header.spec_blocks)

    def check_spec_file(self, path) -> str:
        """The whole resume rule for the decision record at ``path``, on one read: the record must pass ``guards.spec_check`` (the S1-S3 constants
        check, bound to the bytes that were read by ``expected_sha256``) and must have grown by blocks only (``check_spec``). Returns the SHA-256 of
        the bytes (record it in the run's log: the record grows, so the hash changes from resume to resume).

        ``guards.GateRefused('spec', ...)`` for every refusal, including a record that cannot be read or is larger than ``MAX_SPEC_BYTES`` and one that
        changes between the read and the constants check (``hash mismatch``); the message holds no path, record value or hash. It reads only the header,
        so it needs no ``resume``.
        """
        data = None
        try:
            with Path(path).open('rb') as stream:
                data = stream.read(MAX_SPEC_BYTES + 1)
        except OSError:
            data = None
        if data is None or len(data) > MAX_SPEC_BYTES:
            raise guards.GateRefused('spec', 'unreadable: the decision record cannot be read or is too large')
        digest = guards.spec_check(path, expected_sha256=hashlib.sha256(data).hexdigest())
        self.check_spec(data)
        return digest
