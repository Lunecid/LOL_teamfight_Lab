"""Strict JSON and hash input/output of the run kernel (S4b-prep K1, decision D11). Standard library only.

Six functions, each consolidating what the earlier stacks did in several places (those modules are frozen and unchanged):

* ``read_json``: ``comparison.protocol.read_json`` (regular file, no symbolic link, 16 MiB, duplicate keys and non-finite constants
  refused) plus the float hook of ``run_s4a._read`` (``1e999`` is ``inf`` and never reaches ``parse_constant``), and a top level that
  must be an object.
* ``read_json_with_sha`` (S4b-prep K5a): the same reader and the same refusals, and it hands back the SHA-256 of the bytes that it parsed.
  The file is read once. A caller that needs the object and the hash of a file (a profile, a ledger record) used to read it twice, once to
  parse and once to hash, and a file that was changed and changed back in between (A, B, A) gave the hash of B with the object of A. With one
  read there is no second content to disagree with, and ``check`` lets the caller test the same bytes (a profile refuses a carriage return).
* ``parse_json_object`` and ``canonical_sha256`` (S4b-prep K5a review fix): the strict parse of bytes that are already in memory (the same refusals
  and messages as the file readers), and the SHA-256 of the canonical JSON of a value, ``json.dumps(value, sort_keys=True, separators=(',', ':'),
  ensure_ascii=False, allow_nan=False)`` encoded as UTF-8. Two spellings of one JSON value (other white space, other member order, escaped or
  plain non-ASCII text) have the same canonical hash. The ledger uses them to compare a decision record by its blocks instead of by its bytes.
* ``write_json_atomic``: ``run_s4a._write`` (missing folders made, ``allow_nan=False``), ``evaluation.runtime.atomic_json`` (the text
  recipe: indent 2, sorted keys, ``ensure_ascii=False``, one final newline) and ``data.store_core.write_atomic`` (temporary file,
  flush, fsync, rename; the bytes are written as bytes). ``atomic_json`` opens its temporary file in text mode, which on Windows turns
  ``\\n`` into ``\\r\\n``: the hash of its text is then not the hash of the file. This writer writes bytes, so the returned hash is
  the hash of the file.
* ``sha_file``: ``comparison.protocol.sha_file``, unchanged (1 MiB chunks, ``check`` before, between and after).

Symbols
-------
    p, p'    the target path and the temporary path (same folder as p)
    b        the bytes written to p, ``b = utf8(dumps(obj)) + '\\n'``
    h        the returned hash, ``h = SHA-256(b)`` as 64 lowercase hex digits
    M        ``max_bytes``, the largest file ``read_json`` accepts (16 MiB by default)
    raw      the bytes read from the file; at most M + 1 of them are ever read
    M0       the default bound, 16 MiB = 16 << 20 bytes
    d, s     the object that ``read_json_with_sha`` returns and its hash, ``s = SHA-256(raw)`` as 64 lowercase hex digits
    c        the optional ``check`` function of ``read_json_with_sha``; it is called as ``c(raw)``

Equations
---------
    read accepts  <=>  not link(p)  and  is_file(p)  and  |raw| <= M  and  raw is UTF-8 JSON
                       and every object has distinct keys  and  every number is finite  and  the top level is an object
    read_json_with_sha(p) = (d, s)  with  d = parse(raw)  and  s = SHA-256(raw)  for the one ``raw`` that was read; it accepts exactly when
                       ``read`` accepts and c(raw) returns (c is called after the parse, so a refusal of the parse comes first)
    one read      :    the file is opened once; d and s are functions of the same ``raw``, whatever the file becomes after that read
    write accepts <=>  obj is a dict  and  every key, at any depth, is a str  and  obj is strict JSON (finite numbers, JSON types,
                       UTF-8 encodable text, no cycle)  and  |b| <= M0
    write         :    file(p) after  in  { file(p) before, b }        (all or nothing: a stop before the rename keeps the old file)
    write returns :    h = SHA-256(file(p) after)                       (the hash is that of the bytes on disk)
    round trip    :    write accepts obj  =>  read_json(p) == obj      (a tuple comes back as a list; nothing else changes)

해석
----
읽기는 의심스러운 입력을 모두 거부한다. 중복 키는 뒤의 값이 앞의 값을 몰래 덮어쓰는 것을, 유한하지 않은 수는 NaN이 비교와 해시를 조용히
망가뜨리는 것을, 심볼릭 링크는 잠근 파일 대신 다른 파일이 읽히는 것을, 크기 한도는 잘못된 입력이 메모리를 채우는 것을 막는다. 객체가 아닌
최상위 값(목록, 숫자 등)은 프로파일·원장·정체성 파일이 아니다. 거부 메시지에는 키나 값을 싣지 않는다(키가 경기 키일 수 있다).

``read_json_with_sha``는 같은 읽기를 하되 그 바이트의 해시도 함께 돌려준다. 파일을 한 번만 읽으므로, 객체와 해시가 서로 다른 내용에서 나올 수 없다.
이전에는 파싱과 해시를 따로 읽었기 때문에 그 사이에 파일이 B로 바뀌었다가 A로 돌아오면(A→B→A) B의 해시와 A의 설정을 함께 받아들였다. 읽은 바이트에 대한
추가 검사(프로파일의 CR 거부)는 같은 바이트를 ``check``에 넘겨 하므로 그 검사도 파싱·해시와 같은 내용에 대한 것이다.

``parse_json_object``는 이미 메모리에 있는 바이트를 같은 엄격 규칙으로 파싱하고, ``canonical_sha256``은 값의 정규 JSON(키 정렬, 공백 없음, 비ASCII 글자 그대로, UTF-8)의
해시를 낸다. 같은 값을 공백이나 키 순서만 달리 적은 두 파일은 정규 해시가 같다. 그래서 결정 기록처럼 내용은 같고 모양만 바뀔 수 있는 JSON을 바이트가 아니라 값으로 비교할 수 있다.

쓰기는 읽기가 돌려줄 수 있는 것만 쓴다(왕복 불변식). 그래서 읽기가 거부할 모양(객체가 아닌 최상위 값, 기본 한도 M0를 넘는 크기)과, JSON이 조용히
바꾸어 쓰는 모양(문자열이 아닌 키: 정수 20261022는 글자 "20261022"가 되어 읽으면 다른 객체가 된다)은 디스크를 건드리기 전에 거부한다. 이렇게
하지 않으면 쓸 때의 실수가 다시 시작할 때의 거부나 거짓 변조 판정(원장·정체성 재검증)으로 나타난다.

쓰기는 임시 파일(같은 폴더, 프로세스 번호와 난수가 든 이름)에 바이트를 쓰고 flush, fsync한 뒤에야 이름을 바꾼다. 같은 폴더여야 이름 바꾸기가
한 번의 원자적 연산이 되고, fsync가 앞서야 이름이 바뀐 순간 파일 내용이 이미 디스크에 있다. 직렬화는 파일을 건드리기 전에 끝내므로 직렬화
거부(NaN, 지원하지 않는 형, UTF-8로 인코딩할 수 없는 글자)는 옛 파일도 폴더도 바꾸지 않는다. 쓰다가 예외로 멈추면 임시 파일을 지우고 옛 파일은
그대로 남는다(전원이 꺼지는 경우에는 임시 파일만 남을 수 있으나 대상은 여전히 옛 파일이다).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

_CHUNK = 1024 * 1024         # sha_file reads 1 MiB at a time, as comparison.protocol.sha_file does
_MAX_JSON_BYTES = 16 << 20   # M0: the default bound of read_json, and the largest document that write_json_atomic writes


class IORefused(ValueError):
    """A strict input or output refusal of ``run.io``: the file or the object is not acceptable.

    A ``ValueError`` like the refusals of the earlier stacks, so a caller that catches ``ValueError`` sees it. The message names the
    kind of problem and never a key or a value.
    """


def _unique_keys(pairs):
    """``object_pairs_hook``: a dict from the (key, value) pairs of one JSON object; a repeated key is refused (no key in the message)."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise IORefused('duplicate JSON key')
        result[key] = value
    return result


def _finite_float(text):
    """``parse_float`` hook: ``float(text)``, refused when it is not finite. ``json.loads('1e999')`` is ``inf`` and skips ``parse_constant``."""
    value = float(text)
    if not math.isfinite(value):
        raise IORefused('non-finite JSON number')
    return value


def _no_constant(name):
    """``parse_constant`` hook: ``NaN``, ``Infinity`` and ``-Infinity`` are refused."""
    raise IORefused('non-finite JSON constant')


def _only_string_keys(obj) -> bool:
    """True when every key of every dict inside ``obj``, at any depth (lists and tuples included), is a ``str``.

    Iterative (no recursion limit); a container that was seen is not visited again, so a shared sub-object or a cycle cannot loop.
    """
    stack, seen = [obj], set()
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if id(node) in seen:
                continue
            seen.add(id(node))
            if not all(isinstance(key, str) for key in node):
                return False
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(node)
    return True


def _read_bytes(path, max_bytes) -> bytes:
    """The bytes of the regular file ``path`` (one open, one read of at most ``max_bytes + 1`` bytes); ``IORefused`` for a link, a missing file,
    a folder, or more than ``max_bytes`` bytes. The one place where ``read_json`` and ``read_json_with_sha`` touch the file."""
    path = Path(path)
    if path.is_symlink():
        raise IORefused('symbolic link refused: a regular JSON file is required')
    if not path.is_file():
        raise IORefused('regular JSON file required')
    with path.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise IORefused(f'JSON file too large (more than {max_bytes} bytes)')
    return raw


def parse_json_object(raw) -> dict:
    """The JSON object in the bytes ``raw`` under the strict rules of ``read_json``; ``IORefused`` for text that is not UTF-8 JSON, a repeated key,
    a non-finite number, a document nested too deeply, or a top level that is not an object. The refusal is raised outside any ``except`` block
    (no context, no cause), with the messages of the file readers. ``TypeError`` if ``raw`` is not bytes. There is no size bound: the caller holds the bytes."""
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError('raw must be bytes')
    refusal = None
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_keys, parse_float=_finite_float, parse_constant=_no_constant)
    except IORefused:
        raise
    except RecursionError:
        refusal = 'JSON nested too deeply'
    except ValueError:                                              # JSONDecodeError and UnicodeDecodeError are ValueErrors
        refusal = 'file is not valid UTF-8 JSON'
    if refusal is not None:
        # Raised after the except block, so that the exception has no context: the JSONDecodeError holds the whole text (``doc``) and the
        # UnicodeDecodeError the raw bytes (``object``), and neither is reachable from the refusal.
        raise IORefused(refusal)
    if not isinstance(value, dict):
        raise IORefused('the top level of the JSON file must be an object')
    return value


def read_json(path, *, max_bytes=_MAX_JSON_BYTES) -> dict:
    """The JSON object in the file ``path``, read strictly; ``IORefused`` for anything that is not acceptable.

    Refused: a path that is a symbolic link, is missing or is not a regular file; a file of more than ``max_bytes`` bytes (at most
    ``max_bytes + 1`` bytes are read, whatever size the file system reports, and the size is checked before anything is parsed); text
    that is not UTF-8 or not JSON (a byte-order mark included); a repeated key in any object, at any depth; ``NaN``, ``Infinity``,
    ``-Infinity`` and a number that overflows to infinity (``1e999``); a document nested too deeply; a top-level value that is not an
    object. The default ``max_bytes`` is 16 MiB.

    Only the last component of ``path`` is checked for a link, as in ``comparison.protocol.read_json``: a link in a parent folder or a
    Windows junction is followed, so a caller that must keep a file under a root checks the containment itself. Errors of the operating
    system (permission denied, a file removed between the check and the open) are not refusals: they propagate as ``OSError``.
    """
    return parse_json_object(_read_bytes(path, max_bytes))


def read_json_with_sha(path, *, max_bytes=_MAX_JSON_BYTES, check=None) -> tuple[dict, str]:
    """``(object, sha256)`` of the file ``path``: the strict object of ``read_json`` and the SHA-256 (64 lowercase hex digits) of the bytes
    that were parsed. The file is read once, so the pair always describes one content (``sha_file(path)`` of the same bytes is the same hash).

    Every refusal of ``read_json`` applies, with the same messages. ``check``: an optional function ``check(raw)`` that gets the parsed bytes
    after the strict parse accepted them and before the pair is returned; whatever it raises propagates (a profile uses it to refuse a carriage
    return, so that the test is made on the bytes that were parsed and hashed). It is not called for a file that the strict reader refuses.
    """
    raw = _read_bytes(path, max_bytes)
    value = parse_json_object(raw)
    if check is not None:
        check(raw)
    return value, hashlib.sha256(raw).hexdigest()


def canonical_sha256(value) -> str:
    """The SHA-256 (64 lowercase hex digits) of the canonical JSON of ``value``:
    ``json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)`` encoded as UTF-8.

    A tuple is a list; ``1`` and ``1.0`` are different texts and so have different hashes. ``IORefused`` (no context) for what is not strict JSON
    (NaN or infinity, a type JSON cannot hold, text that is not UTF-8 encodable, a cycle) and for a key that is not text at any depth (JSON would
    silently turn ``1`` into ``"1"``)."""
    refusal, data = None, b''
    if not _only_string_keys(value):
        refusal = 'JSON object keys must be strings'
    else:
        try:
            data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
        except (TypeError, ValueError, RecursionError) as error:
            refusal = f'value cannot be hashed as strict JSON ({type(error).__name__})'
    if refusal is not None:
        raise IORefused(refusal)
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path, obj) -> str:
    """Write ``obj`` as JSON to ``path`` all or nothing, and return the SHA-256 (64 hex digits) of the bytes written.

    The bytes are ``json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\\n'`` as UTF-8. Missing
    folders are made. The temporary file is in the folder of ``path``; it is written, flushed and fsynced, then renamed over ``path``.

    The writer writes only what ``read_json`` gives back: ``read_json(path) == obj`` (a tuple comes back as a list). ``IORefused``, before
    the disk is touched (no file, no temporary file, no folder), for: a top level that is not a ``dict``; a key that is not a ``str``
    at any depth (JSON would write ``20261022`` as the text ``"20261022"``, which reads back as another object); an object that is not
    strict JSON (NaN or infinity, a type JSON cannot hold, text that is not UTF-8 encodable, a cycle); output of more than 16 MiB, the
    default bound of ``read_json``. If the write stops before the rename (any exception, ``KeyboardInterrupt`` included), the temporary
    file is removed and ``path`` is as it was.
    """
    refusal, data = None, b''
    if not isinstance(obj, dict):
        refusal = 'the top level of a JSON file must be an object'
    else:
        try:
            data = (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        except (TypeError, ValueError, RecursionError) as error:       # ValueError: NaN or inf, a cycle, a lone surrogate (UnicodeEncodeError)
            refusal = f'object cannot be written as strict JSON ({type(error).__name__})'
        else:
            if len(data) > _MAX_JSON_BYTES:
                refusal = f'JSON document too large to be read back (more than {_MAX_JSON_BYTES} bytes)'
            elif not _only_string_keys(obj):
                refusal = 'JSON object keys must be strings'
    if refusal is not None:
        raise IORefused(refusal)                                       # outside the except block: no context holds the object
    digest = hashlib.sha256(data).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        with open(temporary, 'xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:                                                # already gone, or not removable: the original error matters
            pass
        raise
    return digest


def sha_file(path, *, check=None) -> str:
    """The SHA-256 (64 hex digits) of the file ``path``, read in 1 MiB chunks.

    ``check``: an optional function without arguments, called before the first chunk, after each chunk was read (before it is hashed)
    and after the last one, so that a caller can stop a long hash (a wall-clock budget, a stop request) by raising from it. Whatever
    ``check`` raises propagates and ends the hashing. A missing file raises the ``OSError`` of ``open``.
    """
    if check is not None:
        check()
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(_CHUNK), b''):
            if check is not None:
                check()
            h.update(block)
    if check is not None:
        check()
    return h.hexdigest()
