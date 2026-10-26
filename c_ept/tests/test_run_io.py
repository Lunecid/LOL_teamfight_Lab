"""Tests of the strict JSON and hash input/output of the run kernel (``c_ept.run.io``, S4b-prep K1, D11).

The module consolidates the strict readers and the atomic writers of the earlier stacks: ``comparison.protocol`` (read_json, sha_file),
``workflows.run_s4a`` (_read, _write), ``evaluation.runtime`` (atomic_json) and ``data.store_core`` (write_atomic). The tests pin each
refusal of the reader, the all-or-nothing behaviour of the writer (a write that stops before the rename leaves the old file as it was)
and the identity "the returned hash is the SHA-256 of the bytes on disk". They use synthetic files in ``tmp_path`` only.

해석: 실행 핵심의 원장·프로파일·정체성 파일은 모두 이 입출력 하나로 읽고 쓴다. 읽기는 중복 키, 유한하지 않은 수, 심볼릭 링크, 너무 큰 파일,
객체가 아닌 최상위 값을 거부하고, 쓰기는 임시 파일을 같은 폴더에 쓰고 fsync한 뒤에만 이름을 바꾸므로 중간에 멈춰도 옛 파일이 그대로 남는다.
"""
import ast
import hashlib
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

from c_ept.run import io as run_io


def put(path: Path, text: str) -> Path:
    """Write ``text`` as UTF-8 bytes (no newline translation) and return the path."""
    path.write_bytes(text.encode('utf-8'))
    return path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- the interface ------------------------------------------------------------------------------------------------------------------
def test_the_interface_has_exactly_the_planned_names_and_signatures():
    read = inspect.signature(run_io.read_json)
    assert list(read.parameters) == ['path', 'max_bytes']
    assert read.parameters['max_bytes'].kind is inspect.Parameter.KEYWORD_ONLY
    assert read.parameters['max_bytes'].default == 16 << 20
    assert list(inspect.signature(run_io.write_json_atomic).parameters) == ['path', 'obj']
    digest = inspect.signature(run_io.sha_file)
    assert list(digest.parameters) == ['path', 'check']
    assert digest.parameters['check'].kind is inspect.Parameter.KEYWORD_ONLY and digest.parameters['check'].default is None


def test_the_package_and_its_io_module_import_only_the_standard_library():
    """The kernel reads its first file before anything heavy is imported: no model, numpy or c_ept import in ``c_ept/run/__init__.py`` and ``io.py``."""
    for module in (run_io, sys.modules['c_ept.run']):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, 'a relative import in the strict IO module'
                names.add((node.module or '').split('.')[0])
        assert names <= set(sys.stdlib_module_names), sorted(names - set(sys.stdlib_module_names))


def test_a_refusal_is_a_value_error():
    """Callers that catch ``ValueError`` (the refusals of the earlier stacks are ValueError subclasses) see the refusals of this module."""
    assert issubclass(run_io.IORefused, ValueError)


# ---- read_json: what it accepts ------------------------------------------------------------------------------------------------------
def test_read_json_returns_the_parsed_object(tmp_path):
    doc = {'a': 1, 'b': [1.5, None, True, 'x'], 'c': {'d': {'e': []}}, '한글': '값'}
    path = put(tmp_path / 'doc.json', json.dumps(doc, ensure_ascii=False))
    assert run_io.read_json(path) == doc
    assert run_io.read_json(str(path)) == doc                      # a str path is as good as a Path


def test_read_json_accepts_crlf_and_a_file_of_exactly_max_bytes(tmp_path):
    text = '{\r\n  "a": 1\r\n}\r\n'
    path = put(tmp_path / 'crlf.json', text)
    assert run_io.read_json(path) == {'a': 1}
    assert run_io.read_json(path, max_bytes=len(text.encode('utf-8'))) == {'a': 1}      # the bound is inclusive


def test_read_json_agrees_with_the_comparison_reader_on_a_well_formed_file(tmp_path):
    """Consolidation, not a new meaning: for a file that the earlier strict reader accepts, the result is the same."""
    from c_ept.comparison import protocol
    path = put(tmp_path / 'ok.json', json.dumps({'x': [1, 2.5, {'y': None}], 'z': 'w'}))
    assert run_io.read_json(path) == protocol.read_json(path)


# ---- read_json: each refusal ----------------------------------------------------------------------------------------------------------
def test_read_json_refuses_a_duplicate_key_at_the_top_level_and_below(tmp_path):
    for text in ('{"a": 1, "a": 2}', '{"a": {"b": 1, "b": 2}}', '{"a": [{"b": 1, "b": 1}]}'):
        with pytest.raises(run_io.IORefused, match='duplicate'):
            run_io.read_json(put(tmp_path / 'dup.json', text))


def test_the_duplicate_key_refusal_does_not_print_the_key(tmp_path):
    """A key can be a match key or a name taken from an input: the message says that a key is duplicated, never which one."""
    path = put(tmp_path / 'dup.json', '{"SECRET-KEY-1": 1, "SECRET-KEY-1": 2}')
    with pytest.raises(run_io.IORefused) as info:
        run_io.read_json(path)
    assert 'SECRET-KEY-1' not in str(info.value)


@pytest.mark.parametrize('text', ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '{"a": [1, NaN]}', '{"a": {"b": -Infinity}}'])
def test_read_json_refuses_the_non_finite_constants(tmp_path, text):
    with pytest.raises(run_io.IORefused, match='non-?finite'):
        run_io.read_json(put(tmp_path / 'nf.json', text))


@pytest.mark.parametrize('text', ['{"a": 1e999}', '{"a": -1e999}', '{"a": [0.5, 1E400]}'])
def test_read_json_refuses_a_number_that_overflows_to_infinity(tmp_path, text):
    """``json.loads`` turns ``1e999`` into ``inf`` without calling ``parse_constant``; the float hook closes that gap (run_s4a._read had it)."""
    with pytest.raises(run_io.IORefused, match='non-?finite'):
        run_io.read_json(put(tmp_path / 'big.json', text))


@pytest.mark.parametrize('text', ['[]', '[{"a": 1}]', '"text"', '3', '3.5', 'null', 'true'])
def test_read_json_refuses_a_top_level_value_that_is_not_an_object(tmp_path, text):
    with pytest.raises(run_io.IORefused, match='object'):
        run_io.read_json(put(tmp_path / 'top.json', text))


def test_read_json_refuses_a_file_over_max_bytes(tmp_path):
    path = put(tmp_path / 'big.json', '{"a": "' + 'x' * 100 + '"}')
    size = path.stat().st_size
    assert run_io.read_json(path, max_bytes=size) == {'a': 'x' * 100}
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json(path, max_bytes=size - 1)
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json(path, max_bytes=10)


def test_the_size_check_comes_before_the_parsing(tmp_path, monkeypatch):
    """An oversize file is refused as too large even when its content would be refused too, and ``json.loads`` is not called for it (M2)."""
    path = put(tmp_path / 'big.json', '{"a": NaN, "a": 1e999, "pad": "' + 'x' * 100 + '"}')       # duplicate key, NaN and overflow, and too large
    size = path.stat().st_size
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json(path, max_bytes=size - 1)

    def spy(*args, **kwargs):
        raise AssertionError('json.loads was called')
    monkeypatch.setattr(run_io.json, 'loads', spy)
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json(path, max_bytes=size - 1)                  # over the bound: refused before the parse
    with pytest.raises(AssertionError, match='json.loads was called'):
        run_io.read_json(path, max_bytes=size)                      # within the bound: the parse is reached, so the spy does watch it


@pytest.mark.parametrize('payload', [b'{"a": ', b'\xef\xbb\xbf{}', b'\xff\xfe{}', b'{"a": "\xff"}', b'[' * 100000], ids=range(5))
def test_a_parse_refusal_holds_neither_a_cause_nor_a_context(tmp_path, payload):
    """``from None`` only hides the context from the printed traceback. The refusal is raised after the ``except`` block, so that the
    ``JSONDecodeError`` (whose ``doc`` is the whole text) and the ``UnicodeDecodeError`` (whose ``object`` is the raw bytes) are not in
    the exception chain at all (M1)."""
    path = tmp_path / 'bad.json'
    path.write_bytes(payload)
    with pytest.raises(run_io.IORefused) as info:
        run_io.read_json(path)
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_read_json_refuses_a_document_nested_too_deeply(tmp_path):
    with pytest.raises(run_io.IORefused, match='deeply'):
        run_io.read_json(put(tmp_path / 'deep.json', '{"a":' * 100000 + '1' + '}' * 100000))


def test_the_size_bound_does_not_depend_on_the_size_the_file_system_reports(tmp_path, monkeypatch):
    """The bytes actually read are bounded too (a file that grows after a stat is still refused)."""
    path = put(tmp_path / 'grow.json', '{"a": "' + 'x' * 100 + '"}')
    real = Path.stat

    def small_stat(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        if self == path:
            class Liar:
                st_size = 1
                def __getattr__(self, name):
                    return getattr(result, name)
            return Liar()
        return result
    monkeypatch.setattr(Path, 'stat', small_stat)
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json(path, max_bytes=20)


def test_read_json_refuses_a_symbolic_link(tmp_path, monkeypatch):
    """Deterministic version: the file reports itself as a link (a real link needs a privilege that not every machine grants)."""
    target = put(tmp_path / 'target.json', '{"a": 1}')
    link = tmp_path / 'link.json'
    put(link, '{"a": 1}')
    real = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda self: True if self == link else real(self))
    assert run_io.read_json(target) == {'a': 1}
    with pytest.raises(run_io.IORefused, match='link'):
        run_io.read_json(link)


def test_read_json_refuses_a_real_symbolic_link(tmp_path):
    target = put(tmp_path / 'target.json', '{"a": 1}')
    link = tmp_path / 'link.json'
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip('this machine does not let the test create a symbolic link')
    with pytest.raises(run_io.IORefused, match='link'):
        run_io.read_json(link)
    dangling = tmp_path / 'dangling.json'
    os.symlink(tmp_path / 'missing.json', dangling)
    with pytest.raises(run_io.IORefused):
        run_io.read_json(dangling)


def test_read_json_refuses_a_missing_file_and_a_folder(tmp_path):
    with pytest.raises(run_io.IORefused, match='regular'):
        run_io.read_json(tmp_path / 'missing.json')
    folder = tmp_path / 'folder.json'
    folder.mkdir()
    with pytest.raises(run_io.IORefused, match='regular'):
        run_io.read_json(folder)


@pytest.mark.parametrize('payload', [b'', b'   \n', b'{"a": ', b'{"a": 1,}', b"{'a': 1}", b'{"a": 1} trailing', b'\xef\xbb\xbf{"a": 1}',
                                     b'{"a": "\xff"}', b'\xff\xfe{\x00}\x00'])
def test_read_json_refuses_text_that_is_not_json_or_not_utf8(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_bytes(payload)
    with pytest.raises(run_io.IORefused):
        run_io.read_json(path)


def test_max_bytes_is_keyword_only(tmp_path):
    path = put(tmp_path / 'a.json', '{}')
    with pytest.raises(TypeError):
        run_io.read_json(path, 10)


# ---- write_json_atomic ----------------------------------------------------------------------------------------------------------------
def test_the_returned_hash_is_the_sha256_of_the_bytes_on_disk(tmp_path):
    for name, obj in {'plain': {'b': 1, 'a': [1, 2.5, None, True]}, 'unicode': {'이름': '값 ünï', 'z': {'y': 'x'}},
                      'empty': {}, 'floats': {'f': [0.1, 1e-300, 2.5e22, -0.0]}, 'nested': {'k': {'l': {'m': ['n', {'o': 1}]}}}}.items():
        path = tmp_path / f'{name}.json'
        digest = run_io.write_json_atomic(path, obj)
        assert digest == sha(path.read_bytes()), name
        assert digest == run_io.sha_file(path), name
        assert b'\r' not in path.read_bytes(), name                # bytes are written as they are hashed: no newline translation


def test_the_bytes_are_utf8_with_sorted_keys_two_space_indent_and_a_final_newline(tmp_path):
    obj = {'b': 1, 'a': {'d': [1, 2], 'c': '값'}}
    path = tmp_path / 'fmt.json'
    run_io.write_json_atomic(path, obj)
    expected = (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    assert path.read_bytes() == expected
    assert path.read_bytes().endswith(b'}\n')


def test_a_written_file_reads_back_to_the_same_object(tmp_path):
    obj = {'x': [1, 2.25, {'y': None}], 'name': '값', 'flag': False}
    path = tmp_path / 'round.json'
    run_io.write_json_atomic(path, obj)
    assert run_io.read_json(path) == obj


def test_writing_twice_gives_identical_bytes_and_hash(tmp_path):
    obj = {'b': [3, 2, 1], 'a': {'y': 1, 'x': 2}}
    first = run_io.write_json_atomic(tmp_path / 'one.json', obj)
    second = run_io.write_json_atomic(tmp_path / 'two.json', dict(reversed(list(obj.items()))))      # a different key order
    assert first == second and (tmp_path / 'one.json').read_bytes() == (tmp_path / 'two.json').read_bytes()


def test_a_write_replaces_an_existing_file_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / 'state.json'
    run_io.write_json_atomic(path, {'v': 1})
    digest = run_io.write_json_atomic(path, {'v': 2})
    assert run_io.read_json(path) == {'v': 2} and digest == sha(path.read_bytes())
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state.json']


def test_a_write_makes_the_missing_folders(tmp_path):
    path = tmp_path / 'a' / 'b' / 'state.json'
    run_io.write_json_atomic(path, {'v': 1})
    assert run_io.read_json(path) == {'v': 1}


def test_the_temporary_file_is_in_the_same_folder_is_complete_and_synced_before_the_rename(tmp_path, monkeypatch):
    """Order of events: write, flush, fsync of the temporary file, then the rename; the temporary file holds all the bytes at that point."""
    path = tmp_path / 'sub' / 'state.json'
    path.parent.mkdir()
    events, seen = [], {}
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append('fsync')
        seen['size_at_fsync'] = os.fstat(fd).st_size                # the bytes that the operating system has when fsync is called
        return real_fsync(fd)

    def replace(src, dst):
        events.append('replace')
        seen['src'], seen['dst'] = Path(src), Path(dst)
        seen['bytes'] = Path(src).read_bytes()                      # what is on disk when the rename happens
        return real_replace(src, dst)
    monkeypatch.setattr(os, 'fsync', fsync)
    monkeypatch.setattr(os, 'replace', replace)
    digest = run_io.write_json_atomic(path, {'v': 1})
    assert events == ['fsync', 'replace']
    assert seen['src'].parent == seen['dst'].parent == path.parent and seen['src'] != seen['dst']
    assert seen['dst'] == path and seen['bytes'] == path.read_bytes() and digest == sha(seen['bytes'])
    # flush comes before fsync (review M3): a small document stays in the 8 KiB buffer of the file object until flush, so without the flush
    # the file would still be empty when fsync is called, and fsync would sync nothing.
    assert seen['size_at_fsync'] == len(seen['bytes']) > 0


@pytest.mark.parametrize('stop', ['replace', 'fsync'])
@pytest.mark.parametrize('error', [OSError('disk full'), KeyboardInterrupt()])
def test_an_interrupted_write_leaves_the_old_file_intact_and_no_temporary_file(tmp_path, monkeypatch, stop, error):
    """The writer stops before the rename (the rename itself, or the sync before it): the target is the old file, byte for byte."""
    path = tmp_path / 'state.json'
    old = run_io.write_json_atomic(path, {'v': 'old', 'keep': [1, 2, 3]})
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(os, stop, fail)
    with pytest.raises(type(error)):
        run_io.write_json_atomic(path, {'v': 'new'})
    monkeypatch.undo()
    assert path.read_bytes() == before and run_io.sha_file(path) == old
    assert run_io.read_json(path) == {'v': 'old', 'keep': [1, 2, 3]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state.json']          # the temporary file was removed


def test_an_interrupted_first_write_leaves_no_file_at_all(tmp_path, monkeypatch):
    path = tmp_path / 'new.json'

    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError):
        run_io.write_json_atomic(path, {'v': 1})
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('bad', [{'a': math.nan}, {'a': math.inf}, {'a': -math.inf}, {'a': [1, {'b': math.nan}]}, {'a': [math.inf]}])
def test_write_json_atomic_refuses_a_non_finite_value_and_touches_nothing(tmp_path, bad):
    path = tmp_path / 'state.json'
    run_io.write_json_atomic(path, {'v': 'old'})
    before = path.read_bytes()
    with pytest.raises(run_io.IORefused, match='non-?finite|JSON'):
        run_io.write_json_atomic(path, bad)
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ['state.json']
    fresh = tmp_path / 'fresh.json'
    with pytest.raises(run_io.IORefused):
        run_io.write_json_atomic(fresh, bad)
    assert not fresh.exists() and sorted(p.name for p in tmp_path.iterdir()) == ['state.json']


@pytest.mark.parametrize('bad', [{'a': {1, 2}}, {'a': object()}, {'a': b'bytes'}, {(1, 2): 'tuple key'}, {'a': '\ud800'}])
def test_write_json_atomic_refuses_what_is_not_json_and_touches_nothing(tmp_path, bad):
    """A set, an object, bytes, a key that is not a string or number, and text that is not UTF-8 encodable (a lone surrogate)."""
    path = tmp_path / 'state.json'
    run_io.write_json_atomic(path, {'v': 'old'})
    before = path.read_bytes()
    with pytest.raises(run_io.IORefused) as info:
        run_io.write_json_atomic(path, bad)
    assert info.value.__context__ is None and info.value.__cause__ is None      # nothing of the object hangs on the refusal
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ['state.json']


# ---- the writer writes only what the reader gives back (review I1) -------------------------------------------------------------------------
# Invariant of the pair: for every object that write_json_atomic accepts, read_json(path) is an equal object (a tuple comes back as a list).
# So the writer refuses before it touches the disk: a top level that is not an object, a key that is not a string at any depth (JSON turns
# 20261022, 1.5, True and None into the text of the key, which reads back as a different object), and output over the reader's default bound.
def over_the_cap():
    """An object whose output is one byte over the 16 MiB bound of ``read_json``, and the one that is exactly at the bound."""
    overhead = len((json.dumps({'a': ''}, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    return {'a': 'x' * ((16 << 20) - overhead + 1)}, {'a': 'x' * ((16 << 20) - overhead)}


@pytest.mark.parametrize('bad', [[1, 2, 3], [], [{'a': 1}], 'text', 3, 2.5, None, True, (1, 2),
                                 {1: 'a'}, {20261022: {'seed': 1}}, {1.5: 'a'}, {True: 'a'}, {None: 'a'}, {'a': {2: 'b'}},
                                 {'a': [{'b': {True: 1}}]}, {'a': ({'b': {1: 1}},)}, {'a': 1, 2: 'b'}, {(1, 2): 'tuple key'}],
                         ids=lambda bad: repr(bad)[:40])
def test_write_json_atomic_refuses_a_top_level_that_is_not_an_object_and_a_key_that_is_not_a_string(tmp_path, bad):
    """Each refusal comes before the disk: no file, no temporary file, not even the missing folders; an old file stays as it was."""
    old = tmp_path / 'old.json'
    run_io.write_json_atomic(old, {'v': 'old'})
    before = old.read_bytes()
    with pytest.raises(run_io.IORefused) as info:
        run_io.write_json_atomic(old, bad)
    assert info.value.__context__ is None and info.value.__cause__ is None
    assert old.read_bytes() == before
    fresh = tmp_path / 'newfolder' / 'deeper' / 'fresh.json'
    with pytest.raises(run_io.IORefused):
        run_io.write_json_atomic(fresh, bad)
    assert not fresh.exists() and not (tmp_path / 'newfolder').exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['old.json']


def test_a_refusal_does_not_print_a_key_or_a_value(tmp_path):
    with pytest.raises(run_io.IORefused) as info:
        run_io.write_json_atomic(tmp_path / 'a.json', {'ok': {'SECRET-VALUE-1': {20261022: 'SECRET-VALUE-2'}}})
    assert 'SECRET' not in str(info.value) and '20261022' not in str(info.value)


def test_write_json_atomic_refuses_output_over_the_default_bound_of_read_json(tmp_path):
    over, exact = over_the_cap()
    path = tmp_path / 'sub' / 'big.json'
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.write_json_atomic(path, over)
    assert not path.exists() and not path.parent.exists()
    digest = run_io.write_json_atomic(path, exact)                    # exactly 16 MiB is allowed, as the reader allows it
    assert path.stat().st_size == 16 << 20 and digest == run_io.sha_file(path)
    assert run_io.read_json(path) == exact
    assert sorted(p.name for p in path.parent.iterdir()) == ['big.json']


ROUND_TRIPS = [
    ({'a': 1, 'b': [1, 2.5, None, True, False, 'x'], 'c': {'d': {'e': []}}}, None),
    ({'이름': '값 ünï ✓', 'z': {'y': '\n\t"quote" \\ slash'}}, None),
    ({'seeds': {'20261022': 1, '20261023': 2}, '1': 'a', '10': 'b'}, None),                     # numbers as text keys are strings and stay
    ({'f': [0.1, 1e-300, 1.7976931348623157e308, -0.0, 2.5e22, 1 / 3]}, None),
    ({'big': 2 ** 70, 'neg': -(2 ** 70), 'zero': 0}, None),
    ({}, None),
    ({'t': (1, 2, (3, {'u': (4,)}))}, {'t': [1, 2, [3, {'u': [4]}]]}),                           # a tuple is written as a list and read back as one
]


@pytest.mark.parametrize('obj,expected', ROUND_TRIPS)
def test_write_then_read_returns_an_equal_object(tmp_path, obj, expected):
    path = tmp_path / 'deep' / 'round.json'
    digest = run_io.write_json_atomic(path, obj)
    back = run_io.read_json(path)
    assert back == (obj if expected is None else expected)
    assert digest == sha(path.read_bytes()) == run_io.sha_file(path)
    assert run_io.write_json_atomic(tmp_path / 'again.json', back) == digest      # and writing what was read gives the same bytes


def test_the_refusals_of_the_pair_are_symmetric_for_keys_and_top_level(tmp_path):
    """What the writer refuses for its shape, the reader refuses too when it finds that shape in a file."""
    for text in ('[1, 2, 3]', '"text"', '3', 'null'):
        with pytest.raises(run_io.IORefused):
            run_io.read_json(put(tmp_path / 'top.json', text))
    for bad in ([1, 2, 3], 'text', 3, None):
        with pytest.raises(run_io.IORefused):
            run_io.write_json_atomic(tmp_path / 'top2.json', bad)
    assert not (tmp_path / 'top2.json').exists()


def test_the_bytes_follow_the_recipes_of_runtime_atomic_json_and_run_s4a_write(tmp_path):
    """The recipe of ``evaluation.runtime.atomic_json`` (indent 2, sorted keys, ensure_ascii False, final newline), written as bytes."""
    obj = {'z': 1, 'a': {'b': [1, 2, {'c': None}]}, 'name': '값'}
    path = tmp_path / 'a.json'
    run_io.write_json_atomic(path, obj)
    recipe = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    assert path.read_bytes() == recipe.encode('utf-8')
    ascii_obj = {'z': 1, 'a': {'b': [1, 2, {'c': None}]}}
    run_io.write_json_atomic(path, ascii_obj)
    assert path.read_bytes() == (json.dumps(ascii_obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('ascii')    # run_s4a._write


# ---- sha_file ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('size', [0, 1, 1024 * 1024 - 1, 1024 * 1024, 1024 * 1024 + 1, 2 * 1024 * 1024 + 12345])
def test_sha_file_equals_hashlib_for_empty_small_and_multi_chunk_files(tmp_path, size):
    data = bytes((i * 7 + i // 251) % 256 for i in range(min(size, 4096))) * (size // 4096 + 1)
    data = data[:size]
    path = tmp_path / 'blob.bin'
    path.write_bytes(data)
    assert run_io.sha_file(path) == sha(data) and len(run_io.sha_file(path)) == 64


def test_sha_file_agrees_with_the_comparison_helper(tmp_path):
    from c_ept.comparison import protocol
    path = tmp_path / 'blob.bin'
    path.write_bytes(random.Random(20261026).randbytes(3 * 1024 * 1024 + 5))
    assert run_io.sha_file(path) == protocol.sha_file(path)


def test_sha_file_calls_check_before_between_and_after_the_chunks(tmp_path):
    """The comparison code calls ``check`` once before the first chunk, once per chunk and once after the last: a 2.5 MiB file is 5 calls."""
    path = tmp_path / 'blob.bin'
    path.write_bytes(b'a' * (2 * 1024 * 1024 + 512 * 1024))
    calls = []
    run_io.sha_file(path, check=lambda: calls.append(1))
    assert len(calls) == 1 + 3 + 1
    empty = tmp_path / 'empty.bin'
    empty.write_bytes(b'')
    calls.clear()
    run_io.sha_file(empty, check=lambda: calls.append(1))
    assert len(calls) == 2


def test_an_error_raised_by_check_stops_the_hashing(tmp_path):
    path = tmp_path / 'blob.bin'
    path.write_bytes(b'a' * (3 * 1024 * 1024))
    calls = []

    class Stop(Exception):
        pass

    def check():
        calls.append(1)
        if len(calls) == 3:
            raise Stop

    with pytest.raises(Stop):
        run_io.sha_file(path, check=check)
    assert len(calls) == 3                                           # no call after the one that raised


def test_sha_file_of_a_missing_file_raises_the_os_error(tmp_path):
    with pytest.raises(OSError):
        run_io.sha_file(tmp_path / 'missing.bin')


def test_check_is_keyword_only(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'x')
    with pytest.raises(TypeError):
        run_io.sha_file(path, lambda: None)


# ---- read_json_with_sha: one read, the object and the hash of the same bytes (S4b-prep K5a; K3 re-check n3, A->B->A) ------------------------
# The profile loader used to read a file three or four times (parse, carriage-return check, hash, parse again) and accepted a file that was
# changed and changed back between the reads with the hash of one content and the settings of another. This reader reads the bytes once and
# gives back the parsed object and the SHA-256 of those same bytes; ``check`` lets a caller test the same bytes (the profile's carriage return).
def watch_opens(monkeypatch, target, after_close=None, reads=None):
    """Count the opens of ``target`` for reading only (through ``io.open``, which ``Path.open`` calls, and ``builtins.open``).

    ``after_close(n)``: called with the number of the open (1, 2, ...) when the stream of that open is closed, to change the file at the
    moment when a reader has finished reading it. ``reads``: a list that gets the number of bytes that each ``read`` call returned.
    Returns the list of opens.
    """
    import builtins
    import io as stdlib_io
    opened, real = [], stdlib_io.open

    class Proxy:
        def __init__(self, stream, number):
            self.stream, self.number = stream, number

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.stream.close()
            if after_close is not None:
                after_close(self.number)

        def read(self, *arguments):
            data = self.stream.read(*arguments)
            if reads is not None:
                reads.append(len(data))
            return data

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def spy(file, mode='r', *args, **kwargs):
        stream = real(file, mode, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and 'r' in mode and not set('wax+') & set(mode) and Path(file) == target:     # reads only
            opened.append(mode)
            return Proxy(stream, len(opened))
        return stream
    monkeypatch.setattr(stdlib_io, 'open', spy)
    monkeypatch.setattr(builtins, 'open', spy)
    return opened


def test_read_json_with_sha_has_the_planned_signature():
    signature = inspect.signature(run_io.read_json_with_sha)
    assert list(signature.parameters) == ['path', 'max_bytes', 'check']
    assert signature.parameters['max_bytes'].kind is inspect.Parameter.KEYWORD_ONLY and signature.parameters['max_bytes'].default == 16 << 20
    assert signature.parameters['check'].kind is inspect.Parameter.KEYWORD_ONLY and signature.parameters['check'].default is None


def test_read_json_with_sha_returns_the_object_and_the_hash_of_the_same_bytes(tmp_path):
    doc = {'a': 1, 'b': [1.5, None, True, 'x'], 'c': {'d': {'e': []}}, '한글': '값'}
    path = put(tmp_path / 'doc.json', json.dumps(doc, ensure_ascii=False))
    value, digest = run_io.read_json_with_sha(path)
    assert value == doc == run_io.read_json(path)
    assert digest == sha(path.read_bytes()) == run_io.sha_file(path) and len(digest) == 64
    assert run_io.read_json_with_sha(str(path)) == (doc, digest)                    # a str path is as good as a Path
    written = tmp_path / 'written.json'
    returned = run_io.write_json_atomic(written, doc)
    assert run_io.read_json_with_sha(written) == (doc, returned)                    # the writer's hash, the file's hash and the reader's agree


def test_read_json_with_sha_hashes_the_bytes_as_they_are_crlf_and_all(tmp_path):
    """The hash is that of the file, not of a normal form: CRLF stays CRLF (the reader accepts it, ``check`` may refuse it)."""
    path = put(tmp_path / 'crlf.json', '{\r\n  "a": 1\r\n}\r\n')
    assert run_io.read_json_with_sha(path) == ({'a': 1}, sha(b'{\r\n  "a": 1\r\n}\r\n'))
    exact = put(tmp_path / 'exact.json', '{"a": 1}')
    assert run_io.read_json_with_sha(exact, max_bytes=8) == ({'a': 1}, sha(b'{"a": 1}'))          # the bound is inclusive


REFUSED_BYTES = [
    b'{"a": 1, "a": 2}', b'{"a": {"b": 1, "b": 2}}', b'{"a": [{"b": 1, "b": 1}]}',
    b'{"a": NaN}', b'{"a": Infinity}', b'{"a": -Infinity}', b'{"a": 1e999}', b'{"a": [0.5, -1E400]}',
    b'[]', b'[{"a": 1}]', b'3', b'null', b'"text"',
    b'', b'   \n', b'{"a": ', b'{"a": 1,}', b"{'a': 1}", b'{"a": 1} trailing', b'\xef\xbb\xbf{"a": 1}', b'{"a": "\xff"}', b'\xff\xfe{\x00}\x00',
    b'{"a":' * 100000 + b'1' + b'}' * 100000, b'[' * 100000,
]


@pytest.mark.parametrize('payload', REFUSED_BYTES, ids=range(len(REFUSED_BYTES)))
def test_read_json_with_sha_refuses_exactly_what_read_json_refuses(tmp_path, payload):
    """Same checks, same messages: the new reader is ``read_json`` plus the hash, not a second set of rules."""
    path = tmp_path / 'bad.json'
    path.write_bytes(payload)
    with pytest.raises(run_io.IORefused) as old:
        run_io.read_json(path)
    with pytest.raises(run_io.IORefused) as new:
        run_io.read_json_with_sha(path)
    assert type(new.value) is type(old.value) and str(new.value) == str(old.value)
    assert new.value.__cause__ is None and new.value.__context__ is None


def test_read_json_with_sha_refuses_a_missing_file_a_folder_a_link_and_a_file_over_the_bound(tmp_path, monkeypatch):
    for name in ('missing.json', 'folder.json'):
        (tmp_path / 'folder.json').mkdir(exist_ok=True)
        with pytest.raises(run_io.IORefused, match='regular'):
            run_io.read_json_with_sha(tmp_path / name)
    target = put(tmp_path / 'target.json', '{"a": 1}')
    link = put(tmp_path / 'link.json', '{"a": 1}')
    real = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda self: True if self == link else real(self))
    assert run_io.read_json_with_sha(target)[0] == {'a': 1}
    with pytest.raises(run_io.IORefused, match='link'):
        run_io.read_json_with_sha(link)
    monkeypatch.undo()
    big = put(tmp_path / 'big.json', '{"a": "' + 'x' * 100 + '"}')
    size = big.stat().st_size
    assert run_io.read_json_with_sha(big, max_bytes=size)[0] == {'a': 'x' * 100}
    for limit in (size - 1, 10):
        with pytest.raises(run_io.IORefused, match='large'):
            run_io.read_json_with_sha(big, max_bytes=limit)
    with pytest.raises(TypeError):
        run_io.read_json_with_sha(big, 10)                                           # max_bytes is keyword-only, as in read_json


def test_read_json_with_sha_checks_the_size_before_the_parse_and_the_hash(tmp_path, monkeypatch):
    path = put(tmp_path / 'big.json', '{"a": NaN, "a": 1e999, "pad": "' + 'x' * 100 + '"}')
    size = path.stat().st_size

    def forbidden(*args, **kwargs):
        raise AssertionError('called for an oversize file')
    monkeypatch.setattr(run_io.json, 'loads', forbidden)
    monkeypatch.setattr(run_io.hashlib, 'sha256', forbidden)
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json_with_sha(path, max_bytes=size - 1)


def test_at_most_max_bytes_plus_one_bytes_are_read_from_an_oversize_file(tmp_path, monkeypatch):
    """Both readers refuse a file of more than ``max_bytes`` bytes after reading ``max_bytes + 1`` of them, not after reading all of it."""
    path = put(tmp_path / 'big.json', '{"a": "' + 'x' * 5000 + '"}')
    for reader in (run_io.read_json, run_io.read_json_with_sha):
        reads = []
        opened = watch_opens(monkeypatch, path, reads=reads)
        with pytest.raises(run_io.IORefused, match='large'):
            reader(path, max_bytes=100)
        monkeypatch.undo()
        assert opened == ['rb'] and sum(reads) == 101, (reader.__name__, reads)


def test_read_json_with_sha_opens_the_file_once(tmp_path, monkeypatch):
    path = put(tmp_path / 'once.json', '{"a": 1}')
    opened = watch_opens(monkeypatch, path)
    assert run_io.read_json_with_sha(path) == ({'a': 1}, sha(b'{"a": 1}'))
    assert opened == ['rb']                                                          # one open, for reading bytes: no second read for the hash
    opened.clear()
    assert run_io.read_json(path) == {'a': 1}
    assert opened == ['rb']                                                          # read_json keeps its single read


def test_the_hash_is_that_of_the_parsed_bytes_even_when_the_file_changes_after_the_read(tmp_path, monkeypatch):
    """A change right after the read (a rewrite, then another rewrite back) cannot split the object from the hash: there is no second read."""
    path = put(tmp_path / 'race.json', '{"epochs": 30}')
    first = path.read_bytes()
    changed = []

    def rewrite(number):
        changed.append(number)
        path.write_bytes(b'{"epochs": 31}' if number == 1 else first)               # B after the first read; back to A after a second one
    watch_opens(monkeypatch, path, after_close=rewrite)
    value, digest = run_io.read_json_with_sha(path)
    monkeypatch.undo()                                                               # the reads below are the test's own, not the reader's
    assert changed == [1]                                                          # the reader closed the file once; it never read it again
    assert value == {'epochs': 30} and digest == sha(first) != sha(b'{"epochs": 31}')
    assert path.read_bytes() == b'{"epochs": 31}'                                    # what is on disk now is another content; the pair is still consistent


def test_check_receives_the_parsed_bytes_once_after_the_parse(tmp_path):
    path = put(tmp_path / 'c.json', '{"a": 1}\n')
    seen = []
    value, digest = run_io.read_json_with_sha(path, check=seen.append)
    assert seen == [b'{"a": 1}\n'] and value == {'a': 1} and digest == sha(seen[0])
    seen.clear()
    for payload in (b'{"a": 1, "a": 2}', b'[1]', b'{"a": '):                         # a refusal of the strict reader comes first: check is not called
        path.write_bytes(payload)
        with pytest.raises(run_io.IORefused):
            run_io.read_json_with_sha(path, check=seen.append)
    assert seen == []
    path.write_bytes(b'{"pad": "' + b'x' * 50 + b'"}')
    with pytest.raises(run_io.IORefused, match='large'):
        run_io.read_json_with_sha(path, max_bytes=10, check=seen.append)
    assert seen == []


# ---- parse_json_object and canonical_sha256 (S4b-prep K5a review fix I1: the structural decision-record rule needs both) ---------------------
def test_parse_json_object_is_the_strict_parse_of_bytes_without_a_file():
    assert list(inspect.signature(run_io.parse_json_object).parameters) == ['raw']
    assert run_io.parse_json_object(b'{"a": [1, 2.5, null], "b": {"c": "\xed\x95\x9c"}}') == {'a': [1, 2.5, None], 'b': {'c': '한'}}
    assert run_io.parse_json_object(bytearray(b'{}')) == {}
    for payload in REFUSED_BYTES:                                                     # the same refusals, with the same messages, as the file readers
        with pytest.raises(run_io.IORefused) as bytes_refused:
            run_io.parse_json_object(payload)
        assert bytes_refused.value.__context__ is None and bytes_refused.value.__cause__ is None
    with pytest.raises(TypeError):
        run_io.parse_json_object('{}')


def test_parse_json_object_gives_the_messages_of_the_file_readers(tmp_path):
    for payload in REFUSED_BYTES[:24]:
        path = tmp_path / 'bad.json'
        path.write_bytes(payload)
        with pytest.raises(run_io.IORefused) as from_file:
            run_io.read_json(path)
        with pytest.raises(run_io.IORefused) as from_bytes:
            run_io.parse_json_object(payload)
        assert str(from_file.value) == str(from_bytes.value)


def test_canonical_sha256_is_the_hash_of_sorted_compact_utf8_json():
    """Canonical form: ``json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)`` encoded as UTF-8."""
    value = {'b': 1, 'a': [1, {'c': '한', 'a': None}], 'z': 2.5, 't': True}
    expected = '{"a":[1,{"a":null,"c":"한"}],"b":1,"t":true,"z":2.5}'.encode('utf-8')
    assert run_io.canonical_sha256(value) == sha(expected)
    assert run_io.canonical_sha256({'a': 1, 'b': 2}) == run_io.canonical_sha256({'b': 2, 'a': 1})              # key order does not matter
    assert run_io.canonical_sha256({'a': (1, 2)}) == run_io.canonical_sha256({'a': [1, 2]})                     # a tuple is a list
    assert run_io.canonical_sha256({'a': 1}) != run_io.canonical_sha256({'a': 2}) != run_io.canonical_sha256({'a': '2'})
    assert run_io.canonical_sha256({'a': 1}) != run_io.canonical_sha256({'a': 1.0})                              # 1 and 1.0 are different JSON texts
    assert run_io.canonical_sha256([]) == sha(b'[]') and run_io.canonical_sha256('x') == sha(b'"x"') and run_io.canonical_sha256(None) == sha(b'null')


def test_the_canonical_hash_does_not_depend_on_the_whitespace_of_the_file_it_came_from():
    compact = b'{"a":{"b":[1,2],"c":"x"},"d":3}'
    spaced = b'{\n  "d": 3,\n  "a": {\n    "c": "x",\n    "b": [ 1, 2 ]\n  }\n}\n'
    assert run_io.canonical_sha256(run_io.parse_json_object(compact)) == run_io.canonical_sha256(run_io.parse_json_object(spaced))


@pytest.mark.parametrize('bad', [math.nan, math.inf, {'a': math.nan}, {'a': {1, 2}}, {'a': object()}, {1: 'a'}, {'a': {2: 'b'}}, {(1, 2): 'a'}, '\ud800',
                                 {'a': b'bytes'}])
def test_canonical_sha256_refuses_what_is_not_strict_json_and_keys_that_are_not_text(bad):
    with pytest.raises(run_io.IORefused) as info:
        run_io.canonical_sha256(bad)
    assert info.value.__context__ is None and info.value.__cause__ is None


def test_an_error_raised_by_check_propagates_unchanged_and_with_no_context(tmp_path):
    path = put(tmp_path / 'c.json', '{"a": 1}\r\n')

    class Refuse(run_io.IORefused):
        pass

    def check(raw):
        if b'\r' in raw:
            raise Refuse('a carriage return')
    with pytest.raises(Refuse, match='carriage return') as info:
        run_io.read_json_with_sha(path, check=check)
    assert info.value.__context__ is None and info.value.__cause__ is None
    path.write_bytes(b'{"a": 1}\n')
    assert run_io.read_json_with_sha(path, check=check)[0] == {'a': 1}
