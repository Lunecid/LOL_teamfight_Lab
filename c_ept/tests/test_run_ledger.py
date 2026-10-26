"""Tests of the run ledger of the run kernel (``c_ept.run.ledger``, S4b-prep K5a, decisions D11 and A5-F7).

The earlier stacks keep a ledger of stages in different ways. ``run_s3.Ledger`` and ``run_s4a`` write the stage JSON first, then ``RUN.json``,
then ``ledger.finish`` (review A5-F7: a crash between those writes made the run unresumable, although every file was intact and verifiable by
hash, and none of the refusals was tested). The T009 stack writes a stage receipt (``comparison.runtime._write_receipt``) whose artifact hashes
are re-checked, and keeps a single-owner campaign ledger (``comparison.infrastructure.CampaignLedger``) that refuses a file changed behind its
back. The kernel ledger keeps what each of them proved: stages begin and finish in order (S3), a stage JSON lists input and output hashes that are
re-verified (T009), the ledger file belongs to one owner (T009), and finalisation is idempotent (A5-F7, decision D11): when a stage JSON exists on
resume, its hashes are re-verified and the ledger is completed; any mismatch stays fail-closed.

Layout under the run folder: ``HEADER.json`` (written once), ``LEDGER.json`` (the stages, in order, with their state) and ``stages/<stage>.json``.
Write order of one stage: begin = ledger (state ``begun``, input hashes); finish = stage JSON, then ledger (state ``finished``). A crash between
any two writes of a stage is recoverable, and these tests stop the writer at every write, before and after it takes effect. Creation (header, then
ledger) is the one exception: a stop between its two writes leaves a header without a ledger, which is refused as "creation incomplete".

The tests use synthetic files in ``tmp_path`` only. Nothing here starts a process, uses the GPU or reads a research file.

해석: 원장은 "어느 단계까지 끝났는가"를 해시로 증명하는 기록이다. 단계 JSON을 쓴 뒤 원장을 완성하기 전에 멈추어도, 입력과 출력 해시를 다시 확인해
원장을 마저 완성한다(A5-F7). 확인에서 하나라도 다르면 변조로 보고 거부한다(Review Focus 4). 그래서 시험은 쓰기 지점마다 멈추는 시뮬레이션과, 변조 사례마다
거부를 확인한다. 메시지에는 단계 이름과 개수만 실리고 파일 경로나 내용은 실리지 않는다.
"""
import ast
import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from c_ept.run import guards, ledger as run_ledger
from c_ept.run import io as run_io
from c_ept.run.identity import Identity
from c_ept.run.ledger import Header, Ledger, LedgerRefused, Tampered
from c_ept.tests.test_run_io import watch_opens           # counts the reads of one file and the bytes each read returned

REPO = Path(__file__).resolve().parents[2]
LEDGER_SOURCE = Path(run_ledger.__file__)
LEAK = 'ZZ-LEAK-4417'                       # a marker that must never be echoed by a refusal
STDLIB = set(sys.stdlib_module_names)
PROFILE = b'{\n  "kind": "train"\n}\n'
STAGES = ('train', 'evaluate')


def spec_doc(point=None) -> dict:
    """A synthetic decision record in the shape of the real one: one JSON object with ``operating_point``, other top-level members, and
    ``decisions`` as an object of named blocks (``run_s3.read_decisions`` reads ``doc['decisions']['S3_prices']['prices']['sha256']``).
    A new decision is a new member of ``decisions``; the block names are the decision names."""
    return {'operating_point': point or {'G_ms': 1, 'L_u': 2, 'tau_lead_ms': 3, 'presence_gate': None, 'size_classes': {'T': 1, 'S': 2, 'ASYM': 3, 'P': 4}},
            'data': {'source': 'synthetic', 'matches': 3}, 'held_out_policy': {'role': 'TEST', 'opened': False},
            'decisions': {'C9': {'chosen': 'rule', 'why': '한글 이유'}, 'S2_gate': {'S2G-1': {'cap': 128}, 'S2G-4': {'rule': 'drop columns'}},
                          'S3_prices': {'prices': {'sha256': 'ab' * 32}, 'rules': {'sha256': 'cd' * 32}}},
            'G2_time_base': {'unit': 's'}}


def dump_spec(doc: dict, indent=2) -> bytes:
    return json.dumps(doc, indent=indent, ensure_ascii=False).encode('utf-8')


SPEC = dump_spec(spec_doc())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_header(**changes) -> Header:
    values = dict(identity=Identity(commit='ab' * 20, files={'c_ept/run/io.py': 'cd' * 32, 'c_ept/run/ledger.py': 'ef' * 32}, protected=('c_ept.cohort',)),
                  profile_sha256='12' * 32, runner={'batch_size': 32, 'device': 'cpu', 'effective': {'M': [1, 2]}}, **run_ledger.spec_fingerprint(SPEC))
    values.update(changes)
    return Header(**values)


class World:
    """A run folder with two stages. ``train`` reads a profile outside the run folder and makes two files; ``evaluate`` reads one of them.

    The files of a stage have fixed bytes, so that the work of a stage can be repeated after a crash and gives the same files.
    """
    OUTPUTS = {'train': ['train/best.bin', 'train/log.txt'], 'evaluate': ['evaluate/scores.bin']}

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.run = tmp_path / 'run'
        self.profile = put(tmp_path / 'profile.json', PROFILE)

    def inputs(self, stage):
        return [self.profile] if stage == 'train' else ['train/best.bin']

    def outputs(self, stage):
        return list(self.OUTPUTS[stage])

    def work(self, stage):
        if stage == 'train':
            put(self.run / 'train' / 'best.bin', b'weights-v1|' * 20)
            put(self.run / 'train' / 'log.txt', b'epoch 1 loss 3.5\nepoch 2 loss 3.1\n')
        else:
            put(self.run / 'evaluate' / 'scores.bin', b'scores-v1|' * 8)

    def play(self, ledger, stage):
        ledger.begin(stage, inputs=self.inputs(stage))
        self.work(stage)
        return ledger.finish(stage, self.outputs(stage))

    def complete(self, header=None) -> Ledger:
        ledger = Ledger.create(self.run, header or make_header())
        for stage in STAGES:
            self.play(ledger, stage)
        return ledger

    def continue_run(self, ledger):
        """What a restarted runner does: resume, then run every stage that is not complete (begin again if it was begun)."""
        done = ledger.resume()
        for stage in STAGES:
            if stage not in done:
                self.play(ledger, stage)
        return done

    def snapshot(self):
        """Every file under the run folder, by relative path, with its bytes."""
        return {p.relative_to(self.run).as_posix(): p.read_bytes() for p in sorted(self.run.rglob('*')) if p.is_file()}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def edit_json(path: Path, change):
    """Read a JSON file, apply ``change(doc)``, write it back (the writer's format)."""
    doc = run_io.read_json(path)
    change(doc)
    run_io.write_json_atomic(path, doc)


class Crash(Exception):
    """The simulated stop of the process at a write."""


def stop_writer_at(monkeypatch, number: int, after: bool) -> list:
    """Let ``run_io.write_json_atomic`` raise ``Crash`` at its ``number``-th call: before the write takes effect, or (``after``) right after it.

    The atomic writer is all or nothing, so these two are the only ways a stop can leave a file: written, or not written.
    Returns the list of the paths that were written (completely) before the stop.
    """
    real = run_io.write_json_atomic
    written = []

    def writer(path, obj):
        if len(written) + 1 == number and not after:
            written.append(('stopped', Path(path).name))
            raise Crash
        digest = real(path, obj)
        written.append(Path(path).name)
        if len(written) == number and after:
            raise Crash
        return digest
    monkeypatch.setattr(run_io, 'write_json_atomic', writer)
    return written


def count_writes(monkeypatch) -> list:
    real = run_io.write_json_atomic
    written = []

    def writer(path, obj):
        written.append(Path(path).name)
        return real(path, obj)
    monkeypatch.setattr(run_io, 'write_json_atomic', writer)
    return written


# ---- the interface and the module's own discipline ------------------------------------------------------------------------------------
def test_the_interface_has_the_planned_names_and_signatures():
    assert list(inspect.signature(Ledger).parameters) == ['run_dir']
    create = inspect.signature(Ledger.create)
    assert list(create.parameters) == ['run_dir', 'header']
    begin = inspect.signature(Ledger.begin)
    assert list(begin.parameters) == ['self', 'stage', 'inputs']
    assert begin.parameters['inputs'].kind is inspect.Parameter.KEYWORD_ONLY and begin.parameters['inputs'].default == ()
    assert list(inspect.signature(Ledger.finish).parameters) == ['self', 'stage', 'outputs']
    assert list(inspect.signature(Ledger.resume).parameters) == ['self']
    assert list(inspect.signature(Ledger.check_spec).parameters) == ['self', 'current']
    growth = inspect.signature(run_ledger.check_spec_growth)
    assert list(growth.parameters) == ['current', 'top', 'blocks']                                                  # K5b R1: top replaces outside_sha256
    assert growth.parameters['top'].kind is growth.parameters['blocks'].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(run_ledger.spec_fingerprint).parameters) == ['record']
    assert list(inspect.signature(Ledger.stage).parameters) == ['self', 'name']
    assert [f.name for f in dataclasses.fields(Header)] == ['identity', 'spec_sha256', 'spec_top', 'spec_blocks', 'profile_sha256', 'runner']
    assert list(inspect.signature(Ledger.note).parameters) == ['self', 'stage', 'kind', 'record']                   # K5b: a place for stop reasons and E6 records
    assert not hasattr(Header, 'spec_outside_sha256')
    assert not hasattr(run_ledger, 'BYTE_PREFIX') and not hasattr(Header, 'spec_length')                           # the byte-prefix rule is gone (I1)


def test_the_errors_are_io_refusals_and_a_tamper_is_a_ledger_refusal():
    assert issubclass(Tampered, LedgerRefused) and issubclass(LedgerRefused, run_io.IORefused) and issubclass(LedgerRefused, ValueError)
    assert Tampered is not LedgerRefused


def test_the_file_formats_are_pinned():
    """The names and schemas are the format of files on disk: a run written today is read by a later version of the kernel."""
    assert (run_ledger.SCHEMA_HEADER, run_ledger.SCHEMA_LEDGER, run_ledger.SCHEMA_STAGE) == ('ept-run-header-v1', 'ept-run-ledger-v1', 'ept-run-stage-v1')
    assert (run_ledger.HEADER_FILE, run_ledger.LEDGER_FILE, run_ledger.STAGE_FOLDER) == ('HEADER.json', 'LEDGER.json', 'stages')
    assert run_ledger.STATES == ('begun', 'finished')


def test_the_module_imports_only_the_standard_library_and_public_kernel_modules():
    tree = ast.parse(LEDGER_SOURCE.read_text(encoding='utf-8'))
    allowed = {('c_ept.run', 'io'), ('c_ept.run', 'identity'), ('c_ept.run', 'guards')}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split('.')[0] in STDLIB for a in node.names), [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'a relative import'
            if (node.module or '').split('.')[0] not in STDLIB:
                assert {(node.module, a.name) for a in node.names} <= allowed, (node.module, [a.name for a in node.names])


def test_the_module_follows_the_kernel_boundary_rule_q8():
    """Q8/K7 for ``c_ept/run/``: no private (underscore) name or module from another package, and no command module."""
    commands = ('c_ept.workflows.run_', 'c_ept.workflows.s4a_operator', 'c_ept.workflows.training_operator', 'c_ept.comparison.operator', 'tools')
    for node in ast.walk(ast.parse(LEDGER_SOURCE.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            assert not any(part.startswith('_') and part != '__future__' for part in module.split('.')), module
            assert not any(a.name.startswith('_') for a in node.names if module != '__future__'), module
            assert not module.startswith(commands), module
            assert not any(f'{module}.{a.name}'.startswith(commands) for a in node.names), module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(part.startswith('_') for part in alias.name.split('.')), alias.name
                assert not alias.name.startswith(commands), alias.name


def test_all_json_goes_through_run_io():
    """The ledger reads with ``run.io`` (strict, read once) and writes with ``run.io`` (atomic): no ``json`` import, no direct write or rename."""
    tree = ast.parse(LEDGER_SOURCE.read_text(encoding='utf-8'))
    imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert 'json' not in imported and 'tempfile' not in imported and 'shutil' not in imported
    forbidden = {'write_text', 'write_bytes', 'replace', 'rename', 'fsync', 'unlink', 'rmdir', 'rmtree', 'remove', 'truncate'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
            assert name not in forbidden, name
            if name == 'open':                                    # a plain open() is only ever a read
                modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)] + [k.value.value for k in node.keywords
                                                                                              if k.arg == 'mode' and isinstance(k.value, ast.Constant)]
                assert not any(set('wax+') & set(str(m)) for m in modes)


def test_importing_the_module_loads_no_native_or_model_code():
    code = ('import sys; import c_ept.run.ledger; '
            'bad = sorted({"numpy", "torch", "scipy", "psutil", "c_ept.runlog", "c_ept.config", "c_ept.legacy", "c_ept.runtime"} & sys.modules.keys()); print(bad)')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '[]', done.stdout


# ---- the header: written once at creation ---------------------------------------------------------------------------------------------
def test_create_writes_the_header_once_and_a_second_create_refuses(world):
    header = make_header()
    ledger = Ledger.create(world.run, header)
    before = world.snapshot()
    assert sorted(before) == ['HEADER.json', 'LEDGER.json']
    with pytest.raises(LedgerRefused, match='already'):
        Ledger.create(world.run, make_header(profile_sha256='99' * 32))              # another header does not replace the first
    with pytest.raises(LedgerRefused, match='already'):
        Ledger.create(world.run, header)
    assert world.snapshot() == before                                               # not one byte changed
    assert ledger.header == header


def test_reopening_reads_the_same_header(world):
    header = make_header()
    created = Ledger.create(world.run, header)
    reopened = Ledger(world.run)
    assert reopened.header == created.header == header
    record = run_io.read_json(world.run / 'HEADER.json')
    assert record['identity'] == header.identity.to_record()
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    assert (record['spec_sha256'], record['spec_top'], record['spec_blocks'], record['profile_sha256']) == (
        sha(SPEC), fingerprint['spec_top'], fingerprint['spec_blocks'], '12' * 32)
    assert sorted(record['spec_top']) == ['G2_time_base', 'data', 'held_out_policy', 'operating_point'] and 'spec_outside_sha256' not in record
    assert sorted(record['spec_blocks']) == ['C9', 'S2_gate', 'S3_prices'] and 'spec_length' not in record
    assert record['runner'] == {'batch_size': 32, 'device': 'cpu', 'effective': {'M': [1, 2]}}
    assert Identity.from_record(record['identity']) == header.identity
    assert reopened.stages == () and reopened.resume() == []


def test_the_runner_constants_are_a_free_mapping_that_the_header_keeps_apart_from_its_caller(world):
    runner = {'batch_size': 32, 'nested': {'a': [1, 2]}}
    header = make_header(runner=runner)
    runner['nested']['a'].append(3)
    runner['batch_size'] = 64                                                       # a later change of the caller's mapping
    assert header.runner == {'batch_size': 32, 'nested': {'a': (1, 2)}}             # a list is frozen as a tuple
    assert Ledger.create(world.run, header).header == header
    assert make_header(runner={}).runner == {}


def test_the_runner_constants_are_read_only_at_every_depth(world):
    """K5a review M7: ``MappingProxyType`` froze only the top level; a nested write changed ``to_record()``. Now every depth is frozen."""
    header = make_header(runner={'batch_size': 32, 'nested': {'a': [1, 2], 'b': {'c': [3, {'d': 4}]}}})
    record = header.to_record()
    with pytest.raises(TypeError):
        header.runner['batch_size'] = 1
    with pytest.raises(TypeError):
        header.runner['nested']['x'] = 1
    with pytest.raises(TypeError):
        header.runner['nested']['b']['c'][1]['d'] = 5
    with pytest.raises((TypeError, AttributeError)):
        header.runner['nested']['a'].append(9)
    with pytest.raises(TypeError):
        header.runner['nested']['a'][0] = 9
    assert header.to_record() == record and header.to_record()['runner']['nested']['b']['c'] == [3, {'d': 4}]      # plain lists and dicts again
    record['runner']['nested']['a'].append(7)                                        # the record is a copy: changing it changes nothing
    assert header.to_record()['runner']['nested']['a'] == [1, 2] and header.runner['nested']['a'] == (1, 2)
    assert Ledger.create(world.run, header).header == header


def test_a_runner_mapping_that_contains_itself_or_is_absurdly_deep_is_refused(world):
    cyclic = {'a': 1}
    cyclic['self'] = cyclic
    deep = current = {}
    for _ in range(5000):
        current['x'] = {}
        current = current['x']
    for runner in (cyclic, deep):
        with pytest.raises(run_io.IORefused, match='nested too deeply') as info:
            make_header(runner=runner)
        assert info.value.__context__ is None and info.value.__cause__ is None
    assert not world.run.exists()


def test_create_refuses_when_only_one_of_the_two_creation_files_exists(world):
    """A stop between the two creation writes leaves the header alone; the folder is refused, not completed or overwritten."""
    world.run.mkdir()
    put(world.run / 'HEADER.json', b'{}\n')
    with pytest.raises(LedgerRefused, match='already'):
        Ledger.create(world.run, make_header())
    assert world.snapshot() == {'HEADER.json': b'{}\n'}
    (world.run / 'HEADER.json').unlink()
    put(world.run / 'LEDGER.json', b'{}\n')
    with pytest.raises(LedgerRefused, match='already'):
        Ledger.create(world.run, make_header())
    assert world.snapshot() == {'LEDGER.json': b'{}\n'}


@pytest.mark.parametrize('bad', [
    {'spec_sha256': 'AB' * 32}, {'spec_sha256': 'ab'}, {'spec_sha256': None}, {'spec_sha256': 'ab' * 32 + 'x'}, {'spec_sha256': 'ab' * 32 + '\n'},
    {'profile_sha256': 'g' * 64}, {'profile_sha256': 5}, {'profile_sha256': 'ab' * 32 + 'x'}, {'profile_sha256': '\n' + 'ab' * 32},
    {'spec_top': None}, {'spec_top': [('a', 'ab' * 32)]}, {'spec_top': 'text'}, {'spec_top': {'a': 'x'}}, {'spec_top': {'a': None}},
    {'spec_top': {'a': 'AB' * 32}}, {'spec_top': {'a': 'ab' * 32 + 'x'}}, {'spec_top': {1: 'ab' * 32}}, {'spec_top': {'decisions': 'ab' * 32}},
    {'spec_blocks': None}, {'spec_blocks': [('a', 'ab' * 32)]}, {'spec_blocks': 'text'}, {'spec_blocks': {'a': 'x'}}, {'spec_blocks': {'a': None}},
    {'spec_blocks': {'a': 'AB' * 32}}, {'spec_blocks': {'a': 'ab' * 32 + 'x'}}, {'spec_blocks': {1: 'ab' * 32}},
    {'identity': {'commit': 'ab' * 20}}, {'identity': None}, {'runner': [1, 2]}, {'runner': 'text'}, {'runner': None}, {'runner': {1: 'a'}},
    {'runner': {(1, 2): 'a'}},
])
def test_a_header_with_a_bad_field_is_refused_when_it_is_made_and_nothing_is_created(world, bad):
    with pytest.raises(run_io.IORefused):
        make_header(**bad)                                                          # the header itself refuses; no ledger is needed to find out
    assert not world.run.exists()                                                   # no folder, no file


def test_a_record_with_no_decision_blocks_and_an_empty_runner_are_accepted():
    header = make_header(spec_blocks={}, spec_top={}, runner={})
    assert dict(header.spec_blocks) == {} and dict(header.spec_top) == {} and header.runner == {}
    with pytest.raises(TypeError):
        header.spec_blocks['a'] = 'ab' * 32                                          # the block map is read-only too
    with pytest.raises(TypeError):
        header.spec_top['a'] = 'ab' * 32


def test_a_header_record_is_read_back_strictly():
    header = make_header()
    record = header.to_record()
    assert Header.from_record(record) == header
    assert set(record) == {'schema', 'identity', 'spec_sha256', 'spec_top', 'spec_blocks', 'profile_sha256', 'runner'}
    for broken in ({**record, 'extra': 1}, {k: v for k, v in record.items() if k != 'runner'}, {**record, 'schema': 'ept-run-header-v0'},
                   {**record, 'runner': [1]}, {**record, 'runner': None}, {**record, 'spec_sha256': 'x'}, {**record, 'identity': {'commit': 'x'}},
                   [record], None, 'text'):
        with pytest.raises(run_io.IORefused):
            Header.from_record(broken)
    assert Header.from_record({**record, 'runner': {'a': 1}}).runner == {'a': 1}


def test_a_runner_value_that_is_not_strict_json_is_refused_before_anything_is_created(world):
    for runner in ({'x': float('nan')}, {'x': object()}, {'x': {'y': {1, 2}}}):
        with pytest.raises(run_io.IORefused):
            Ledger.create(world.run, make_header(runner=runner))
        assert not world.run.exists()


def test_create_needs_a_header_object(world):
    with pytest.raises(LedgerRefused):
        Ledger.create(world.run, {'identity': 1})
    assert not world.run.exists()


def test_opening_a_folder_without_a_ledger_is_refused(world):
    with pytest.raises(LedgerRefused, match='no ledger'):
        Ledger(world.run)                                                           # no folder
    world.run.mkdir()
    with pytest.raises(LedgerRefused, match='no ledger'):
        Ledger(world.run)                                                           # an empty folder
    put(world.run / 'HEADER.json', b'{}\n')
    with pytest.raises(LedgerRefused, match='creation incomplete') as info:
        Ledger(world.run)                                                           # a header alone (a stop during creation)
    assert not isinstance(info.value, Tampered)                                     # a plain crash is not reported as a tamper (K5a review M1)


def test_an_edited_or_missing_header_is_refused_at_open_even_with_no_stage(world):
    Ledger.create(world.run, make_header())
    header = world.run / 'HEADER.json'
    original = header.read_bytes()
    edit_json(header, lambda doc: doc['runner'].update(device='cuda'))
    with pytest.raises(Tampered, match='header file does not match'):
        Ledger(world.run)
    put(header, original + b' ')                                                    # one byte more
    with pytest.raises(Tampered, match='header file does not match'):
        Ledger(world.run)
    header.unlink()
    with pytest.raises(Tampered, match='header file is missing'):
        Ledger(world.run)
    put(header, original)
    assert Ledger(world.run).header == make_header()


def test_a_deleted_ledger_is_a_tamper_when_the_run_has_a_stage_folder_and_an_incomplete_creation_when_not(world):
    """A header without a ledger is a stop during creation, unless the run already has stage files: then the ledger was deleted."""
    Ledger.create(world.run, make_header())
    (world.run / 'LEDGER.json').unlink()
    with pytest.raises(LedgerRefused, match='creation incomplete') as info:
        Ledger(world.run)
    assert not isinstance(info.value, Tampered) and 'start a new run folder' in str(info.value)
    (world.run / 'stages').mkdir()
    with pytest.raises(Tampered, match='ledger file is missing'):
        Ledger(world.run)


def test_a_header_that_is_malformed_is_refused_even_when_the_ledger_names_it(world):
    """The ledger names the header by its hash, so a header that is replaced and named consistently still has to be a header."""
    Ledger.create(world.run, make_header())
    for garbage in ({}, {'schema': 'ept-run-header-v1'}, {'schema': 'other', 'runner': {}}):
        digest = run_io.write_json_atomic(world.run / 'HEADER.json', garbage)
        edit_json(world.run / 'LEDGER.json', lambda doc: doc.update(header_sha256=digest))
        with pytest.raises(Tampered, match='header record is malformed') as info:
            Ledger(world.run)
        assert info.value.__context__ is None and info.value.__cause__ is None


def test_a_header_edited_after_the_open_is_refused_at_the_next_operation(world):
    ledger = Ledger.create(world.run, make_header())
    header = world.run / 'HEADER.json'
    header.write_bytes(header.read_bytes() + b' ')
    before = world.snapshot()
    for operation in (lambda: ledger.begin('train'), ledger.resume):
        with pytest.raises(Tampered, match='header file changed'):
            operation()
    assert world.snapshot() == before


def test_the_header_hash_in_the_ledger_must_name_the_header(world):
    Ledger.create(world.run, make_header())
    edit_json(world.run / 'LEDGER.json', lambda doc: doc.update(header_sha256='00' * 32))
    with pytest.raises(Tampered, match='header'):
        Ledger(world.run)


# ---- the lifecycle: begin, finish, resume --------------------------------------------------------------------------------------------
def test_a_run_of_two_stages_records_the_input_and_output_hashes_in_the_stage_files_and_the_ledger(world):
    ledger = world.complete()
    assert ledger.stages == (('train', 'finished'), ('evaluate', 'finished'))
    assert ledger.resume() == ['train', 'evaluate']
    header_sha = run_io.sha_file(world.run / 'HEADER.json')
    profile_key = world.profile.as_posix()
    best, log = sha(b'weights-v1|' * 20), sha(b'epoch 1 loss 3.5\nepoch 2 loss 3.1\n')
    train = run_io.read_json(world.run / 'stages' / 'train.json')
    assert train == {'schema': run_ledger.SCHEMA_STAGE, 'stage': 'train', 'header_sha256': header_sha, 'previous_sha256': header_sha,
                     'inputs': {profile_key: sha(PROFILE)}, 'outputs': {'train/best.bin': best, 'train/log.txt': log}}
    evaluate = run_io.read_json(world.run / 'stages' / 'evaluate.json')
    assert evaluate['inputs'] == {'train/best.bin': best} and evaluate['outputs'] == {'evaluate/scores.bin': sha(b'scores-v1|' * 8)}
    assert evaluate['previous_sha256'] == run_io.sha_file(world.run / 'stages' / 'train.json')                   # the chain: the previous stage file's hash
    doc = run_io.read_json(world.run / 'LEDGER.json')
    assert doc['schema'] == run_ledger.SCHEMA_LEDGER and doc['header_sha256'] == header_sha
    assert [(e['stage'], e['state']) for e in doc['stages']] == [('train', 'finished'), ('evaluate', 'finished')]
    assert [e['previous_sha256'] for e in doc['stages']] == [header_sha, doc['stages'][0]['stage_sha256']]
    for entry in doc['stages']:
        stage = run_io.read_json(world.run / 'stages' / f'{entry["stage"]}.json')
        assert entry['inputs'] == stage['inputs'] and entry['outputs'] == stage['outputs'] and entry['previous_sha256'] == stage['previous_sha256']
        assert entry['stage_sha256'] == run_io.sha_file(world.run / 'stages' / f'{entry["stage"]}.json')
        for key, digest in {**entry['inputs'], **entry['outputs']}.items():
            path = Path(key) if Path(key).is_absolute() else world.run / key
            assert digest == run_io.sha_file(path)                                    # the hashes are those of run_io.sha_file


def test_begin_records_the_input_hashes_and_finish_returns_the_hash_of_the_stage_file(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    assert ledger.stages == (('train', 'begun'),)
    assert run_io.read_json(world.run / 'LEDGER.json')['stages'] == [{'stage': 'train', 'state': 'begun', 'inputs': {world.profile.as_posix(): sha(PROFILE)}}]
    world.work('train')
    returned = ledger.finish('train', world.outputs('train'))
    assert returned == run_io.sha_file(world.run / 'stages' / 'train.json')
    assert run_io.read_json(world.run / 'LEDGER.json')['stages'][0]['stage_sha256'] == returned


def test_the_paths_may_be_text_or_paths_and_an_absolute_path_inside_the_run_folder_is_keyed_relative(world):
    ledger = Ledger.create(world.run, make_header())
    put(world.run / 'data' / 'in.bin', b'in')
    ledger.begin('train', inputs=[str(world.run / 'data' / 'in.bin'), world.profile])
    world.work('train')
    ledger.finish('train', [world.run / 'train' / 'best.bin', 'train/log.txt'])
    entry = run_io.read_json(world.run / 'stages' / 'train.json')
    assert sorted(entry['inputs']) == sorted(['data/in.bin', world.profile.as_posix()])
    assert sorted(entry['outputs']) == ['train/best.bin', 'train/log.txt']


def test_a_stage_with_no_inputs_and_no_outputs_is_legal(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('check')
    ledger.finish('check', [])
    assert run_io.read_json(world.run / 'stages' / 'check.json')['inputs'] == {} == run_io.read_json(world.run / 'stages' / 'check.json')['outputs']
    assert Ledger(world.run).resume() == ['check']


# ---- the chain of finished stages, the input binding and resume-first (K5a review I2) --------------------------------------------------------
CHAIN = ('prepare', 'train', 'evaluate')


def chain_work(world, stage):
    """Three stages in a line: ``prepare`` -> ``train`` (reads prepare/a.bin) -> ``evaluate`` (reads train/best.bin)."""
    files = {'prepare': ('prepare/a.bin', b'prepared-v1|' * 5), 'train': ('train/best.bin', b'weights-v1|' * 20), 'evaluate': ('evaluate/scores.bin', b'scores-v1|' * 8)}
    path, data = files[stage]
    put(world.run / path, data)
    return path


def chain_inputs(stage):
    return {'prepare': [], 'train': ['prepare/a.bin'], 'evaluate': ['train/best.bin']}[stage]


def play_chain(world, ledger, stages=CHAIN):
    for stage in stages:
        ledger.begin(stage, inputs=chain_inputs(stage))
        ledger.finish(stage, [chain_work(world, stage)])


def chain_run(world):
    ledger = Ledger.create(world.run, make_header())
    play_chain(world, ledger)
    return ledger


@pytest.mark.parametrize('removed', ['prepare', 'train'])
def test_removing_a_finished_stage_that_is_followed_by_another_is_refused(world, removed):
    """A whole record (the ledger entry and the stage file) of the first or a middle stage is removed: the chain of the next record is broken."""
    chain_run(world)
    edit_json(world.run / 'LEDGER.json', lambda doc: doc.update(stages=[e for e in doc['stages'] if e['stage'] != removed]))
    (world.run / 'stages' / f'{removed}.json').unlink()
    before = world.snapshot()
    with pytest.raises(Tampered, match='chain'):
        Ledger(world.run).resume()
    assert world.snapshot() == before


def test_removing_the_last_finished_stage_is_benign_the_stage_runs_again(world):
    """Nothing follows it, so nothing is broken: the run is the run before that stage, and the stage is begun and finished again."""
    chain_run(world)
    edit_json(world.run / 'LEDGER.json', lambda doc: doc['stages'].pop())
    (world.run / 'stages' / 'evaluate.json').unlink()
    ledger = Ledger(world.run)
    assert ledger.resume() == ['prepare', 'train']
    play_chain(world, ledger, ('evaluate',))
    assert Ledger(world.run).resume() == ['prepare', 'train', 'evaluate']


def test_the_chain_names_the_previous_record_and_the_header_for_the_first(world):
    chain_run(world)
    header_sha = run_io.sha_file(world.run / 'HEADER.json')
    entries = run_io.read_json(world.run / 'LEDGER.json')['stages']
    assert entries[0]['previous_sha256'] == header_sha
    assert entries[1]['previous_sha256'] == entries[0]['stage_sha256'] == run_io.sha_file(world.run / 'stages' / 'prepare.json')
    assert entries[2]['previous_sha256'] == entries[1]['stage_sha256']
    for name in CHAIN:
        assert run_io.read_json(world.run / 'stages' / f'{name}.json')['previous_sha256'] == next(e for e in entries if e['stage'] == name)['previous_sha256']


@pytest.mark.parametrize('edit', [lambda doc: doc['stages'][1].update(previous_sha256='00' * 32), lambda doc: doc['stages'][0].update(previous_sha256='00' * 32),
                                  lambda doc: doc['stages'][2].update(previous_sha256=doc['stages'][0]['stage_sha256']),
                                  lambda doc: doc['stages'][1].pop('previous_sha256'), lambda doc: doc['stages'][1].update(previous_sha256='zz')],
                         ids=['middle', 'first', 'skips-one', 'missing', 'not-a-hash'])
def test_a_broken_chain_in_the_ledger_is_refused_by_the_open_alone(world, edit):
    chain_run(world)
    edit_json(world.run / 'LEDGER.json', edit)
    with pytest.raises(Tampered):
        Ledger(world.run)


def test_the_chain_of_the_unfinalised_stage_is_checked_too(world, monkeypatch):
    ledger = Ledger.create(world.run, make_header())
    play_chain(world, ledger, ('prepare',))
    ledger.begin('train', inputs=chain_inputs('train'))
    chain_work(world, 'train')
    stop_writer_at(monkeypatch, 1, after=True)
    with pytest.raises(Crash):
        ledger.finish('train', ['train/best.bin'])                                  # the stage file is written, the ledger is not
    monkeypatch.undo()
    stage_file = world.run / 'stages' / 'train.json'
    good = stage_file.read_bytes()
    edit_json(stage_file, lambda doc: doc.update(previous_sha256='00' * 32))
    before = world.snapshot()
    with pytest.raises(Tampered):
        Ledger(world.run).resume()
    assert world.snapshot() == before
    stage_file.write_bytes(good)
    assert Ledger(world.run).resume() == ['prepare', 'train']


def test_begin_binds_an_input_to_the_output_that_an_earlier_stage_recorded(world):
    """A runner bug that rewrites ``train/best.bin`` after ``finish('train')`` must not feed an uncertified checkpoint to ``evaluate``."""
    ledger = Ledger.create(world.run, make_header())
    play_chain(world, ledger, ('prepare', 'train'))
    before = world.snapshot()
    put(world.run / 'train' / 'best.bin', b'weights-v2|' * 20)                       # changed between the stages
    with pytest.raises(Tampered, match="stage 'evaluate'.*earlier stage"):
        ledger.begin('evaluate', inputs=['train/best.bin'])
    assert ledger.stages == (('prepare', 'finished'), ('train', 'finished'))
    assert {k: v for k, v in world.snapshot().items() if k != 'train/best.bin'} == {k: v for k, v in before.items() if k != 'train/best.bin'}
    put(world.run / 'train' / 'best.bin', b'weights-v1|' * 20)                       # restored: the recorded hash holds again
    ledger.begin('evaluate', inputs=['train/best.bin'])
    assert ledger.stages[-1] == ('evaluate', 'begun')


def test_the_input_binding_applies_to_a_restart_and_leaves_other_inputs_alone(world):
    ledger = Ledger.create(world.run, make_header())
    play_chain(world, ledger, ('prepare',))
    ledger.begin('train', inputs=['prepare/a.bin', world.profile])                  # the profile is no earlier output: not bound
    put(world.run / 'prepare' / 'a.bin', b'prepared-v2|')
    with pytest.raises(Tampered):
        ledger.begin('train', inputs=['prepare/a.bin', world.profile])              # begun again on a changed earlier output
    put(world.run / 'prepare' / 'a.bin', b'prepared-v1|' * 5)
    ledger.begin('train', inputs=['prepare/a.bin', world.profile])
    put(world.profile, PROFILE + b' ')
    with pytest.raises(LedgerRefused, match='other inputs'):
        ledger.begin('train', inputs=['prepare/a.bin', world.profile])


def test_an_opened_ledger_refuses_begin_and_finish_until_resume_has_run(world):
    play_chain(world, Ledger.create(world.run, make_header()), ('prepare',))
    opened = Ledger(world.run)
    before = world.snapshot()
    with pytest.raises(LedgerRefused, match='resume') as info:
        opened.begin('train', inputs=chain_inputs('train'))
    assert not isinstance(info.value, Tampered)
    with pytest.raises(LedgerRefused, match='resume'):
        opened.finish('train', [])
    assert world.snapshot() == before
    assert opened.resume() == ['prepare']
    opened.begin('train', inputs=chain_inputs('train'))
    assert opened.stages[-1] == ('train', 'begun')
    chain_work(world, 'train')
    opened.finish('train', ['train/best.bin'])
    assert Ledger(world.run).resume() == ['prepare', 'train']


def test_a_created_ledger_counts_as_resumed_and_a_failed_resume_does_not(world):
    created = Ledger.create(world.run, make_header())
    created.begin('prepare')                                                         # no resume() needed after create
    created.finish('prepare', [chain_work(world, 'prepare')])
    flip(world.run / 'prepare' / 'a.bin')
    opened = Ledger(world.run)
    with pytest.raises(Tampered):
        opened.resume()
    with pytest.raises(LedgerRefused, match='resume'):
        opened.begin('train', inputs=[])                                             # the resume did not succeed: still not resumed
    put(world.run / 'prepare' / 'a.bin', b'prepared-v1|' * 5)
    assert opened.resume() == ['prepare']
    opened.begin('train', inputs=chain_inputs('train'))


@pytest.mark.parametrize('name', ['', 'Train', 'a b', 'a/b', 'a\\b', '..', '.', 'a.b', 'a-b', '1train', 'train ', 'x' * 49, 'con', 'nul', 'com1', 'lpt9', 'stage\n',
                                  'é', None, 5, b'train'])
def test_a_stage_name_must_be_a_short_lowercase_word_that_is_safe_as_a_file_name(world, name):
    ledger = Ledger.create(world.run, make_header())
    before = world.snapshot()
    with pytest.raises(LedgerRefused):
        ledger.begin(name)
    with pytest.raises(LedgerRefused):
        ledger.finish(name, [])
    assert world.snapshot() == before


@pytest.mark.parametrize('name', ['train', 'b0_scales', 'a', 'x' * 48, 'train_ept_seed2', 'eval2'])
def test_the_ordinary_stage_names_are_accepted(world, name):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin(name)
    ledger.finish(name, [])
    assert ledger.resume() == [name]


def test_begin_refuses_what_would_mix_two_runs_of_a_stage(world, monkeypatch):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    before = world.snapshot()
    writes = count_writes(monkeypatch)
    ledger.begin('train', inputs=world.inputs('train'))                              # begin again on the same inputs: a restart, no new write
    assert writes == [] and world.snapshot() == before
    with pytest.raises(LedgerRefused, match='still begun'):
        ledger.begin('evaluate')                                                   # one stage at a time
    put(world.profile, PROFILE + b' ')                                              # the input changed
    with pytest.raises(LedgerRefused, match='other inputs'):
        ledger.begin('train', inputs=world.inputs('train'))
    with pytest.raises(LedgerRefused, match='other inputs'):
        ledger.begin('train', inputs=[])
    assert ledger.stages == (('train', 'begun'),) and writes == []


def test_begin_refuses_a_finished_stage_a_missing_input_and_a_bad_path(world):
    ledger = world.complete()
    with pytest.raises(LedgerRefused, match='finished'):
        ledger.begin('train', inputs=world.inputs('train'))
    fresh = Ledger.create(world.root / 'second', make_header())
    (world.root / 'sub').mkdir()
    for inputs in ([world.root / 'missing.bin'], ['../outside.bin'], ['/abs.bin'], ['a\\b.bin'], ['a/../b.bin'], [''], ['a//b'], [5], [None], 'text',
                   {'a': 'b'}, b'bytes', [world.profile, world.profile], [world.root],
                   [str(world.root / 'sub' / '..' / 'profile.json')], [world.root / 'sub' / '..' / 'profile.json']):    # an absolute path with ".." in it
        with pytest.raises(LedgerRefused):
            fresh.begin('train', inputs=inputs)
    assert fresh.stages == () and sorted(p.name for p in (world.root / 'second').iterdir()) == ['HEADER.json', 'LEDGER.json']
    fresh.begin('train', inputs=[str(world.profile)])                               # the same file by its plain absolute path is fine
    assert fresh.stages == (('train', 'begun'),)


def test_a_one_character_text_is_not_a_list_of_one_path(world):
    """A bare string iterates by character: ``'x'`` would pass as the list ``['x']``. Inputs and outputs must be lists, not text."""
    ledger = Ledger.create(world.run, make_header())
    put(world.run / 'x', b'1')
    put(world.root / 'y', b'1')
    with pytest.raises(LedgerRefused, match='list'):
        ledger.begin('train', inputs='x')
    with pytest.raises(LedgerRefused, match='list'):
        ledger.begin('train', inputs=str(world.root / 'y'))
    ledger.begin('train')
    with pytest.raises(LedgerRefused, match='list'):
        ledger.finish('train', 'x')
    ledger.finish('train', ['x'])
    assert run_io.read_json(world.run / 'stages' / 'train.json')['outputs'] == {'x': sha(b'1')}


@pytest.mark.skipif(os.name != 'nt', reason='a drive-relative path (C:name) is a Windows form')
def test_a_drive_relative_path_is_not_a_second_name_for_a_file_of_the_run(world):
    """On Windows ``run_dir / 'C:name'`` is ``run_dir / 'name'`` when the drive is the same: one file, two keys. The colon rule refuses it."""
    ledger = Ledger.create(world.run, make_header())
    put(world.run / 'escape.bin', b'x')
    alias = f'{world.run.drive}escape.bin'
    with pytest.raises(LedgerRefused):
        ledger.begin('train', inputs=[alias])
    ledger.begin('train')
    with pytest.raises(LedgerRefused):
        ledger.finish('train', [alias])
    ledger.finish('train', ['escape.bin'])
    assert run_io.read_json(world.run / 'stages' / 'train.json')['outputs'] == {'escape.bin': sha(b'x')}


def test_finish_refuses_a_stage_that_was_not_begun_was_finished_or_has_a_missing_output(world):
    ledger = Ledger.create(world.run, make_header())
    with pytest.raises(LedgerRefused, match='not begun'):
        ledger.finish('train', [])
    ledger.begin('train', inputs=world.inputs('train'))
    before = world.snapshot()
    with pytest.raises(LedgerRefused):
        ledger.finish('train', world.outputs('train'))                              # the work was not done: the outputs are missing
    assert world.snapshot() == before and not (world.run / 'stages').exists()      # no stage file, no ledger change
    world.work('train')
    ledger.finish('train', world.outputs('train'))
    with pytest.raises(LedgerRefused, match='finished'):
        ledger.finish('train', world.outputs('train'))


@pytest.mark.parametrize('outputs', ['train/best.bin', b'train/best.bin', {'train/best.bin': 'x'}, [5], [None], ['../outside.bin'], ['/abs.bin'], ['a\\b.bin'],
                                     ['train/best.bin', 'train/best.bin'], ['train'], ['train/'], ['C:/x.bin'], ['train/best.bin ']])
def test_finish_refuses_bad_output_lists_and_writes_nothing(world, outputs):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    world.work('train')
    put(world.root / 'outside.bin', b'x')
    before = world.snapshot()
    with pytest.raises(LedgerRefused):
        ledger.finish('train', outputs)
    assert world.snapshot() == before and not (world.run / 'stages').exists()


def test_an_output_outside_the_run_folder_is_refused_even_as_an_absolute_path(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train')
    outside = put(world.root / 'elsewhere.bin', b'x')
    with pytest.raises(LedgerRefused):
        ledger.finish('train', [outside])
    assert ledger.stages == (('train', 'begun'),)


def test_finish_refuses_when_an_input_changed_after_begin_and_writes_nothing(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    world.work('train')
    put(world.profile, PROFILE.replace(b'train', b'evals'))
    before = world.snapshot()
    with pytest.raises(Tampered, match="'train'.*input"):
        ledger.finish('train', world.outputs('train'))
    assert world.snapshot() == before and not (world.run / 'stages').exists()
    world.profile.unlink()
    with pytest.raises(Tampered, match="'train'.*input"):
        ledger.finish('train', world.outputs('train'))


def test_a_stage_that_has_a_stage_file_is_completed_by_resume_not_begun_or_finished_again(world, monkeypatch):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    world.work('train')
    stop_writer_at(monkeypatch, 2, after=False)                                      # the stage file is written (write 1), the ledger is not (write 2)
    with pytest.raises(Crash):
        ledger.finish('train', world.outputs('train'))
    monkeypatch.undo()
    assert (world.run / 'stages' / 'train.json').exists()
    with pytest.raises(LedgerRefused, match='stage file'):                           # the object that crashed (it is resumed) is told why
        ledger.begin('train', inputs=world.inputs('train'))
    with pytest.raises(LedgerRefused, match='stage file'):
        ledger.finish('train', world.outputs('train'))
    again = Ledger(world.run)
    with pytest.raises(LedgerRefused, match='resume'):
        again.begin('train', inputs=world.inputs('train'))
    with pytest.raises(LedgerRefused, match='resume'):
        again.finish('train', world.outputs('train'))
    assert again.resume() == ['train'] and again.stages == (('train', 'finished'),)


def test_resume_is_idempotent_ordered_and_writes_nothing_when_nothing_is_pending(world, monkeypatch):
    ledger = world.complete()
    first = world.snapshot()
    writes = count_writes(monkeypatch)
    assert ledger.resume() == ['train', 'evaluate'] == ledger.resume() == Ledger(world.run).resume()
    assert writes == [] and world.snapshot() == first


def test_a_second_owner_cannot_write_after_the_ledger_changed_behind_it(world):
    """Ref: ``CampaignLedger._verify``: the ledger belongs to one owner; a file that is not the one this object read or wrote is refused."""
    first = Ledger.create(world.run, make_header())
    second = Ledger(world.run)
    first.begin('train', inputs=world.inputs('train'))
    before = world.snapshot()
    with pytest.raises(Tampered, match='changed'):
        second.begin('train', inputs=world.inputs('train'))
    with pytest.raises(Tampered, match='changed'):
        second.resume()
    assert world.snapshot() == before
    third = Ledger(world.run)                                                         # a fresh open sees the file as it is
    assert third.stages == (('train', 'begun'),) and third.resume() == []
    world.work('train')
    third.finish('train', world.outputs('train'))
    with pytest.raises(Tampered, match='changed'):
        first.finish('train', world.outputs('train'))                               # finish too: first still believes that train is only begun
    assert Ledger(world.run).stages == (('train', 'finished'),)
    third = Ledger(world.run)
    edit_json(world.run / 'LEDGER.json', lambda doc: doc['stages'][0]['inputs'].update({'x': '00' * 32}))
    with pytest.raises(Tampered, match='changed'):
        third.begin('train', inputs=world.inputs('train'))
    (world.run / 'LEDGER.json').unlink()
    with pytest.raises(Tampered, match='missing|changed'):
        third.begin('train', inputs=world.inputs('train'))


# ---- a stop at every write point (Review Focus 4: recoverable, A5-F7) -------------------------------------------------------------------
# The writes of a full two-stage run, in order:
#   1 HEADER.json   2 LEDGER.json (empty)                        create
#   3 LEDGER.json (train begun)                                  begin train
#   4 stages/train.json   5 LEDGER.json (train finished)         finish train
#   6 LEDGER.json (evaluate begun)                               begin evaluate
#   7 stages/evaluate.json   8 LEDGER.json (evaluate finished)   finish evaluate
# A stop at write n is "before" (the write did not happen) or "after" (it happened, the caller did not see it return).
CRASH_POINTS = [
    (1, False, 'nothing'), (1, True, 'header only'), (2, False, 'header only'), (2, True, []),
    (3, False, []), (3, True, []),
    (4, False, []), (4, True, ['train']),                    # after 4: the stage file exists, the ledger says begun: resume completes it
    (5, False, ['train']), (5, True, ['train']),
    (6, False, ['train']), (6, True, ['train']),
    (7, False, ['train']), (7, True, ['train', 'evaluate']),
    (8, False, ['train', 'evaluate']), (8, True, ['train', 'evaluate']),
]


def uninterrupted_run(world, stages=STAGES):
    """The files of a run that was never stopped (the run folder is removed again afterwards): the reference for a recovered run."""
    ledger = Ledger.create(world.run, make_header())
    for stage in stages:
        world.play(ledger, stage)
    files = world.snapshot()
    shutil.rmtree(world.run)
    return files


@pytest.mark.parametrize('number,after,expected', CRASH_POINTS, ids=[f'write{n}-{"after" if a else "before"}' for n, a, _ in CRASH_POINTS])
def test_finalisation_completes_after_a_crash_between_writes(world, monkeypatch, number, after, expected):
    """A5-F7: stop the writer at every write point. On the next start the ledger is completed from the stage file (hashes re-verified) or the
    stage is begun again; the finished run is byte for byte the run that was never stopped. A stop during creation leaves a folder that is refused."""
    expected_files = uninterrupted_run(world)
    stop_writer_at(monkeypatch, number, after)
    with pytest.raises(Crash):
        ledger = Ledger.create(world.run, make_header())
        for stage in STAGES:
            world.play(ledger, stage)
    monkeypatch.undo()
    assert not list(world.run.rglob('*.tmp')) if world.run.exists() else True               # the atomic writer leaves no temporary file
    if expected in ('nothing', 'header only'):
        listing = sorted(world.snapshot()) if world.run.exists() else []
        assert listing == ([] if expected == 'nothing' else ['HEADER.json'])
        if expected == 'header only':
            with pytest.raises(LedgerRefused):
                Ledger(world.run)                                                   # fail-closed: a folder whose creation did not finish is not opened ...
            with pytest.raises(LedgerRefused, match='already'):
                Ledger.create(world.run, make_header())                             # ... and is not overwritten
            assert sorted(world.snapshot()) == ['HEADER.json']
            return
        ledger = Ledger.create(world.run, make_header())                           # nothing was written: creation can simply be done again
        assert world.continue_run(ledger) == []
    else:
        ledger = Ledger(world.run)
        done = world.continue_run(ledger)
        assert done == expected
    assert world.snapshot() == expected_files
    assert Ledger(world.run).resume() == ['train', 'evaluate']
    assert Ledger(world.run).stages == (('train', 'finished'), ('evaluate', 'finished'))


def test_a_stop_during_the_completion_by_resume_is_recoverable_too(world, monkeypatch):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    world.work('train')
    stop_writer_at(monkeypatch, 1, after=True)
    with pytest.raises(Crash):
        ledger.finish('train', world.outputs('train'))                              # the stage file is written, the ledger is not
    monkeypatch.undo()
    for after in (False, True):
        stop_writer_at(monkeypatch, 1, after=after)                                  # the one write of the completion
        with pytest.raises(Crash):
            Ledger(world.run).resume()
        monkeypatch.undo()
    final = Ledger(world.run)
    assert final.resume() == ['train'] and final.stages == (('train', 'finished'),)
    after_resume = world.snapshot()
    assert Ledger(world.run).resume() == ['train'] and world.snapshot() == after_resume


def test_the_completion_by_resume_writes_only_the_ledger_and_gives_the_ledger_an_uninterrupted_run_would_have(world, monkeypatch):
    expected_files = uninterrupted_run(world, stages=('train',))
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    world.work('train')
    stop_writer_at(monkeypatch, 1, after=True)
    with pytest.raises(Crash):
        ledger.finish('train', world.outputs('train'))
    monkeypatch.undo()
    stage_file = (world.run / 'stages' / 'train.json').read_bytes()
    writes = count_writes(monkeypatch)
    assert Ledger(world.run).resume() == ['train']
    assert writes == ['LEDGER.json']                                                  # the stage file is not written again
    assert (world.run / 'stages' / 'train.json').read_bytes() == stage_file
    assert world.snapshot() == expected_files


# ---- tampering: a stage file that does not match the files is refused (Review Focus 4) ---------------------------------------------------
def flip(path: Path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def tamper_output_byte(w):
    flip(w.run / 'train' / 'best.bin')


def tamper_output_log(w):
    flip(w.run / 'train' / 'log.txt')


def tamper_last_output(w):
    flip(w.run / 'evaluate' / 'scores.bin')


def tamper_external_input_byte(w):
    flip(w.profile)


def tamper_deleted_output(w):
    (w.run / 'train' / 'log.txt').unlink()


def tamper_deleted_input(w):
    w.profile.unlink()


def tamper_output_replaced_by_a_folder(w):
    (w.run / 'evaluate' / 'scores.bin').unlink()
    (w.run / 'evaluate' / 'scores.bin').mkdir()


def tamper_stage_file_byte(w):
    flip(w.run / 'stages' / 'train.json')


def tamper_stage_file_output_hash(w):
    edit_json(w.run / 'stages' / 'train.json', lambda doc: doc['outputs'].update({'train/best.bin': '00' * 32}))


def tamper_stage_file_and_ledger_stage_hash_together(w):
    edit_json(w.run / 'stages' / 'train.json', lambda doc: doc['outputs'].update({'train/best.bin': '00' * 32}))
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0].update(stage_sha256=run_io.sha_file(w.run / 'stages' / 'train.json')))


def tamper_stage_file_and_ledger_entry_together(w):
    """A consistent forgery of both records: the files on disk still have the real hashes, so it is refused."""
    edit_json(w.run / 'stages' / 'train.json', lambda doc: doc['outputs'].update({'train/best.bin': '00' * 32}))
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0]['outputs'].update({'train/best.bin': '00' * 32}))
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0].update(stage_sha256=run_io.sha_file(w.run / 'stages' / 'train.json')))


def tamper_stage_file_extra_key(w):
    edit_json(w.run / 'stages' / 'train.json', lambda doc: doc.update(extra=1))


def tamper_stage_file_schema(w):
    edit_json(w.run / 'stages' / 'train.json', lambda doc: doc.update(schema='ept-run-stage-v0'))


def tamper_stage_file_other_stage_name(w):
    edit_json(w.run / 'stages' / 'train.json', lambda doc: doc.update(stage='evaluate'))


def tamper_stage_file_deleted(w):
    (w.run / 'stages' / 'evaluate.json').unlink()


def tamper_stage_file_from_another_run(w):
    other = World(w.root)                                                          # the same profile file ...
    other.run = w.root / 'run_other'
    other.complete(make_header(profile_sha256='77' * 32))                          # ... the same file names and bytes, another header
    put(w.run / 'stages' / 'train.json', (other.run / 'stages' / 'train.json').read_bytes())
    assert run_io.read_json(w.run / 'stages' / 'train.json')['inputs'] == run_io.read_json(other.run / 'stages' / 'train.json')['inputs']


def tamper_ledger_output_hash(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][1]['outputs'].update({'evaluate/scores.bin': '00' * 32}))


def tamper_ledger_input_hash(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][1]['inputs'].update({'train/best.bin': '00' * 32}))


def tamper_ledger_stage_sha(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0].update(stage_sha256='00' * 32))


def tamper_ledger_output_dropped(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0]['outputs'].pop('train/log.txt'))


def tamper_ledger_output_added(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][1]['outputs'].update({'evaluate/ghost.bin': '00' * 32}))


def tamper_ledger_stage_dropped_but_stage_file_kept(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'].pop())             # an extra stage file: "unknown stage"


def tamper_ledger_stage_added_without_a_stage_file(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'].append({'stage': 'ghost', 'state': 'finished', 'inputs': {}, 'outputs': {}, 'stage_sha256': '00' * 32}))


def tamper_extra_stage_file(w):
    put(w.run / 'stages' / 'ghost.json', (w.run / 'stages' / 'train.json').read_bytes())


def tamper_extra_file_in_the_stage_folder(w):
    put(w.run / 'stages' / 'notes.txt', b'hello')


def tamper_extra_folder_in_the_stage_folder(w):
    (w.run / 'stages' / 'train.json.d').mkdir()


def tamper_ledger_entry_with_an_extra_key(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0].update(extra=1))


def tamper_ledger_duplicate_stage(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'].append(dict(doc['stages'][0])))


def tamper_ledger_duplicate_stage_name_with_a_valid_chain(w):
    """A begun entry that repeats the name of a finished stage: the chain is untouched (a begun entry has no chain field), so only the rule
    "one entry per stage" refuses it."""
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'].append({'stage': 'train', 'state': 'begun', 'inputs': {}}))


def tamper_ledger_unknown_state(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0].update(state='done'))


def tamper_ledger_a_begun_stage_before_a_finished_one(w):
    """The first stage is turned back into a begun entry and the chain of the second is made consistent with that: only the rule "a begun entry
    is the last one" refuses it."""
    def change(doc):
        doc['stages'][0].update(state='begun')
        for key in ('outputs', 'stage_sha256', 'previous_sha256'):
            doc['stages'][0].pop(key)
        doc['stages'][1]['previous_sha256'] = doc['header_sha256']
    edit_json(w.run / 'LEDGER.json', change)


def forge_record(w, stage='evaluate', *, inputs=None, outputs=None):
    """Change the input or output record of ``stage`` in its stage file and in its ledger entry alike (and the stage-file hash that the ledger
    keeps), so that the two records agree with each other; only the files can still tell. The default is the last stage: forging an earlier one
    also breaks the chain of the stage after it, which would refuse the run for that reason and hide the rule that a test is about."""
    changes = {key: value for key, value in (('inputs', inputs), ('outputs', outputs)) if value is not None}
    stage_file = w.run / 'stages' / f'{stage}.json'
    edit_json(stage_file, lambda doc: doc.update(changes))
    edit_json(w.run / 'LEDGER.json', lambda doc: next(e for e in doc['stages'] if e['stage'] == stage).update(changes, stage_sha256=run_io.sha_file(stage_file)))


def tamper_ledger_output_path_leaves_the_run_folder(w):
    """Both records are forged consistently, with a real file outside the run folder whose hash matches: still refused (a malformed path)."""
    put(w.root / 'outside.bin', b'x')
    forge_record(w, outputs={'../outside.bin': sha(b'x')})


def tamper_output_key_is_absolute(w):
    """A real file outside the run folder with its true hash, named by an absolute path in both records: an output must be inside the run folder."""
    forge_record(w, outputs={w.profile.as_posix(): sha(PROFILE)})


def tamper_stage_file_reformatted(w):
    """The same content in other white space: the parsed stage file is what the ledger records, only its hash tells."""
    path = w.run / 'stages' / 'evaluate.json'
    path.write_bytes(json.dumps(run_io.read_json(path), indent=4, sort_keys=True).encode('utf-8'))


def tamper_stage_file_is_a_folder(w):
    stage_file = w.run / 'stages' / ('evaluate.json' if (w.run / 'stages' / 'evaluate.json').exists() else 'train.json')
    stage_file.unlink()
    stage_file.mkdir()


def tamper_ledger_is_a_folder(w):
    (w.run / 'LEDGER.json').unlink()
    (w.run / 'LEDGER.json').mkdir()


def tamper_stages_folder_is_a_file(w):
    shutil.rmtree(w.run / 'stages')
    put(w.run / 'stages', b'not a folder')


def tamper_ledger_stage_sha_is_not_a_hash(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][0].update(stage_sha256='zz'))


def tamper_ledger_last_stage_sha_is_not_a_hash(w):
    """The last finished entry: no later entry compares with it, so only the shape check of the open can refuse it before the files are read."""
    edit_json(w.run / 'LEDGER.json', lambda doc: doc['stages'][-1].update(stage_sha256='zz'))


def tamper_ledger_extra_field(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc.update(extra=1))


def tamper_ledger_stages_is_an_empty_object(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc.update(stages={}))


def tamper_ledger_header_hash_is_not_a_hash(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc.update(header_sha256='zz'))


def tamper_ledger_schema(w):
    edit_json(w.run / 'LEDGER.json', lambda doc: doc.update(schema='ept-run-ledger-v0'))


def tamper_ledger_not_json(w):
    (w.run / 'LEDGER.json').write_bytes(b'{"schema": ')


def tamper_ledger_duplicate_json_key(w):
    text = (w.run / 'LEDGER.json').read_text(encoding='utf-8')
    (w.run / 'LEDGER.json').write_text(text.replace('"schema":', '"schema": "x", "schema":', 1), encoding='utf-8', newline='')


def tamper_ledger_deleted(w):
    (w.run / 'LEDGER.json').unlink()


def tamper_header_edited(w):
    edit_json(w.run / 'HEADER.json', lambda doc: doc['runner'].update(device='cuda'))


TAMPERS = [tamper_output_byte, tamper_output_log, tamper_last_output, tamper_external_input_byte,
           tamper_deleted_output, tamper_deleted_input, tamper_output_replaced_by_a_folder,
           tamper_stage_file_byte, tamper_stage_file_output_hash, tamper_stage_file_and_ledger_stage_hash_together,
           tamper_stage_file_and_ledger_entry_together, tamper_stage_file_extra_key, tamper_stage_file_schema, tamper_stage_file_other_stage_name,
           tamper_stage_file_deleted, tamper_stage_file_from_another_run,
           tamper_ledger_output_hash, tamper_ledger_input_hash, tamper_ledger_stage_sha, tamper_ledger_output_dropped, tamper_ledger_output_added,
           tamper_ledger_stage_dropped_but_stage_file_kept, tamper_ledger_stage_added_without_a_stage_file,
           tamper_extra_stage_file, tamper_extra_file_in_the_stage_folder, tamper_extra_folder_in_the_stage_folder,
           tamper_ledger_entry_with_an_extra_key, tamper_ledger_duplicate_stage, tamper_ledger_duplicate_stage_name_with_a_valid_chain, tamper_ledger_unknown_state,
           tamper_ledger_a_begun_stage_before_a_finished_one, tamper_ledger_output_path_leaves_the_run_folder,
           tamper_output_key_is_absolute, tamper_stages_folder_is_a_file, tamper_ledger_stage_sha_is_not_a_hash, tamper_ledger_last_stage_sha_is_not_a_hash, tamper_ledger_extra_field,
           tamper_stage_file_is_a_folder, tamper_ledger_is_a_folder, tamper_stage_file_reformatted,
           tamper_ledger_stages_is_an_empty_object, tamper_ledger_header_hash_is_not_a_hash,
           tamper_ledger_schema, tamper_ledger_not_json, tamper_ledger_duplicate_json_key, tamper_ledger_deleted, tamper_header_edited]

# The tampers that damage a record's shape: the open alone (no hashing of any file) must refuse them.
RECORD_TAMPERS = [tamper_ledger_entry_with_an_extra_key, tamper_ledger_duplicate_stage, tamper_ledger_duplicate_stage_name_with_a_valid_chain, tamper_ledger_unknown_state,
                  tamper_ledger_a_begun_stage_before_a_finished_one, tamper_ledger_output_path_leaves_the_run_folder, tamper_output_key_is_absolute,
                  tamper_ledger_stage_sha_is_not_a_hash, tamper_ledger_last_stage_sha_is_not_a_hash, tamper_ledger_extra_field, tamper_ledger_stages_is_an_empty_object, tamper_ledger_is_a_folder,
                  tamper_ledger_header_hash_is_not_a_hash, tamper_ledger_schema, tamper_ledger_not_json, tamper_ledger_duplicate_json_key,
                  tamper_ledger_deleted, tamper_header_edited]


@pytest.mark.parametrize('tamper', TAMPERS, ids=[t.__name__.removeprefix('tamper_') for t in TAMPERS])
def test_tampered_stage_file_is_refused(world, tamper):
    """Review Focus 4: a changed output byte, a changed input byte, a deleted output, an edited stage file, an edited ledger entry, an extra
    unknown stage (and the malformed records around them) are all refused, as ``Tampered``, and the refusal writes nothing. The refusal comes
    from opening the ledger (a malformed record) or from ``resume`` (a hash that does not match the file)."""
    world.complete()
    tamper(world)
    before = world.snapshot()
    with pytest.raises(Tampered) as info:
        Ledger(world.run).resume()
    assert world.snapshot() == before                                               # a refusal changes nothing in the run folder
    message = str(info.value)
    assert message and str(world.root) not in message and 'weights' not in message and sha(PROFILE) not in message


@pytest.mark.parametrize('tamper', RECORD_TAMPERS, ids=[t.__name__.removeprefix('tamper_') for t in RECORD_TAMPERS])
def test_a_record_that_is_malformed_is_refused_by_the_open_alone(world, tamper):
    """The shape of the header and ledger records is checked when the ledger is opened, before any file is hashed."""
    world.complete()
    tamper(world)
    with pytest.raises(Tampered):
        Ledger(world.run)


@pytest.mark.skipif(os.name != 'nt', reason='a backslash path is a Windows form')
def test_an_input_key_in_backslash_form_is_a_malformed_record(world):
    """The keys of the records are in one normal form (forward slashes). A record that names the same real file with backslashes is refused."""
    world.complete()
    forge_record(world, inputs={str(world.profile): sha(PROFILE)})
    assert '\\' in str(world.profile)
    with pytest.raises(Tampered):
        Ledger(world.run)


@pytest.mark.parametrize('tamper', [tamper_output_byte, tamper_deleted_output, tamper_stage_file_byte, tamper_stage_file_output_hash,
                                    tamper_stage_file_and_ledger_entry_together, tamper_extra_stage_file], ids=lambda t: t.__name__.removeprefix('tamper_'))
def test_a_run_that_is_tampered_stays_refused_on_every_resume_and_is_not_repaired(world, tamper):
    world.complete()
    tamper(world)
    for _ in range(2):
        with pytest.raises(Tampered):
            Ledger(world.run).resume()


def test_the_tamper_cases_each_break_a_run_that_resumes_cleanly_before(world):
    """Guard against vacuous cases: the untouched run resumes, so each refusal above comes from its tamper and not from the set-up."""
    world.complete()
    assert Ledger(world.run).resume() == ['train', 'evaluate']
    assert len(TAMPERS) == len({t.__name__ for t in TAMPERS}) >= 30


# the stage whose stage file was written but whose ledger entry is still "begun": resume re-verifies it before it completes the ledger
def make_in_flight(world, monkeypatch):
    ledger = Ledger.create(world.run, make_header())
    world.play(ledger, 'train')
    ledger.begin('evaluate', inputs=world.inputs('evaluate'))
    world.work('evaluate')
    stop_writer_at(monkeypatch, 1, after=True)
    with pytest.raises(Crash):
        ledger.finish('evaluate', world.outputs('evaluate'))
    monkeypatch.undo()
    assert Ledger(world.run).stages == (('train', 'finished'), ('evaluate', 'begun'))


IN_FLIGHT_TAMPERS = [tamper_last_output, tamper_output_replaced_by_a_folder, tamper_stage_file_is_a_folder,
                     lambda w: (w.run / 'evaluate' / 'scores.bin').unlink(),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc['outputs'].update({'evaluate/scores.bin': '00' * 32})),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc['inputs'].update({'train/best.bin': '00' * 32})),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc['inputs'].pop('train/best.bin')),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc.update(header_sha256='00' * 32)),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc.update(stage='train')),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc.update(extra=1)),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc.update(schema='ept-run-stage-v0')),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc.update(outputs={w.profile.as_posix(): sha(PROFILE)})),
                     lambda w: edit_json(w.run / 'stages' / 'evaluate.json', lambda doc: doc['outputs'].update({'../x': '00' * 32})),
                     lambda w: (w.run / 'stages' / 'evaluate.json').write_bytes(b'{"schema": '),
                     lambda w: flip(w.run / 'stages' / 'evaluate.json'),
                     lambda w: put(w.run / 'stages' / 'ghost.json', b'{}\n')]


@pytest.mark.parametrize('index', range(len(IN_FLIGHT_TAMPERS)))
def test_an_unfinalised_stage_file_is_re_verified_and_a_mismatch_keeps_the_run_fail_closed(world, monkeypatch, index):
    make_in_flight(world, monkeypatch)
    IN_FLIGHT_TAMPERS[index](world)
    before = world.snapshot()
    with pytest.raises(Tampered):
        Ledger(world.run).resume()
    assert world.snapshot() == before                                               # the ledger was not completed
    assert run_io.read_json(world.run / 'LEDGER.json')['stages'][1]['state'] == 'begun'


def test_an_unfinalised_stage_file_that_matches_is_completed(world, monkeypatch):
    make_in_flight(world, monkeypatch)
    ledger = Ledger(world.run)
    assert ledger.resume() == ['train', 'evaluate'] and ledger.stages == (('train', 'finished'), ('evaluate', 'finished'))
    entry = run_io.read_json(world.run / 'LEDGER.json')['stages'][1]
    assert entry['stage_sha256'] == run_io.sha_file(world.run / 'stages' / 'evaluate.json')
    assert entry['inputs'] == {'train/best.bin': sha(b'weights-v1|' * 20)}


def test_a_stage_file_with_no_ledger_entry_at_all_is_an_unknown_stage_not_a_stage_to_complete(world):
    """The ledger entry is written first (begin), so a stage file without any entry cannot come from a crash: it is refused, not adopted."""
    ledger = Ledger.create(world.run, make_header())
    world.play(ledger, 'train')
    put(world.run / 'stages' / 'evaluate.json', (world.run / 'stages' / 'train.json').read_bytes())
    with pytest.raises(Tampered, match='unexpected'):
        Ledger(world.run).resume()


def test_a_leftover_temporary_file_in_the_stage_folder_is_not_a_tamper(world):
    ledger = world.complete()
    put(world.run / 'stages' / f'train.json.{os.getpid()}.deadbeef.tmp', b'{"half": ')
    assert Ledger(world.run).resume() == ['train', 'evaluate'] and ledger.resume() == ['train', 'evaluate']


def test_a_linked_output_is_refused_at_finish_and_at_resume(world, monkeypatch):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    world.work('train')
    target = world.run / 'train' / 'log.txt'
    real = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda self: True if self == target else real(self))
    with pytest.raises(LedgerRefused):
        ledger.finish('train', world.outputs('train'))
    monkeypatch.undo()
    ledger.finish('train', world.outputs('train'))
    monkeypatch.setattr(Path, 'is_symlink', lambda self: True if self == target else real(self))
    with pytest.raises(Tampered):
        Ledger(world.run).resume()


def test_the_messages_carry_stage_names_and_counts_never_paths_or_contents(world):
    ledger = Ledger.create(world.run, make_header())
    secret_output = put(world.run / 'train' / f'{LEAK}.bin', f'{LEAK} contents'.encode())
    secret_input = put(world.root / f'{LEAK}-input.bin', f'{LEAK} input'.encode())
    ledger.begin('train', inputs=[secret_input])
    ledger.finish('train', [secret_output])
    messages = {}
    secret_output.write_bytes(f'{LEAK} changed'.encode())
    with pytest.raises(Tampered) as info:
        Ledger(world.run).resume()
    messages['output'] = str(info.value)
    secret_output.write_bytes(f'{LEAK} contents'.encode())
    secret_input.write_bytes(f'{LEAK} changed'.encode())
    with pytest.raises(Tampered) as info:
        Ledger(world.run).resume()
    messages['input'] = str(info.value)
    secret_input.write_bytes(f'{LEAK} input'.encode())
    reopened = Ledger(world.run)
    reopened.resume()
    with pytest.raises(LedgerRefused) as info:
        reopened.begin('evaluate', inputs=[world.root / f'{LEAK}-missing.bin'])
    messages['missing input'] = str(info.value)
    with pytest.raises(LedgerRefused) as info:
        reopened.begin('evaluate', inputs=[f'../{LEAK}.bin'])
    messages['bad path'] = str(info.value)
    with pytest.raises(LedgerRefused) as info:
        reopened.begin('evaluate', inputs=[f'HEADER.json'])
    messages['reserved path'] = str(info.value)
    with pytest.raises(LedgerRefused) as info:
        Ledger.create(world.run, make_header())
    messages['second create'] = str(info.value)
    for kind, message in messages.items():
        assert LEAK not in message and str(world.root) not in message and 'contents' not in message, (kind, message)
    assert "stage 'train'" in messages['output'] and '1 of 1 output' in messages['output']
    assert "stage 'train'" in messages['input'] and '1 of 1 input' in messages['input']
    assert Ledger(world.run).resume() == ['train']                                    # with the files back, the run is accepted again


def test_a_count_of_changed_files_is_in_the_message(world):
    world.complete()
    flip(world.run / 'train' / 'best.bin')
    flip(world.run / 'train' / 'log.txt')
    with pytest.raises(Tampered) as info:
        Ledger(world.run).resume()
    assert "stage 'train'" in str(info.value) and '2 of 2' in str(info.value)


def test_a_malformed_record_is_refused_without_echoing_it(world):
    world.complete()
    (world.run / 'LEDGER.json').write_bytes(('{"schema": "' + LEAK + '", "stages": [}').encode())
    with pytest.raises(Tampered) as info:
        Ledger(world.run)
    assert LEAK not in str(info.value) and info.value.__context__ is None and info.value.__cause__ is None


# ---- the spec check on resume (K5a review I1, ruling (a): structural append-only at the level of decision blocks) -------------------------------
# The decision record that ``guards.spec_check`` reads is one JSON object: ``operating_point``, other top-level members, and ``decisions``, an object whose
# members are the decision blocks (``run_s3.read_decisions`` reads ``doc['decisions']['S3_prices']``). A new decision is a new member of ``decisions``,
# inserted before a closing brace, so a byte-prefix rule cannot accept it. The header keeps one hash for every top-level member outside ``decisions``
# (K5b ruling R1: a new top-level section such as ``G2_time_base`` also grew the record) and one for every decision block; a resume accepts the record
# when every stored hash still holds and may add top-level members and blocks.
def spec_with_new_block(text: bytes) -> bytes:
    """The record text with one more member in ``decisions``, inserted before its closing brace (the last member of the record is not ``decisions``)."""
    needle = b'\n  },\n  "G2_time_base"'
    assert text.count(needle) == 1
    return text.replace(needle, b',\n    "S3_gate_c": {\n      "note": "a later decision"\n    }\n  },\n  "G2_time_base"')


def edited_doc(change) -> bytes:
    doc = spec_doc()
    change(doc)
    return dump_spec(doc)


def test_spec_fingerprint_hashes_the_bytes_every_top_level_member_and_every_decision_block():
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    doc = spec_doc()
    assert set(fingerprint) == {'spec_sha256', 'spec_top', 'spec_blocks'}
    assert fingerprint['spec_sha256'] == sha(SPEC)
    assert fingerprint['spec_top'] == {key: run_io.canonical_sha256(value) for key, value in doc.items() if key != 'decisions'}
    assert sorted(fingerprint['spec_top']) == ['G2_time_base', 'data', 'held_out_policy', 'operating_point']      # not 'decisions': its members are the blocks
    assert fingerprint['spec_blocks'] == {name: run_io.canonical_sha256(block) for name, block in doc['decisions'].items()}
    assert sorted(fingerprint['spec_blocks']) == ['C9', 'S2_gate', 'S3_prices']
    assert fingerprint['spec_blocks']['S3_prices'] == sha(('{"prices":{"sha256":"%s"},"rules":{"sha256":"%s"}}' % ('ab' * 32, 'cd' * 32)).encode())
    literal = '{"chosen":"rule","why":"한글 이유"}'.encode('utf-8')                    # the canonical form: sorted keys, no spaces, UTF-8 not escaped
    assert fingerprint['spec_blocks']['C9'] == sha(literal)
    assert make_header().spec_blocks == fingerprint['spec_blocks']


def test_spec_fingerprint_without_decisions_has_no_blocks_and_hashes_the_whole_record():
    fingerprint = run_ledger.spec_fingerprint(b'{"operating_point": {"G_ms": 1}}')
    assert fingerprint['spec_blocks'] == {} and fingerprint['spec_top'] == {'operating_point': run_io.canonical_sha256({'G_ms': 1})}


@pytest.mark.parametrize('record', [b'', b'[]', b'3', b'{"a": ', b'{"a": 1, "a": 2}', b'{"a": NaN}', b'\xef\xbb\xbf{}', b'{"decisions": []}', b'{"decisions": 3}',
                                    b'{"decisions": null}', b'{"decisions": "text"}'])
def test_spec_fingerprint_refuses_a_record_that_is_not_an_object_with_an_object_of_decisions(record):
    with pytest.raises(guards.GateRefused) as info:
        run_ledger.spec_fingerprint(record)
    assert info.value.gate == 'spec' and info.value.__context__ is None and info.value.__cause__ is None


def test_spec_fingerprint_needs_bytes():
    for bad in ('text', None, 5, [b'{}']):
        with pytest.raises(TypeError):
            run_ledger.spec_fingerprint(bad)


def test_an_equal_record_and_a_reformatted_one_are_accepted():
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    check = lambda current: run_ledger.check_spec_growth(current, top=fingerprint['spec_top'], blocks=fingerprint['spec_blocks'])
    check(SPEC)
    check(dump_spec(spec_doc(), indent=4))                                           # whitespace only
    check(json.dumps(spec_doc(), separators=(',', ':'), ensure_ascii=True).encode('utf-8'))                  # compact, escaped non-ASCII
    reordered = {key: spec_doc()[key] for key in reversed(list(spec_doc()))}         # another member order at the top
    reordered['decisions'] = {key: reordered['decisions'][key] for key in reversed(list(reordered['decisions']))}
    check(dump_spec(reordered))
    check(SPEC + b'\n\n')


def test_a_new_decision_inserted_before_the_closing_brace_is_accepted_by_both_gates_and_the_ledger(tmp_path, world):
    """The realistic growth: a new member of ``decisions``, written in the record's own format. The old byte-prefix rule refused it."""
    from c_ept import config
    point = {'G_ms': config.G_MS, 'L_u': config.L_U, 'tau_lead_ms': config.TAU_LEAD_MS, 'presence_gate': None, 'size_classes': {'T': 1, 'S': 2, 'ASYM': 3, 'P': 4}}
    original = dump_spec(spec_doc(point))
    grown = spec_with_new_block(original)
    assert grown != original and json.loads(grown)['decisions']['S3_gate_c'] == {'note': 'a later decision'} and len(json.loads(grown)['decisions']) == 4
    record = tmp_path / 'SPEC.json'
    record.write_bytes(original)
    digest = guards.spec_check(record)
    header = make_header(**run_ledger.spec_fingerprint(original))
    assert header.spec_sha256 == digest                                               # the header keeps the hash that spec_check returned
    ledger = Ledger.create(world.run, header)
    ledger.check_spec(original)
    record.write_bytes(grown)
    assert guards.spec_check(record) != digest                                        # spec_check accepts the grown record (it returns its new hash) ...
    ledger.check_spec(grown)                                                          # ... and so does the ledger
    assert Ledger(world.run).header.spec_sha256 == digest


def test_an_edited_block_a_removed_block_and_a_changed_or_removed_top_level_member_are_refused_with_the_gate_spec(world):
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    ledger = Ledger.create(world.run, make_header())
    cases = {
        'edited block': edited_doc(lambda d: d['decisions']['C9'].update(chosen='other')),
        'edited nested value': edited_doc(lambda d: d['decisions']['S2_gate']['S2G-1'].update(cap=129)),
        'edited block to a number': edited_doc(lambda d: d['decisions'].update(C9=7)),
        'removed block': edited_doc(lambda d: d['decisions'].pop('S2_gate')),
        'all blocks removed': edited_doc(lambda d: d.update(decisions={})),
        'decisions removed': edited_doc(lambda d: d.pop('decisions')),
        'a block renamed': edited_doc(lambda d: d['decisions'].update({'C9_renamed': d['decisions'].pop('C9')})),
        'changed top-level field': edited_doc(lambda d: d['data'].update(matches=4)),
        'changed operating point': edited_doc(lambda d: d['operating_point'].update(G_ms=2)),
        'removed top-level field': edited_doc(lambda d: d.pop('held_out_policy')),
        'removed top-level section that grew after the decisions': edited_doc(lambda d: d.pop('G2_time_base')),
        'changed top-level section that grew after the decisions': edited_doc(lambda d: d['G2_time_base'].update(unit='ms')),
        'a top-level field renamed': edited_doc(lambda d: d.update(data_renamed=d.pop('data'))),
        'a number as another type': edited_doc(lambda d: d['data'].update(matches='3')),
        '1 as 1.0 inside a block': edited_doc(lambda d: d['decisions']['S2_gate']['S2G-1'].update(cap=128.0)),
    }
    for name, current in cases.items():
        for refuse in (lambda: ledger.check_spec(current),
                       lambda: run_ledger.check_spec_growth(current, top=fingerprint['spec_top'], blocks=fingerprint['spec_blocks'])):
            with pytest.raises(guards.GateRefused) as info:
                refuse()
            assert info.value.gate == 'spec' and isinstance(info.value, RuntimeError), name
            assert info.value.__context__ is None and info.value.__cause__ is None, name
    ledger.check_spec(SPEC)                                                           # the unedited record is still accepted


def test_new_top_level_sections_are_accepted_next_to_new_blocks_and_every_stored_member_must_stay(world):
    """K5b ruling R1: the real record also grew by whole top-level sections (for example ``G2_time_base``), so a new top-level key is accepted, as a new
    block is; a stored key may not change and may not disappear."""
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    ledger = Ledger.create(world.run, make_header())
    check = lambda current: run_ledger.check_spec_growth(current, top=fingerprint['spec_top'], blocks=fingerprint['spec_blocks'])
    for current in (edited_doc(lambda d: d.update(new_section={'a': [1, 2]})),
                    edited_doc(lambda d: d.update(new_section=None, another=3)),
                    edited_doc(lambda d: (d.update(S4b_time_base={'unit': 'ms'}), d['decisions'].update(D21={'rule': 'x'})))):
        check(current)
        ledger.check_spec(current)
    original = spec_doc()
    stored = {key: value for key, value in original.items() if key != 'decisions'}
    for key in stored:                                                                 # every stored top-level key is protected, one by one
        for change, reason in ((lambda d: d.pop(key), 'removed'), (lambda d: d.update({key: ['changed']}), 'changed')):
            current = edited_doc(change)
            with pytest.raises(guards.GateRefused) as info:
                check(current)
            assert info.value.gate == 'spec' and reason in info.value.reason and key not in str(info.value), (key, reason)
    assert Ledger(world.run).header.spec_top == fingerprint['spec_top']              # the header keeps one hash per stored top-level key


def test_each_refusal_of_the_spec_rule_has_its_own_fixed_reason():
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    arguments = dict(top=fingerprint['spec_top'], blocks=fingerprint['spec_blocks'])
    reasons = {}
    for name, current in {'outside': edited_doc(lambda d: d['data'].update(matches=4)), 'removed': edited_doc(lambda d: d['decisions'].pop('C9')),
                          'removed top': edited_doc(lambda d: d.pop('data')),
                          'changed': edited_doc(lambda d: d['decisions']['C9'].update(chosen='other')), 'unreadable': b'{"a": ',
                          'not decisions': b'{"decisions": 3}'}.items():
        with pytest.raises(guards.GateRefused) as info:
            run_ledger.check_spec_growth(current, **arguments)
        reasons[name] = info.value.reason
    assert len(set(reasons.values())) == 6
    assert 'changed outside' in reasons['outside'] and 'removed outside' in reasons['removed top']
    assert 'missing' in reasons['removed'] and 'differs' in reasons['changed'] and 'unreadable' in reasons['unreadable']


def test_the_spec_refusals_carry_neither_the_record_nor_a_hash_nor_a_block_name():
    fingerprint = run_ledger.spec_fingerprint(SPEC)
    doc = spec_doc()
    doc['decisions'][LEAK] = {'secret': LEAK}
    doc['decisions']['C9'] = {'secret': LEAK}
    doc['data'] = {'secret': LEAK}
    removed = spec_doc()
    removed['decisions'].pop('C9')
    removed['decisions'][LEAK] = {'secret': LEAK}
    for current in (dump_spec(doc), dump_spec(removed), json.dumps({'secret': LEAK}).encode(), b'{"secret": "' + LEAK.encode() + b'", '):
        with pytest.raises(guards.GateRefused) as info:
            run_ledger.check_spec_growth(current, top=fingerprint['spec_top'], blocks=fingerprint['spec_blocks'])
        text = str(info.value)
        assert LEAK not in text and not any(h in text for h in fingerprint['spec_top'].values()) and 'C9' not in text and 'S3_prices' not in text
        assert 'held_out_policy' not in text and 'G2_time_base' not in text and 'operating_point' not in text


@pytest.mark.parametrize('arguments', [dict(top=None, blocks={}), dict(top=[('a', 'ab' * 32)], blocks={}), dict(top='text', blocks={}),
                                       dict(top={'a': 'x'}, blocks={}), dict(top={'a': None}, blocks={}), dict(top={1: 'ab' * 32}, blocks={}),
                                       dict(top={'a': 'AB' * 32}, blocks={}), dict(top={'a': 'ab' * 32 + 'x'}, blocks={}),
                                       dict(top={'decisions': 'ab' * 32}, blocks={}),                                  # decisions is never a top-level member
                                       dict(top={}, blocks=None), dict(top={}, blocks=[('a', 'ab' * 32)]), dict(top={}, blocks={'a': 'x'}),
                                       dict(top={}, blocks={'a': None}), dict(top={}, blocks={1: 'ab' * 32})])
def test_the_spec_rule_refuses_malformed_stored_values(arguments):
    with pytest.raises(ValueError):
        run_ledger.check_spec_growth(SPEC, **arguments)
    with pytest.raises(TypeError):
        run_ledger.check_spec_growth('text', top={}, blocks={})


def test_the_byte_prefix_rule_is_gone():
    assert not hasattr(run_ledger, 'BYTE_PREFIX')
    source = LEDGER_SOURCE.read_text(encoding='utf-8')
    assert 'current[:length]' not in source and 'spec_length' not in source


def test_the_ledger_checks_the_current_record_against_the_header(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.check_spec(SPEC)
    ledger.check_spec(spec_with_new_block(SPEC))
    with pytest.raises(guards.GateRefused):
        ledger.check_spec(SPEC.replace(b'"rule"', b'"other"'))
    reopened = Ledger(world.run)
    with pytest.raises(guards.GateRefused):
        reopened.check_spec(SPEC[:-1])                                                # a truncated record is not JSON
    reopened.check_spec(spec_with_new_block(SPEC))                                    # check_spec needs no resume: it reads only the header


def real_spec(tmp_path, change=None) -> Path:
    """A synthetic decision record whose operating point equals the constants of ``c_ept.config`` (so that ``guards.spec_check`` accepts it)."""
    from c_ept import config
    point = {'G_ms': config.G_MS, 'L_u': config.L_U, 'tau_lead_ms': config.TAU_LEAD_MS, 'presence_gate': None, 'size_classes': {'T': 1, 'S': 2, 'ASYM': 3, 'P': 4}}
    doc = spec_doc(point)
    if change is not None:
        change(doc)
    return put(tmp_path / 'SPEC.json', dump_spec(doc))


def test_check_spec_file_reads_the_record_once_and_runs_both_gates_on_those_bytes(world, tmp_path, monkeypatch):
    """K5a review I1: the resume rule is "passes ``spec_check`` and grew only by blocks". One call, one read, one hash that binds the two."""
    record = real_spec(tmp_path)
    ledger = Ledger.create(world.run, make_header(**run_ledger.spec_fingerprint(record.read_bytes())))
    seen = []
    real = guards.spec_check
    monkeypatch.setattr(guards, 'spec_check', lambda path, *, expected_sha256=None: (seen.append(expected_sha256), real(path, expected_sha256=expected_sha256))[1])
    assert ledger.check_spec_file(record) == sha(record.read_bytes()) == ledger.header.spec_sha256
    assert seen == [sha(record.read_bytes())]                                         # spec_check was asked for the hash of the bytes that were read
    grown = spec_with_new_block(record.read_bytes())
    record.write_bytes(grown)
    assert ledger.check_spec_file(str(record)) == sha(grown) != ledger.header.spec_sha256        # a new decision: accepted by both, the new hash is returned
    assert seen[-1] == sha(grown)


def test_check_spec_file_refuses_what_either_gate_refuses(world, tmp_path):
    record = real_spec(tmp_path)
    ledger = Ledger.create(world.run, make_header(**run_ledger.spec_fingerprint(record.read_bytes())))
    makers = {
        'edited block': lambda: real_spec(tmp_path, lambda d: d['decisions']['C9'].update(chosen='other')),
        'removed block': lambda: real_spec(tmp_path, lambda d: d['decisions'].pop('C9')),
        'changed outside': lambda: real_spec(tmp_path, lambda d: d['data'].update(matches=9)),
        'constants differ from the code': lambda: real_spec(tmp_path, lambda d: d['operating_point'].update(G_ms=-1)),     # spec_check refuses it
        'not a record': lambda: put(tmp_path / 'SPEC.json', b'not json ' + LEAK.encode()),
    }
    for name, make in makers.items():
        refused = make()
        with pytest.raises(guards.GateRefused) as info:
            ledger.check_spec_file(refused)
        assert info.value.gate == 'spec' and LEAK not in str(info.value) and str(tmp_path) not in str(info.value), name
        assert info.value.__context__ is None and info.value.__cause__ is None, name
    with pytest.raises(guards.GateRefused, match='unreadable') as info:
        ledger.check_spec_file(tmp_path / 'absent.json')
    assert str(tmp_path) not in str(info.value) and info.value.__context__ is None


def test_check_spec_file_refuses_a_record_that_changes_after_it_was_read(world, tmp_path, monkeypatch):
    record = real_spec(tmp_path)
    ledger = Ledger.create(world.run, make_header(**run_ledger.spec_fingerprint(record.read_bytes())))
    real = guards.spec_check

    def changing(path, *, expected_sha256=None):
        Path(path).write_bytes(spec_with_new_block(Path(path).read_bytes()))          # another process appends a decision between the two reads
        return real(path, expected_sha256=expected_sha256)
    monkeypatch.setattr(guards, 'spec_check', changing)
    with pytest.raises(guards.GateRefused, match='hash mismatch'):
        ledger.check_spec_file(record)


def test_check_spec_file_reads_at_most_the_bound(world, tmp_path, monkeypatch):
    record = real_spec(tmp_path)
    ledger = Ledger.create(world.run, make_header(**run_ledger.spec_fingerprint(record.read_bytes())))
    assert run_ledger.MAX_SPEC_BYTES == 16 << 20
    size = record.stat().st_size
    monkeypatch.setattr(run_ledger, 'MAX_SPEC_BYTES', 10)
    reads = []
    watch_opens(monkeypatch, record, reads=reads)
    with pytest.raises(guards.GateRefused, match='unreadable'):
        ledger.check_spec_file(record)
    assert sum(reads) == 11 < size                                                   # bound + 1 bytes were read, not the whole file
    monkeypatch.undo()
    monkeypatch.setattr(run_ledger, 'MAX_SPEC_BYTES', size - 1)
    with pytest.raises(guards.GateRefused, match='unreadable'):
        ledger.check_spec_file(record)
    monkeypatch.setattr(run_ledger, 'MAX_SPEC_BYTES', size)                          # a record of exactly the bound is read
    ledger.check_spec_file(record)


def test_the_fingerprint_is_bound_to_the_record_that_spec_check_checked(tmp_path):
    """``guards.spec_check`` returns the SHA-256 of the record bytes after the constants check; the header's ``spec_sha256`` is that hash, and the
    fingerprint is taken from the same bytes (one read, bound by ``expected_sha256``)."""
    from c_ept import config
    point = {'G_ms': config.G_MS, 'L_u': config.L_U, 'tau_lead_ms': config.TAU_LEAD_MS, 'presence_gate': None, 'size_classes': {'T': 1, 'S': 2, 'ASYM': 3, 'P': 4}}
    record = tmp_path / 'SPEC.json'
    record.write_bytes(dump_spec(spec_doc(point)))
    current = record.read_bytes()
    digest = guards.spec_check(record, expected_sha256=sha(current))
    assert run_ledger.spec_fingerprint(current)['spec_sha256'] == digest
    record.write_bytes(spec_with_new_block(current))                                  # a record that changed after the read is refused by the hash binding
    with pytest.raises(guards.GateRefused, match='hash mismatch'):
        guards.spec_check(record, expected_sha256=sha(current))


# ---- agreement with the T009 components that the ledger reuses (Ref comments in the module) --------------------------------------------
PATH_ACCEPTED = ['a.bin', 'a/b.bin', 'a/b/c.bin', 'run-1/x_y.bin']
PATH_REFUSED = ['../x.bin', '/x.bin', 'a/../b.bin', 'C:/x.bin', 'a:b.bin', '.', '', 'a//b.bin', 'a\\b.bin', 'a/./b.bin', 'a/', 'a/ ', 'a/b.', '..']


def test_the_path_rule_is_at_least_as_strict_as_the_t009_artifact_rule(world):
    """``comparison.protocol.inside`` (used by the T009 stage receipt) accepts a relative path without ``..``, ``:`` or a leading root. Whatever the
    ledger accepts as an output path, ``inside`` accepts too; the ledger also refuses the empty, ``.``, doubled-slash and backslash forms."""
    from c_ept.comparison import protocol
    ledger = Ledger.create(world.run, make_header())
    for number, case in enumerate(PATH_ACCEPTED + PATH_REFUSED):
        stage = f'check{number}'
        ledger.begin(stage)
        if case in PATH_ACCEPTED:
            put(world.run / case, b'x')
        try:
            ledger.finish(stage, [case])
            accepted_by_ledger = True
        except LedgerRefused:
            accepted_by_ledger = False
            ledger.finish(stage, [])                                               # close the stage again: one stage at a time
        try:
            protocol.inside(world.run, case)
            accepted_by_t009 = True
        except Exception:
            accepted_by_t009 = False
        assert accepted_by_ledger == (case in PATH_ACCEPTED), case
        if accepted_by_ledger:
            assert accepted_by_t009, case


def test_the_stage_file_lists_the_same_artifact_hashes_as_the_t009_receipt(tmp_path):
    from c_ept.comparison import runtime as comparison_runtime
    output = tmp_path / 'campaign'
    (output / 'prepare').mkdir(parents=True)
    files = [put(output / 'inputs.json', b'{"a": 1}'), put(output / 'plan.bin', b'plan-bytes' * 9)]
    receipt = comparison_runtime._write_receipt({'output': str(output)}, 'prepare', None, {}, files)
    run = tmp_path / 'run'
    ledger = Ledger.create(run, make_header())
    put(run / 'inputs.json', b'{"a": 1}')
    put(run / 'plan.bin', b'plan-bytes' * 9)
    ledger.begin('prepare')
    ledger.finish('prepare', ['inputs.json', 'plan.bin'])
    assert run_io.read_json(run / 'stages' / 'prepare.json')['outputs'] == receipt['artifacts']             # same keys (flat names) and same hashes
    # one difference, recorded: the T009 receipt writes ``str(Path)`` keys (a backslash on Windows); the kernel always writes forward slashes.


def test_an_external_edit_is_refused_by_the_t009_campaign_ledger_and_by_this_ledger(tmp_path):
    from c_ept.comparison import infrastructure
    campaign = infrastructure.CampaignLedger(tmp_path / 'campaign' / 'ledger.json')
    edited = (tmp_path / 'campaign' / 'ledger.json')
    edited.write_bytes(edited.read_bytes() + b' ')
    with pytest.raises(infrastructure.CampaignError, match='changed'):
        campaign.snapshot()
    ledger = Ledger.create(tmp_path / 'run', make_header())
    (tmp_path / 'run' / 'LEDGER.json').write_bytes((tmp_path / 'run' / 'LEDGER.json').read_bytes() + b' ')
    with pytest.raises(Tampered, match='changed'):
        ledger.begin('train')


def frozen_run_s3_ledger():
    """The class ``Ledger`` of the frozen S3 module (``workflows/run_s3.py``), compiled alone from its syntax tree with stand-ins for what it uses.

    The module itself is not imported (it is a command module that imports numpy and the models); only the class node is taken.
    """
    import types
    source = REPO / 'c_ept' / 'workflows' / 'run_s3.py'
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef) and n.name == 'Ledger')
    refused = type('RunRefused', (Exception,), {})
    namespace = {'json': json, 'Path': Path, 'SCHEMA': 's3_stages_v1', 'RunRefused': refused, '_now': lambda: 'T',
                 'render': lambda doc: json.dumps(doc, sort_keys=True, indent=2),
                 'store_core': types.SimpleNamespace(write_atomic=lambda path, text: Path(path).write_text(text, encoding='utf-8'))}
    exec(compile(ast.Module([node], []), str(source), 'exec'), namespace)
    return namespace['Ledger'], refused


def test_begin_and_finish_agree_with_the_s3_stage_ledger_they_refer_to(world, tmp_path):
    """K5a review M6. ``# Ref: run_s3.Ledger.begin/finish``: both begin a stage on inputs and begin it again on the same inputs; both refuse to
    begin it again on other inputs; both record the inputs and the outputs of a finished stage. (The kernel also refuses to begin a finished
    stage and records hashes taken from the files, where S3 takes digests from its caller.)"""
    s3_ledger, s3_refused = frozen_run_s3_ledger()
    s3_dir = tmp_path / 's3'
    s3_dir.mkdir()
    s3 = s3_ledger(s3_dir)
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    s3.begin('train', {'profile': sha(PROFILE)}, 'digest-1')
    ledger.begin('train', inputs=world.inputs('train'))                              # begun again on the same inputs: accepted by both
    s3.begin('train', {'profile': sha(PROFILE)}, 'digest-1')
    put(world.profile, PROFILE + b' ')
    with pytest.raises(LedgerRefused, match='other inputs'):                         # begun again on other inputs: refused by both
        ledger.begin('train', inputs=world.inputs('train'))
    with pytest.raises(s3_refused):
        s3.begin('train', {'profile': sha(PROFILE + b' ')}, 'digest-2')
    put(world.profile, PROFILE)
    world.work('train')
    ledger.finish('train', world.outputs('train'))
    s3.finish('train', {'train/best.bin': sha(b'weights-v1|' * 20)})
    recorded, s3_recorded = ledger.stage('train'), s3.stage('train')
    assert recorded.state == 'finished' and s3.complete('train')
    assert set(recorded.inputs) == {world.profile.as_posix()} and set(s3_recorded['inputs']) == {'profile'}      # inputs are recorded by both
    assert set(recorded.outputs) == set(world.outputs('train')) and set(s3_recorded['outputs']) == {'train/best.bin'}   # outputs are recorded by both


# ---- review minors M2-M5, M9 ----------------------------------------------------------------------------------------------------------------
def test_a_ledger_replaced_by_a_folder_after_the_open_is_a_tamper_not_an_os_error(world):
    ledger = Ledger.create(world.run, make_header())
    (world.run / 'LEDGER.json').unlink()
    (world.run / 'LEDGER.json').mkdir()
    for operation in (lambda: ledger.begin('train'), ledger.resume):
        with pytest.raises(Tampered, match='ledger file'):
            operation()


def test_an_os_error_while_hashing_propagates_unchanged_and_a_later_resume_succeeds(world, monkeypatch):
    """A permission error or a sharing violation is not evidence of a change: it is not a ``Tampered`` (K5b maps ``OSError`` to a retryable exit)."""
    world.complete()
    before = world.snapshot()
    real = run_io.sha_file

    def failing(path, **kwargs):
        if Path(path).name == 'best.bin':
            raise PermissionError('denied')
        return real(path, **kwargs)
    monkeypatch.setattr(run_io, 'sha_file', failing)
    with pytest.raises(PermissionError) as info:
        Ledger(world.run).resume()
    assert not isinstance(info.value, run_io.IORefused)
    monkeypatch.undo()
    assert world.snapshot() == before and Ledger(world.run).resume() == ['train', 'evaluate']


@pytest.mark.parametrize('name', ['LEDGER.json', 'HEADER.json'])
def test_an_os_error_while_checking_the_ledger_files_is_not_a_tamper_either(world, monkeypatch, name):
    ledger = Ledger.create(world.run, make_header())
    real = run_io.sha_file

    def failing(path, **kwargs):
        if Path(path).name == name:
            raise PermissionError('sharing violation')
        return real(path, **kwargs)
    monkeypatch.setattr(run_io, 'sha_file', failing)
    for operation in (lambda: ledger.begin('train'), ledger.resume):
        with pytest.raises(PermissionError) as info:
            operation()
        assert not isinstance(info.value, run_io.IORefused)
    monkeypatch.undo()
    ledger.begin('train')
    assert ledger.stages == (('train', 'begun'),)


@pytest.mark.parametrize('name', ['LEDGER.json', 'HEADER.json'])
def test_a_ledger_file_that_vanishes_between_the_check_and_the_read_is_a_tamper(world, monkeypatch, name):
    ledger = Ledger.create(world.run, make_header())
    real = run_io.sha_file

    def vanishing(path, **kwargs):
        if Path(path).name == name:
            raise FileNotFoundError(2, 'removed by another process')
        return real(path, **kwargs)
    monkeypatch.setattr(run_io, 'sha_file', vanishing)
    for operation in (lambda: ledger.begin('train'), ledger.resume):
        with pytest.raises(Tampered, match='missing'):
            operation()


def test_the_header_keeps_its_own_copy_of_the_block_map():
    blocks = dict(run_ledger.spec_fingerprint(SPEC)['spec_blocks'])
    header = make_header(spec_blocks=blocks)
    blocks['injected'] = 'ab' * 32
    blocks['C9'] = '00' * 32
    assert 'injected' not in header.spec_blocks and header.spec_blocks['C9'] != '00' * 32
    assert list(header.spec_blocks) == sorted(header.spec_blocks)                    # sorted by name: one record for one header


def test_the_stage_accessor_gives_the_recorded_inputs_and_outputs_read_only(world):
    """K5a review M3: K6 checks that a finished stage recorded the outputs it expects."""
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    begun = ledger.stage('train')
    assert (begun.stage, begun.state) == ('train', 'begun') and dict(begun.inputs) == {world.profile.as_posix(): sha(PROFILE)} and dict(begun.outputs) == {}
    world.work('train')
    ledger.finish('train', world.outputs('train'))
    done = ledger.stage('train')
    assert (done.stage, done.state) == ('train', 'finished')
    assert dict(done.outputs) == {'train/best.bin': sha(b'weights-v1|' * 20), 'train/log.txt': sha(b'epoch 1 loss 3.5\nepoch 2 loss 3.1\n')}
    assert done.inputs == begun.inputs
    with pytest.raises(TypeError):
        done.outputs['x'] = 'y'
    with pytest.raises(TypeError):
        done.inputs['x'] = 'y'
    with pytest.raises(dataclasses.FrozenInstanceError):
        done.state = 'begun'
    assert Ledger(world.run).stage('train') == done                                   # the same record after a reopen
    with pytest.raises(LedgerRefused, match='not in the ledger'):
        ledger.stage('evaluate')
    with pytest.raises(LedgerRefused):
        ledger.stage('Bad Name')


def test_the_module_docstring_states_what_is_refused_what_is_not_and_the_power_loss_residual():
    """K5a review M1, M3, M9: the claims of the docstring are the ones the tests prove."""
    doc = run_ledger.__doc__
    assert 'external anchor' in doc and 'power loss' in doc and 'creation incomplete' in doc
    assert '앵커' in doc and '전원' in doc
    assert 'canonical' in doc and 'sort_keys=True, separators' in doc                # the canonical JSON of the structural spec rule is spelled out
    assert 'Creation (H, then L) is the exception' in doc and 'begin (L) and finish (F_s, then L)' in doc      # the recovery claim is scoped to begin and finish
    assert 'creation incomplete' in (Ledger.__doc__ or '') and 'Every later stop is recoverable' in Ledger.__doc__
    assert 'every write point' not in doc and 'Every stop' not in doc and 'every stop' not in doc.replace('Every later stop', '')


@pytest.mark.parametrize('make', [lambda: Path('train') / 'best.bin', lambda: PurePosixPath('train/best.bin'), lambda: PureWindowsPath('train\\best.bin')],
                         ids=['Path', 'PurePosixPath', 'PureWindowsPath'])
def test_a_relative_path_object_is_accepted_whatever_its_separator(world, make):
    """K5a review M4: ``os.fspath(Path('a/b'))`` is ``a\\b`` on Windows; a path object is converted with ``as_posix()`` before the checks."""
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train')
    world.work('train')
    ledger.finish('train', [make()])
    assert sorted(ledger.stage('train').outputs) == ['train/best.bin']


def test_an_object_that_is_not_a_path_is_never_turned_into_a_path_by_its_text(world):
    """``5`` or ``True`` is not the file ``5`` or ``True``: only text and path objects name files."""
    ledger = Ledger.create(world.run, make_header())
    for name in ('5', 'True', 'None', '1.5'):
        put(world.run / name, b'x')
    ledger.begin('train')
    for outputs in ([5], [True], [None], [1.5], [b'5'], [('5',)], [{'a': 1}]):
        with pytest.raises(LedgerRefused):
            ledger.finish('train', outputs)
    ledger.finish('train', [])
    for inputs in ([5], [True], [None]):
        with pytest.raises(LedgerRefused):
            ledger.begin('evaluate', inputs=inputs)


def test_a_path_like_object_that_is_not_a_pure_path_is_converted_like_a_path(world):
    class PathLike:
        def __fspath__(self):
            return str(Path('train') / 'best.bin')                                    # a backslash on Windows, a slash elsewhere

    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train')
    world.work('train')
    ledger.finish('train', [PathLike()])
    assert sorted(ledger.stage('train').outputs) == ['train/best.bin']


def test_a_text_with_a_backslash_is_still_refused_and_a_path_object_is_not_a_way_around_the_rules(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train')
    world.work('train')
    for outputs in (['train\\best.bin'], [PurePosixPath('../train/best.bin')], [PureWindowsPath('..\\train\\best.bin')],
                    [PureWindowsPath('train\\best.bin ')], [PureWindowsPath('C:best.bin')], [PurePosixPath('HEADER.json')]):
        with pytest.raises(LedgerRefused):
            ledger.finish('train', outputs)
    assert ledger.stages == (('train', 'begun'),)


@pytest.mark.parametrize('key', ['HEADER.json', 'LEDGER.json', 'header.json', 'Ledger.JSON', 'stages', 'stages/train.json', 'Stages/x.json', 'stages/sub/y'])
def test_the_ledgers_own_files_cannot_be_inputs_or_outputs(world, key):
    """K5a review M5: the next ledger write changes them, so the run would be a tamper from then on."""
    ledger = Ledger.create(world.run, make_header())
    with pytest.raises(LedgerRefused, match='ledger'):
        ledger.begin('train', inputs=[key])
    with pytest.raises(LedgerRefused, match='ledger'):
        ledger.begin('train', inputs=[world.run / key])
    ledger.begin('train')
    for call in (lambda: ledger.finish('train', [key]), lambda: ledger.finish('train', [world.run / key])):
        with pytest.raises(LedgerRefused, match='ledger'):
            call()
    ledger.finish('train', [])
    assert ledger.stages == (('train', 'finished'),)


def test_a_file_with_the_name_of_a_ledger_file_deeper_in_the_run_folder_is_an_ordinary_file(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train')
    put(world.run / 'train' / 'LEDGER.json', b'a model card')
    put(world.run / 'train' / 'stages' / 'x', b'x')
    ledger.finish('train', ['train/LEDGER.json', 'train/stages/x'])
    assert sorted(ledger.stage('train').outputs) == ['train/LEDGER.json', 'train/stages/x']


# ---- K5b: a place for stop reasons and E6 records (notes), and the stop/budget ``check`` that the hashing honours ---------------------------------
# The supervisor records why a stage stopped (budget, failure, refusal) and the E6 check records in the ledger. A note is a small strict-JSON record that
# belongs to a stage name (or to the start of the run, ``None``); it is written into the ledger file, which has one owner. A ledger without notes has
# exactly the bytes of K5a (no ``notes`` member), so the K5a guarantees stay as they were.
def note_world(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    return ledger


def test_notes_hold_a_stop_reason_and_an_e6_record_and_survive_a_reopen(world):
    ledger = note_world(world)
    assert ledger.notes == ()
    ledger.note(None, 'e6', {'running': False, 'matched': [], 'scanned': 7})
    ledger.note('train', 'stop', {'reason': 'budget', 'detail': 'wall-clock budget exceeded'})
    expected = (run_ledger.Note(stage=None, kind='e6', record={'running': False, 'matched': (), 'scanned': 7}),
                run_ledger.Note(stage='train', kind='stop', record={'reason': 'budget', 'detail': 'wall-clock budget exceeded'}))
    assert ledger.notes == expected
    reopened = Ledger(world.run)
    assert reopened.notes == expected and reopened.stages == (('train', 'begun'),)
    doc = run_io.read_json(world.run / 'LEDGER.json')
    assert doc['notes'] == [{'stage': None, 'kind': 'e6', 'record': {'running': False, 'matched': [], 'scanned': 7}},
                            {'stage': 'train', 'kind': 'stop', 'record': {'reason': 'budget', 'detail': 'wall-clock budget exceeded'}}]
    with pytest.raises(TypeError):
        ledger.notes[0].record['scanned'] = 8                                       # read-only at every depth
    assert run_ledger.NOTE_KINDS == ('stop', 'e6')


def test_a_ledger_without_notes_has_no_notes_member_and_the_k5a_bytes(world):
    ledger = world.complete()
    doc = run_io.read_json(world.run / 'LEDGER.json')
    assert set(doc) == {'schema', 'header_sha256', 'stages'} and ledger.notes == ()
    reopened = Ledger(world.run)
    assert reopened.resume() == list(STAGES) and reopened.notes == ()


def test_notes_do_not_disturb_the_stage_writes_the_chain_or_the_resume(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.note(None, 'e6', {'running': False})
    for stage in STAGES:
        ledger.note(stage, 'e6', {'running': False, 'before': stage})
        world.play(ledger, stage)
    ledger.note('evaluate', 'stop', {'reason': 'failure'})
    reopened = Ledger(world.run)
    assert reopened.resume() == list(STAGES) and len(reopened.notes) == 4
    assert [n.stage for n in reopened.notes] == [None, 'train', 'evaluate', 'evaluate']
    # the completion after a stop between the two writes of finish keeps the notes, too
    other = World(world.root / 'second')
    led = Ledger.create(other.run, make_header())
    led.note('train', 'e6', {'running': False})
    led.begin('train', inputs=other.inputs('train'))
    other.work('train')
    real_commit = led._commit
    led._commit = lambda entries: (_ for _ in ()).throw(Crash())                      # the stage file is written, the ledger is not
    with pytest.raises(Crash):
        led.finish('train', other.outputs('train'))
    led._commit = real_commit
    again = Ledger(other.run)
    assert again.resume() == ['train'] and [n.kind for n in again.notes] == ['e6']


@pytest.mark.parametrize('call', [
    lambda l: l.note('Train', 'stop', {}), lambda l: l.note('', 'stop', {}), lambda l: l.note(5, 'stop', {}), lambda l: l.note('con', 'stop', {}),
    lambda l: l.note('train', 'other', {}), lambda l: l.note('train', None, {}), lambda l: l.note('train', 'stop', []), lambda l: l.note('train', 'stop', 'text'),
    lambda l: l.note('train', 'stop', None), lambda l: l.note('train', 'stop', {'x': float('nan')}), lambda l: l.note('train', 'stop', {'x': object()}),
    lambda l: l.note('train', 'stop', {1: 'a'}), lambda l: l.note('train', 'stop', {'x': {2: 'a'}})])
def test_a_note_that_is_not_acceptable_is_refused_and_nothing_is_written(world, call):
    ledger = note_world(world)
    before = world.snapshot()
    with pytest.raises((LedgerRefused, run_io.IORefused)):
        call(ledger)
    assert world.snapshot() == before and ledger.notes == ()


def test_a_note_needs_a_resumed_ledger_and_the_owner_bytes(world):
    world.complete()
    opened = Ledger(world.run)
    with pytest.raises(LedgerRefused, match='resume'):
        opened.note('train', 'stop', {'reason': 'x'})
    opened.resume()
    opened.note('train', 'stop', {'reason': 'x'})
    stale = Ledger(world.run)
    stale.resume()
    opened.note('train', 'stop', {'reason': 'y'})                                    # the file now differs from what ``stale`` last wrote
    with pytest.raises(Tampered):
        stale.note('train', 'stop', {'reason': 'z'})


def test_the_number_of_notes_is_bounded(world, monkeypatch):
    assert run_ledger.MAX_NOTES >= 1000
    monkeypatch.setattr(run_ledger, 'MAX_NOTES', 3)
    ledger = note_world(world)
    for number in range(3):
        ledger.note('train', 'e6', {'n': number})
    before = world.snapshot()
    with pytest.raises(LedgerRefused, match='notes'):
        ledger.note('train', 'e6', {'n': 3})
    assert world.snapshot() == before and len(ledger.notes) == 3


@pytest.mark.parametrize('notes', [
    'text', {}, [],                                                   # [] is a second byte form of "no notes": K5b review m6
    [None], [[]], [{'stage': 'train', 'kind': 'stop'}], [{'stage': 'train', 'kind': 'stop', 'record': {}, 'extra': 1}],
    [{'stage': 'train', 'kind': 'bogus', 'record': {}}], [{'stage': 'Train', 'kind': 'stop', 'record': {}}], [{'stage': 5, 'kind': 'stop', 'record': {}}],
    [{'stage': 'train', 'kind': 'stop', 'record': []}], [{'stage': 'train', 'kind': 'stop', 'record': 'text'}]])
def test_a_ledger_whose_notes_are_malformed_is_a_tamper_when_it_is_opened(world, notes):
    world.complete()
    edit_json(world.run / 'LEDGER.json', lambda doc: doc.update(notes=notes))
    with pytest.raises(Tampered):
        Ledger(world.run)


def test_notes_are_part_of_the_owner_check_a_changed_note_is_found(world):
    ledger = note_world(world)
    ledger.note('train', 'stop', {'reason': 'budget'})
    edit_json(world.run / 'LEDGER.json', lambda doc: doc['notes'][0]['record'].update(reason='nothing'))
    with pytest.raises(Tampered):
        ledger.note('train', 'stop', {'reason': 'again'})


class Halt(Exception):
    """The exception that a stop or budget check raises."""


def test_the_check_is_called_while_the_ledger_hashes_and_what_it_raises_stops_the_operation(world):
    ledger = Ledger.create(world.run, make_header())
    assert ledger.check is None
    calls = []
    ledger.check = lambda: calls.append(1)
    ledger.begin('train', inputs=world.inputs('train'))
    after_begin = len(calls)
    assert after_begin >= 4                                                           # ledger and header (owner check), and the input
    world.work('train')
    ledger.finish('train', world.outputs('train'))
    assert len(calls) > after_begin + 6                                               # the owner check, the inputs again and the outputs
    before = len(calls)
    assert Ledger(world.run).resume() == ['train'] and len(calls) == before          # a ledger has its own ``check``: the new object has none
    again = Ledger(world.run)
    again.check = lambda: calls.append(1)
    again.resume()
    assert len(calls) > before + 4                                                    # resume hashes every recorded file, honouring the check
    again.check = None
    assert again.check is None
    with pytest.raises(TypeError):
        again.check = 'not callable'


@pytest.mark.parametrize('where', ['begin', 'finish', 'resume', 'owner check'])
def test_a_check_that_raises_stops_the_hashing_of_every_operation_without_a_write(world, where):
    ledger = Ledger.create(world.run, make_header())
    if where != 'begin':
        ledger.begin('train', inputs=world.inputs('train'))
        world.work('train')
    if where == 'resume':
        ledger.finish('train', world.outputs('train'))
    before = world.snapshot()
    calls = []
    threshold = {'begin': 8, 'finish': 8, 'resume': 8, 'owner check': 1}[where]       # the owner check hashes two small files first (6 calls)

    def halt():
        calls.append(1)
        if len(calls) >= threshold:
            raise Halt
    ledger.check = halt
    operation = {'begin': lambda: ledger.begin('train', inputs=world.inputs('train')), 'finish': lambda: ledger.finish('train', world.outputs('train')),
                 'resume': ledger.resume, 'owner check': lambda: ledger.note('train', 'stop', {'reason': 'x'})}[where]
    with pytest.raises(Halt):
        operation()
    assert len(calls) == threshold                                                    # the stop came from inside the hashing, at the call that raised
    assert world.snapshot() == before                                                 # nothing was written, no half-written stage file
    ledger.check = None
    if where == 'begin':
        ledger.begin('train', inputs=world.inputs('train'))                           # the same operation goes through once the stop is gone


def test_a_check_can_stop_the_hash_of_a_large_output_between_two_chunks(world):
    ledger = Ledger.create(world.run, make_header())
    ledger.begin('train', inputs=world.inputs('train'))
    put(world.run / 'train' / 'big.bin', b'x' * (3 * (1 << 20) + 5))                 # four 1 MiB reads
    before = world.snapshot()
    calls = []

    def halt_in_the_big_file():
        calls.append(1)
        if len(calls) == 12:                                    # the owner check (6) and the small input (3) come first; then 1 + 4 + 1 calls for big.bin
            raise Halt
    ledger.check = halt_in_the_big_file
    with pytest.raises(Halt):
        ledger.finish('train', ['train/big.bin'])
    assert world.snapshot() == before and len(calls) == 12            # stopped after two of the four chunks
    ledger.check = None
    ledger.finish('train', ['train/big.bin'])


def test_check_spec_file_accepts_a_new_top_level_section_and_refuses_a_changed_one(world, tmp_path):
    record = real_spec(tmp_path)
    ledger = Ledger.create(world.run, make_header(**run_ledger.spec_fingerprint(record.read_bytes())))
    real_spec(tmp_path, lambda d: d.update(S4b_time_base={'unit': 'ms'}))
    assert ledger.check_spec_file(tmp_path / 'SPEC.json') == sha((tmp_path / 'SPEC.json').read_bytes())
    real_spec(tmp_path, lambda d: d['G2_time_base'].update(unit='ms'))
    with pytest.raises(guards.GateRefused, match='changed outside'):
        ledger.check_spec_file(tmp_path / 'SPEC.json')


def test_check_paths_applies_the_rules_of_begin_and_finish_without_a_ledger_and_without_touching_the_disk(tmp_path):
    """K5b: the supervisor checks the paths of its whole plan before it reserves a run folder, with the rule that ``begin`` and ``finish`` will apply."""
    run = tmp_path / 'run'
    run_ledger.check_paths(run, inputs=[tmp_path / 'profile.json', 'a/b.bin', Path('x/y.bin'), run / 'inside.bin'], outputs=['train/best.bin', run / 'z.bin'])
    run_ledger.check_paths(run)
    assert not run.exists() and not (tmp_path / 'profile.json').exists()                  # nothing is created, nothing has to exist
    for inputs in (['../x.bin'], ['a\\b.bin'], [5], 'text', [None], ['a', 'a'], ['HEADER.json'], ['stages/x.json'], [''], ['a//b']):
        with pytest.raises(LedgerRefused):
            run_ledger.check_paths(run, inputs=inputs)
    for outputs in (['../x.bin'], ['/abs.bin'], [tmp_path / 'elsewhere.bin'], ['LEDGER.json'], ['stages'], ['a', 'a'], [5], 'text', [None]):
        with pytest.raises(LedgerRefused):
            run_ledger.check_paths(run, outputs=outputs)
    assert run_ledger.valid_stage_name('train') and not run_ledger.valid_stage_name('Train') and not run_ledger.valid_stage_name('con')
