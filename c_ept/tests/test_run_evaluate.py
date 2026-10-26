"""Tests of the evaluation runner of the run kernel (``c_ept.run.evaluate``; S4b-prep K6b, decisions D4, D13, D15, D16, D20, D24, D27).

The evaluation runner turns a hash-locked evaluate profile into the evaluation of trained models: it reads the training identity of every checkpoint from the
ledger and the record of its training run (K6a), refuses what does not fit together (a mixture of settings, representatives of the gold bins that differ from the
training's, loss scales of another condition, a matched B1 whose dropped columns differ), evaluates the whole STOP split with each fit's best checkpoint (D13),
makes the C4 statistic (three seed values, paired, a match bootstrap on the first rows of the common matrix), and keeps the DEV access ledger and the TEST gate
(D4, D20).

These tests use the toy EPT world of ``ept_toys`` (eight matches), real tiny trainings of K6a on it, hand-made ledgers, ``tmp_path`` folders and short CPU children
that see ``CUDA_VISIBLE_DEVICES=''``. No real data, no GPU, no model server, and no TEST data exists or is read: the TEST gate is tested with a fake backend. The
inference of the DEV adapter (``run_inference``) is tested in ``tests/test_ept_dev_adapters.py``; here a backend that is handed in answers for it.

해석: 이 시험은 평가 프로파일 한 장이 체크포인트의 학습 정체성 읽기, 섞임 거부, STOP 전체의 최선 체크포인트 평가, C4 통계, DEV 열람 원장, TEST 관문으로 이어지는 길을
장난감 자료와 가짜 추론 뒤꿈치로 확인한다. TEST 자료는 없고 읽지도 않는다.
"""
import copy
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept import config as C
from c_ept.ept import checkpoints as ept_checkpoints
from c_ept.ept import data, loss
from c_ept.ept import train as ept_train
from c_ept.evaluation import guard as eval_guard
from c_ept.run import evaluate as run_evaluate
from c_ept.run import io as run_io
from c_ept.run import ledger as run_ledger
from c_ept.run import numeric as run_numeric
from c_ept.run import profile as run_profile
from c_ept.run import runners
from c_ept.run.identity import Identity
from c_ept.tests import run_evaluate_bodies as bodies
from c_ept.tests.ept_toys import build_ept_world, toy_static
from c_ept.valuation import value

KEY = 'KR_1234567890'                                                 # a match key: <platform>_<game id>
SCALES = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}
S4B = {'kind': 'train', 'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'swap_mode': 'random_view', 'on_wall_cap': 'best_stop',
       'column_drop': 'store_and_subset', 'stats_population': 'selected_fit', 'selection_seed': 20261022, 'epochs': 2, 'patience': 5}
SEEDS = (20261022, 20261023, 20261024)
CPU = runners.RunnerConstants(device='cpu', threads=1)
FAKE_BODY = {'module': 'c_ept.tests.test_run_evaluate', 'name': 'fake_stage_body', 'file_sha256': 'e' * 64}              # the stage body that a direct call of ``evaluate`` names (K6c)
FAKE = {'loader': 'c_ept.tests.test_run_evaluate:fake_loader', 'manifest_sha256': 'b' * 64, 'allow_custom': True, 'body': FAKE_BODY}      # what an evaluation with a made-up loader and backend says about itself
E6_RECORD = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 5, 'cmdline_unreadable': 0, 'rule': 'test rule'}


def evaluate_profile(tmp_path, name='evaluate.json', **fields):
    """A loaded evaluate profile: the file is written with LF line ends (the lock is the hash of its bytes)."""
    path = tmp_path / name
    path.write_bytes((json.dumps({'kind': 'evaluate', **fields}, indent=2) + '\n').encode('utf-8'))
    return run_profile.load(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class FakeClock:
    """A clock that a test advances by hand; every call returns the next whole second."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        self.now += 1.0
        return self.now


# ---- the profile without its role ---------------------------------------------------------------------------------------------------------------
def test_the_profile_hash_without_the_role_is_the_same_for_dev_and_test_and_differs_for_any_other_field(tmp_path):
    """DEV rehearsal and TEST differ in ``role`` (so their files hash differently), nothing else may differ."""
    models = ['seed1', 'seed2', 'seed3']
    checkpoints = {name: str(tmp_path / name) for name in models}
    lock = 'ab' * 32                                                                                    # a TEST profile fixes the S5 lock; the DEV rehearsal carries the same one (D28)
    dev = evaluate_profile(tmp_path, 'dev.json', role='DEV', models=models, checkpoints=checkpoints, holdout_lock_sha256=lock)
    test = evaluate_profile(tmp_path, 'test.json', role='TEST', models=models, checkpoints=checkpoints, holdout_lock_sha256=lock)
    assert dev.sha256 != test.sha256
    assert run_evaluate.profile_without_role_sha256(dev) == run_evaluate.profile_without_role_sha256(test)
    expected = run_io.canonical_sha256({key: value for key, value in dev.as_dict().items() if key not in ('role', 'composed')})       # K6e: a null composed block is left out (the hash of every earlier profile)
    assert dev['composed'] is None and run_evaluate.profile_without_role_sha256(dev) == expected
    assert 'composed' not in run_evaluate.profile_record(dev) and run_evaluate.profile_record(dev) == {key: value for key, value in dev.as_dict().items() if key != 'composed'}
    for change in ({'exposure': 'nominal'}, {'winner_truth': 'binned'}, {'bootstrap_B': 1000}, {'b0': True}, {'models': ['seed1', 'seed2']}, {'holdout_lock_sha256': 'cd' * 32},
                   {'composed': {'sample_matches': 3, 'sample_seed': 1, 'draws': 8, 'mc_batches': 4, 'draw_seed': 2}}):
        other = evaluate_profile(tmp_path, 'other.json', **{'role': 'TEST', 'models': models, 'checkpoints': checkpoints, 'holdout_lock_sha256': lock, **change})
        assert run_evaluate.profile_without_role_sha256(other) != run_evaluate.profile_without_role_sha256(dev), change
    with pytest.raises(run_io.IORefused, match='evaluate'):
        run_evaluate.profile_without_role_sha256(run_profile.load(_train_profile(tmp_path)))


def _train_profile(tmp_path):
    path = tmp_path / 'train.json'
    path.write_bytes(b'{"kind": "train"}\n')
    return path


# ---- the access ledger (D4, D20) ---------------------------------------------------------------------------------------------------------------
def test_an_access_ledger_is_created_once_and_starts_with_a_created_entry_and_no_opened_test(tmp_path):
    clock = FakeClock()
    ledger = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    entries = ledger.entries()
    assert [e['event'] for e in entries] == ['created'] and entries[0]['seq'] == 0 and entries[0]['previous_sha256'] is None
    assert entries[0]['when'] == 1001.0 and ledger.test_opened() is False
    document = run_io.read_json(tmp_path / 'access' / run_evaluate.ACCESS_FILE)
    assert document['schema'] == run_evaluate.ACCESS_SCHEMA and document['entries'] == entries
    with pytest.raises(run_evaluate.AccessRefused, match='already'):
        run_evaluate.AccessLedger.create(tmp_path / 'access')                                       # never overwritten, not even by an equal one


def test_a_dev_access_appends_one_entry_that_names_the_previous_one_by_hash(tmp_path):
    clock = FakeClock()
    ledger = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    first = ledger.append('dev_access', {'run': 'a'}, clock=clock)
    second = ledger.append('dev_access', {'run': 'b'}, clock=clock)
    entries = ledger.entries()
    assert [e['event'] for e in entries] == ['created', 'dev_access', 'dev_access'] and [e['seq'] for e in entries] == [0, 1, 2]
    assert entries[1] == first and entries[2] == second and first['record'] == {'run': 'a'} and second['when'] == first['when'] + 1.0
    assert entries[1]['previous_sha256'] == run_io.canonical_sha256(entries[0])
    assert entries[2]['previous_sha256'] == run_io.canonical_sha256(entries[1])
    assert ledger.test_opened() is False


@pytest.mark.parametrize('mutate, word', [
    (lambda doc: doc['entries'].pop(0), 'created'),                                                 # the first entry is gone
    (lambda doc: doc['entries'][1].update(record={'run': 'forged'}), 'chain'),                     # an entry was edited: the next one names the old hash
    (lambda doc: doc['entries'].insert(1, dict(doc['entries'][1])), 'seq'),                        # an entry was repeated
    (lambda doc: doc['entries'][1].update(event='other'), 'event'),
    (lambda doc: doc.update(schema='other'), 'schema'),
    (lambda doc: doc.update(extra=1), 'fields'),
    (lambda doc: doc['entries'][1].update(previous_sha256=None), 'chain')])
def test_a_tampered_access_ledger_is_refused_when_it_is_read(tmp_path, mutate, word):
    clock = FakeClock()
    ledger = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    ledger.append('dev_access', {'run': 'a'}, clock=clock)
    ledger.append('dev_access', {'run': 'b'}, clock=clock)
    path = tmp_path / 'access' / run_evaluate.ACCESS_FILE
    document = run_io.read_json(path)
    mutate(document)
    run_io.write_json_atomic(path, document)
    with pytest.raises(run_evaluate.AccessRefused, match=word):
        ledger.entries()
    with pytest.raises(run_evaluate.AccessRefused):
        ledger.append('dev_access', {'run': 'c'})                                                   # nothing is appended to a ledger that does not verify


def test_a_missing_or_foreign_access_ledger_is_refused_not_created_on_the_fly(tmp_path):
    ledger = run_evaluate.AccessLedger(tmp_path / 'absent')
    with pytest.raises(run_evaluate.AccessRefused, match='no access ledger'):
        ledger.entries()
    with pytest.raises(run_evaluate.AccessRefused, match='no access ledger'):
        ledger.append('dev_access', {})
    assert not (tmp_path / 'absent').exists()
    clock = FakeClock()
    made = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    with pytest.raises(run_evaluate.AccessRefused, match='event'):
        made.append('created', {})                                                                  # only dev_access and test_opened are appended
    with pytest.raises(run_evaluate.AccessRefused, match='event'):
        made.append('anything', {})


def test_a_busy_access_ledger_is_refused_not_overwritten(tmp_path):
    clock = FakeClock()
    ledger = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    (tmp_path / 'access' / run_evaluate.ACCESS_LOCK).write_text('pid=1\n', encoding='utf-8')       # another writer, or a crashed one
    with pytest.raises(run_evaluate.AccessRefused, match='lock'):
        ledger.append('dev_access', {'run': 'x'})
    assert [e['event'] for e in ledger.entries()] == ['created']


def test_a_test_opening_is_written_once_and_a_second_one_is_refused_also_if_the_ledger_file_was_cut_back(tmp_path):
    clock = FakeClock()
    ledger = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    opened = ledger.append('test_opened', {'run': 'first'}, clock=clock)
    assert ledger.test_opened() is True and opened['event'] == 'test_opened'
    with pytest.raises(run_evaluate.AccessRefused, match='opened'):
        ledger.append('test_opened', {'run': 'second'})
    assert [e['event'] for e in ledger.entries()] == ['created', 'test_opened']
    # someone cuts the entry out of the ledger file: the marker that was created exclusively still says that TEST was opened
    path = tmp_path / 'access' / run_evaluate.ACCESS_FILE
    document = run_io.read_json(path)
    document['entries'].pop()
    run_io.write_json_atomic(path, document)
    assert [e['event'] for e in ledger.entries()] == ['created'] and ledger.test_opened() is True
    with pytest.raises(run_evaluate.AccessRefused, match='opened'):
        ledger.append('test_opened', {'run': 'third'})


# ---- the TEST gate (D20, Review Focus 5) --------------------------------------------------------------------------------------------------------
MODELS = ['seed1', 'seed2', 'seed3']
SOURCES = {'loader': 'c_ept.run.evaluate:load_eval_inputs', 'backend': 'c_ept.run.evaluate:Backend', 'manifest_sha256': 'a' * 64, 'allow_custom': False, 'body': FAKE_BODY}
IDENTITY_A = Identity(commit='ab' * 20, files={'c_ept/run/evaluate.py': 'cd' * 32}, protected=())
IDENTITY_B = Identity(commit='ab' * 20, files={'c_ept/run/evaluate.py': 'ef' * 32}, protected=())           # the same commit, an imported file that differs


def lock_document(access, **override) -> dict:
    """The S5 lock document of a ledger as the test writes it (independent of the code under test): the schema, the folder and the hash of the ``created`` entry. ``override``
    replaces a field, ``None`` removes it."""
    document = {'schema': 'ept-run-holdout-lock-v1', 'access_dir': Path(access.folder).resolve().as_posix(), 'created_entry_sha256': run_io.canonical_sha256(access.entries()[0])}
    document.update(override)
    return {key: val for key, val in document.items() if val is not None}


def write_lock(path, access, **override) -> str:
    """Write the lock file of ``access`` (strict JSON, LF) and return its sha256: the value of ``holdout_lock_sha256`` of the profiles."""
    path = Path(path)
    path.write_bytes((json.dumps(lock_document(access, **override), indent=2, sort_keys=True) + '\n').encode('utf-8'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GateWorld:
    """What the TEST gate looks at: a lock file with its hash, a finished DEV rehearsal run (a real ledger with the record and the report of the evaluate stage), the
    access ledger in which the rehearsal was logged, and the TEST profile that differs from the DEV profile in ``role`` only."""

    def __init__(self, tmp_path, lock_override=None, **profile_fields):
        """``lock_override``: fields of the S5 lock document that are replaced (``None`` removes one); the profiles carry the hash of that lock, so it is the lock that was fixed."""
        from c_ept.tests.test_run_ledger import make_header
        self.tmp, self.make_header = tmp_path, make_header
        self.clock = FakeClock()
        self.access = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=self.clock)
        self.lock = tmp_path / 'S5_LOCK.json'
        self.lock_sha256 = write_lock(self.lock, self.access, **(lock_override or {}))
        self.fields = {'models': MODELS, 'checkpoints': {name: str(tmp_path / name) for name in MODELS}, 'holdout_lock_sha256': self.lock_sha256, **profile_fields}
        self.dev = evaluate_profile(tmp_path, 'dev.json', role='DEV', **self.fields)
        self.test = evaluate_profile(tmp_path, 'test.json', role='TEST', **self.fields)
        self.run = tmp_path / 'rehearsal'
        self.sources, self.models = dict(SOURCES), {name: sha_text(name) for name in MODELS}

    def rehearse(self, *, identity=IDENTITY_A, profile=None, finish=True, role='DEV', log=True, name=None, record_profile=None, sources=None, models=None, access_override=None,
                 access=None):
        """``record_profile``: the profile whose hash the record of the rehearsal names, when it is not the profile that the log names (a record that lies). ``sources`` and
        ``models`` are what the record of the rehearsal says about the loader, the backend and the inputs (K6b I2) and about the training records of its checkpoints. ``access``:
        the ledger in which the rehearsal is logged, when it is not this world's."""
        ledger_of_the_log = access or self.access
        profile = profile or self.dev
        recorded = record_profile or profile
        sources = dict(self.sources if sources is None else sources)
        models = dict(self.models if models is None else models)
        run = self.tmp / name if name else self.run
        header = self.make_header(identity=identity, profile_sha256=profile.sha256,
                                  runner={'evaluation': {'role': role, 'profile_without_role_sha256': run_evaluate.profile_without_role_sha256(profile)}})
        ledger = run_ledger.Ledger.create(run, header)
        access = None
        if log:
            entry = ledger_of_the_log.append('dev_access', {'role': 'DEV', 'profile_without_role_sha256': run_evaluate.profile_without_role_sha256(profile),
                                                      'identity_sha256': run_io.canonical_sha256(identity.to_record()), 'sources': sources}, clock=self.clock)
            access = {'event': entry['event'], 'seq': entry['seq'], 'entry_sha256': run_io.canonical_sha256(entry)}
        if access_override is not None:
            access = access_override                                                                  # a record that names an entry that is not in the ledger
        record = {'schema': run_evaluate.RECORD_SCHEMA, 'role': role, 'profile_sha256': recorded.sha256,
                  'profile_without_role_sha256': run_evaluate.profile_without_role_sha256(recorded), 'sources': sources,
                  'models': {model: {'train_record_sha256': digest} for model, digest in models.items()}, 'access': access}
        run_io.write_json_atomic(run / run_evaluate.RECORD_FILE, record)
        run_io.write_json_atomic(run / run_evaluate.REPORT_FILE, {'schema': run_evaluate.REPORT_SCHEMA, 'role': role})
        ledger.begin(run_evaluate.STAGE, inputs=[])
        if finish:
            ledger.finish(run_evaluate.STAGE, [run_evaluate.RECORD_FILE, run_evaluate.REPORT_FILE])
        return run

    def check(self, profile=None, gate=None, **over):
        """``check_evidence`` of the gate for the TEST profile with the sources and the training records that a TEST run of this world would have."""
        return (gate or self.gate()).check_evidence(profile or self.test, sources=over.get('sources', self.sources), models=over.get('models', self.models))

    def open(self, identity=IDENTITY_A, profile=None, gate=None, **over):
        return (gate or self.gate()).open(profile or self.test, identity, sources=over.get('sources', self.sources), models=over.get('models', self.models), clock=self.clock)

    def gate(self, **over):
        values = dict(lock_path=self.lock, rehearsal_run=self.run, access=self.access)
        values.update(over)
        return run_evaluate.HoldoutGate(**values)


@pytest.fixture
def gate_world(tmp_path):
    world = GateWorld(tmp_path)
    world.rehearse()
    return world


def test_test_role_requires_lock_and_rehearsal(gate_world):
    """The three things D20 asks for, one by one: the lock file with the hash that was fixed, a finished DEV rehearsal of the same profile (without its role) and the
    same code, and a ledger in which TEST was never opened. Each missing one refuses, nothing is opened by a refusal."""
    world = gate_world
    evidence = world.check()
    assert evidence['lock_sha256'] == world.lock_sha256 and evidence['never_opened'] is True
    assert evidence['rehearsal']['profile_without_role_sha256'] == run_evaluate.profile_without_role_sha256(world.test)
    assert evidence['rehearsal']['identity_sha256'] == run_io.canonical_sha256(IDENTITY_A.to_record())
    world.gate().check_identity(evidence, IDENTITY_A)                                                # the same code as the rehearsal's
    assert world.access.test_opened() is False


def test_the_lock_file_must_be_there_and_must_have_the_hash_that_the_profile_fixes(gate_world):
    """Decision D28: the hash of the lock file is a field of the TEST profile (``holdout_lock_sha256``), not an argument of the caller."""
    world = gate_world
    other = evaluate_profile(world.tmp, 'other_lock.json', role='TEST', **{**world.fields, 'holdout_lock_sha256': '0' * 64})          # the profile fixes another file
    with pytest.raises(run_evaluate.AccessRefused, match='lock'):
        world.check(other)
    with pytest.raises(run_evaluate.AccessRefused, match='lock'):
        world.check(gate=world.gate(lock_path=world.tmp / 'absent.json'))
    world.lock.write_bytes(world.lock.read_bytes() + b' ')                                           # the lock was edited after its hash was taken
    with pytest.raises(run_evaluate.AccessRefused, match='lock'):
        world.check()
    assert world.access.test_opened() is False


def test_a_test_profile_without_the_hash_of_the_lock_cannot_exist(tmp_path):
    fields = {'models': MODELS, 'checkpoints': {name: str(tmp_path / name) for name in MODELS}}
    with pytest.raises(run_profile.ProfileRefused, match='holdout_lock_sha256'):
        evaluate_profile(tmp_path, 'nolock.json', role='TEST', **fields)
    with pytest.raises(run_profile.ProfileRefused, match='holdout_lock_sha256'):
        evaluate_profile(tmp_path, 'nulllock.json', role='TEST', holdout_lock_sha256=None, **fields)
    dev = evaluate_profile(tmp_path, 'dev.json', role='DEV', **fields)
    test = evaluate_profile(tmp_path, 'test.json', role='TEST', holdout_lock_sha256='ab' * 32, **fields)
    assert dev['holdout_lock_sha256'] is None and run_evaluate.profile_without_role_sha256(dev) != run_evaluate.profile_without_role_sha256(test)    # the lock is part of the profile


@pytest.mark.parametrize('override, word', [({'schema': 'other'}, 'lock'), ({'access_dir': None}, 'lock'), ({'created_entry_sha256': None}, 'lock'), ({'extra': 1}, 'lock'),
                                            ({'created_entry_sha256': 'nothex'}, 'lock'), ({'access_dir': 7}, 'lock'),
                                            ({'access_dir': 'C:/somewhere/else/access'}, 'folder'), ({'created_entry_sha256': '0' * 64}, 'created entry')])
def test_the_lock_must_name_this_access_ledger_by_its_folder_and_its_created_entry(tmp_path, override, word):
    world = GateWorld(tmp_path, lock_override=override)                                               # a lock that the profile fixes, with a wrong or malformed content
    world.rehearse()
    with pytest.raises(run_evaluate.AccessRefused, match=word):
        world.check()
    with pytest.raises(run_evaluate.AccessRefused, match=word):
        world.open()
    assert world.access.test_opened() is False and not world.access.marker.exists()


def test_the_lock_document_of_a_ledger_is_what_the_gate_reads(gate_world):
    world = gate_world
    assert run_evaluate.holdout_lock_document(world.access) == lock_document(world.access)
    assert run_io.read_json(world.lock) == lock_document(world.access) and set(lock_document(world.access)) == {'schema', 'access_dir', 'created_entry_sha256'}


# ---- the four ways round the one opening that D28 closes (review C1) ------------------------------------------------------------------------------
def test_a_new_ledger_with_a_new_rehearsal_does_not_reopen_test(gate_world):
    world = gate_world
    world.open()                                                                                      # TEST was opened, as it can be once
    fresh = run_evaluate.AccessLedger.create(world.tmp / 'access_new', clock=FakeClock(3000.0))        # a ledger in which TEST was never opened
    world.rehearse(name='rehearsal_new', access=fresh)                                                # and a DEV rehearsal that is logged in it, under the same profile and code
    assert fresh.test_opened() is False
    with pytest.raises(run_evaluate.AccessRefused, match='folder'):
        world.check(gate=world.gate(access=fresh, rehearsal_run=world.tmp / 'rehearsal_new'))
    with pytest.raises(run_evaluate.AccessRefused, match='folder'):
        world.open(gate=world.gate(access=fresh, rehearsal_run=world.tmp / 'rehearsal_new'))
    assert fresh.test_opened() is False and [e['event'] for e in world.access.entries()].count('test_opened') == 1


def test_a_copy_of_the_ledger_folder_taken_before_the_opening_does_not_reopen_test(gate_world):
    import shutil
    world = gate_world
    shutil.copytree(world.access.folder, world.tmp / 'access_copy')                                   # the copy holds the logged rehearsal and no opening
    world.open()
    copy = run_evaluate.AccessLedger(world.tmp / 'access_copy')
    assert copy.test_opened() is False and any(e['event'] == 'dev_access' for e in copy.entries())
    with pytest.raises(run_evaluate.AccessRefused, match='folder'):
        world.check(gate=world.gate(access=copy))
    assert copy.test_opened() is False


def test_a_ledger_that_was_deleted_with_its_marker_and_made_again_does_not_reopen_test(gate_world):
    world = gate_world
    world.open()
    world.access.path.unlink()
    world.access.marker.unlink()                                                                      # nothing is left of the opening ...
    again = run_evaluate.AccessLedger.create(world.access.folder, clock=FakeClock(9000.0))           # ... and the ledger is made again in the same folder
    world.rehearse(name='rehearsal_again', access=again)
    assert again.test_opened() is False
    with pytest.raises(run_evaluate.AccessRefused, match='created entry'):                           # the lock names the created entry of the first ledger
        world.check(gate=world.gate(access=again, rehearsal_run=world.tmp / 'rehearsal_again'))
    assert again.test_opened() is False


def test_a_rehearsal_made_under_another_lock_is_refused(gate_world):
    world = gate_world
    lock2 = world.tmp / 'S5_LOCK_2.json'
    sha2 = write_lock(lock2, world.access)                                                           # the same content ...
    lock2.write_bytes(lock2.read_bytes().replace(b'\n', b'\n\n'))                                  # ... in other bytes, hence another lock
    sha2 = hashlib.sha256(lock2.read_bytes()).hexdigest()
    other_dev = evaluate_profile(world.tmp, 'dev2.json', role='DEV', **{**world.fields, 'holdout_lock_sha256': sha2})
    world.rehearse(profile=other_dev, name='rehearsal_lock2')
    with pytest.raises(run_evaluate.AccessRefused, match='another profile'):
        world.check(gate=world.gate(rehearsal_run=world.tmp / 'rehearsal_lock2'))


def test_the_happy_path_opens_once_and_the_opening_names_the_lock_and_the_ledger(gate_world):
    world = gate_world
    entry = world.open()
    record = entry['record']
    assert record['lock_sha256'] == world.lock_sha256 == world.test['holdout_lock_sha256'] and record['created_entry_sha256'] == lock_document(world.access)['created_entry_sha256']
    assert record['access_dir'] == lock_document(world.access)['access_dir']
    with pytest.raises(run_evaluate.AccessRefused, match='opened'):
        world.open()


def test_the_gate_is_for_the_test_role_only(gate_world):
    world = gate_world
    for profile in (world.dev, evaluate_profile(world.tmp, 'stop.json', role='STOP', **world.fields)):
        with pytest.raises(run_evaluate.AccessRefused, match='TEST'):
            world.check(profile)


def test_a_rehearsal_that_is_missing_unfinished_not_dev_or_of_another_profile_is_refused(tmp_path):
    cases = [('no run folder', lambda w: w.gate(rehearsal_run=w.tmp / 'nowhere'), 'rehearsal'),
             ('unfinished', lambda w: (w.rehearse(finish=False, name='unfinished', log=False), w.gate(rehearsal_run=w.tmp / 'unfinished'))[1], 'finished'),
             ('a STOP run', lambda w: (w.rehearse(role='STOP', name='stoprun'), w.gate(rehearsal_run=w.tmp / 'stoprun'))[1], 'DEV'),
             ('another profile', lambda w: (w.rehearse(profile=evaluate_profile(w.tmp, 'other.json', role='DEV', **{**w.fields, 'bootstrap_B': 1000}), name='other'),
                                            w.gate(rehearsal_run=w.tmp / 'other'))[1], 'profile')]
    for number, (label, make, word) in enumerate(cases):
        base = tmp_path / f'case{number}'
        base.mkdir()
        world = GateWorld(base)
        gate = make(world)
        with pytest.raises(run_evaluate.AccessRefused, match=word):
            world.check(gate=gate)
        assert world.access.test_opened() is False, label


def test_a_rehearsal_whose_files_changed_after_the_ledger_hashed_them_is_refused(gate_world):
    world = gate_world
    path = world.run / run_evaluate.REPORT_FILE
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(run_evaluate.AccessRefused, match='changed'):
        world.check()


def test_a_rehearsal_record_that_names_another_profile_than_the_log_does_is_refused(tmp_path):
    world = GateWorld(tmp_path)
    other = evaluate_profile(tmp_path, 'other.json', role='DEV', **{**world.fields, 'bootstrap_B': 1000})
    world.rehearse(record_profile=other)                                                              # logged for this profile, but the record of the run says another one
    with pytest.raises(run_evaluate.AccessRefused, match='another profile'):
        world.check()


def test_a_rehearsal_that_was_not_logged_in_the_access_ledger_is_refused(tmp_path):
    world = GateWorld(tmp_path)
    world.rehearse(log=False)
    with pytest.raises(run_evaluate.AccessRefused, match='logged'):
        world.check()


def test_a_rehearsal_of_other_code_is_refused_by_the_identity_check(tmp_path):
    world = GateWorld(tmp_path)
    world.rehearse(identity=IDENTITY_A)
    evidence = world.check()
    with pytest.raises(run_evaluate.AccessRefused, match='identity'):
        world.gate().check_identity(evidence, IDENTITY_B)                                            # the same commit but a file of the import list differs
    with pytest.raises(run_evaluate.AccessRefused, match='identity'):
        world.open(IDENTITY_B)
    assert world.access.test_opened() is False


def test_a_missing_or_opened_access_ledger_refuses_the_gate(gate_world, tmp_path):
    world = gate_world
    with pytest.raises(run_evaluate.AccessRefused, match='access ledger'):
        world.check(gate=world.gate(access=run_evaluate.AccessLedger(tmp_path / 'nowhere')))
    world.access.append('test_opened', {'run': 'someone else'}, clock=world.clock)
    with pytest.raises(run_evaluate.AccessRefused, match='opened'):
        world.check()


def test_test_role_opens_once(gate_world):
    """Review Focus 5: TEST is never opened twice and never before the lock and the rehearsal. The opening is one entry with the evidence; the same gate refuses
    a second opening, also when the profile and the code are the same."""
    world = gate_world
    entry = world.open()
    assert entry['event'] == 'test_opened' and world.access.test_opened() is True
    record = entry['record']
    assert record['lock_sha256'] == world.lock_sha256 and record['profile_sha256'] == world.test.sha256
    assert record['profile_without_role_sha256'] == run_evaluate.profile_without_role_sha256(world.test)
    assert record['identity_sha256'] == run_io.canonical_sha256(IDENTITY_A.to_record()) and record['identity_commit'] == IDENTITY_A.commit
    assert record['rehearsal_record_sha256'] == run_io.sha_file(world.run / run_evaluate.RECORD_FILE)
    with pytest.raises(run_evaluate.AccessRefused, match='opened'):
        world.open()
    assert [e['event'] for e in world.access.entries()].count('test_opened') == 1
    other = evaluate_profile(world.tmp, 'other.json', role='TEST', **{**world.fields, 'exposure': 'nominal'})
    with pytest.raises(run_evaluate.AccessRefused):                                                  # nor can another TEST profile open it again
        world.open(profile=other)


# ---- the toy world, the stores and real tiny trainings of K6a ----------------------------------------------------------------------------------
def scales_document(**changes) -> dict:
    """A complete, valid ``B0_SCALES.json`` record written by hand (see ``test_run_runners``); ``changes`` replace top-level fields."""
    from c_ept.ept import scales
    keys = list(loss.BLOCKS)
    document = {'schema': 'b0_scales_v1', 's3_run': '20261022_c_ept_s3_3', 'role': 'FIT', 'rule': scales.RULE,
                's1': {'run': '20261020_c_ept_s1_1', 'files_sha256': '1' * 64}, 's2': {'run': '20261021_c_ept_s2_3', 'files_sha256': '2' * 64},
                'labels_store_sha256': '3' * 64, 'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64, 'fit/b0/b0_heads.npz': '5' * 64},
                'scales': dict(SCALES), 'n_rows': {k: 10 for k in keys}, 'n_unscored': {k: 0 for k in keys}, 'by_size': {},
                'n_rows_unit': {k: {'M2Rate': 'frame_row', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
                'n_unscored_unit': {k: {'M2Rate': 'time_bin_cell', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
                'size_normalised': sorted(f'{h}/{k}' for h in loss.SIZE_NORMALISED for k in ('masked', 'open'))}
    document.update(changes)
    document.setdefault('scales_sha256', hashlib.sha256(json.dumps(document['scales'], sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest())
    return document


def write_scales(folder, name, **changes):
    path = Path(folder) / name
    path.write_bytes(json.dumps(scales_document(**changes), indent=1).encode('utf-8'))
    return path


@pytest.fixture(autouse=True)
def restore_torch_state(monkeypatch):
    """``train`` and ``evaluate`` change process-wide settings (threads, deterministic algorithms, the global random streams, and with K6c the numeric policy: matmul precision, TF32,
    the cuDNN flags, ``fill_uninitialized_memory`` and ``CUBLAS_WORKSPACE_CONFIG``); give them back after each test."""
    from c_ept.tests.test_run_numeric import read_flags, write_flags
    threads, flags = torch.get_num_threads(), read_flags()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))                # monkeypatch puts back what was there, or nothing
    yield
    torch.set_num_threads(threads)
    write_flags(flags)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """The EPT toy world whose units all have a kill by their first-kill team (the rows of the shifted kill law), 7 events and 5 edges per view."""
    built = build_ept_world(tmp_path_factory.mktemp('evaluate_world'), n_events=7, n_edges=5, consistent_first_kill=True)
    yield built
    del built
    gc.collect()                                                                                      # Windows keeps a folder with open memory maps


@pytest.fixture(scope='module')
def reps_sha(world):
    return value.gold_reps_sha256(value.fit_gold_bin_means(world.s1, world.labels, population='in_scope').means)


def quiet_probe():
    return dict(E6_RECORD)


class Factory:
    """Real tiny trainings of K6a on the toy world, each with the ledger of a finished run (the stages ``stats`` and ``train`` with their outputs), cached by name."""

    def __init__(self, root, world, reps_sha):
        self.root, self.world, self.reps_sha, self.runs = Path(root), world, reps_sha, {}

    def scales_for(self, fields):
        kill_law, k_condition = fields.get('kill_law', 'poisson'), fields.get('k_condition', 'raw')
        extra = {'k_reps_sha256': self.reps_sha} if k_condition == 'binned' else {}
        return write_scales(self.root, f'scales_{kill_law}_{k_condition}.json', kill_law=kill_law, k_condition=k_condition, **extra)

    def inputs_for(self, fields):
        return runners.Inputs(s1=self.world.s1, s2=self.world.s2, labels=self.world.labels, static=toy_static(), static_source=None, scales=dict(SCALES),
                              scales_path=self.scales_for(fields))

    def make(self, name, finish_train=True, **fields) -> Path:
        from c_ept.tests.test_run_ledger import make_header
        if name in self.runs:
            return self.runs[name]
        fields = {'epochs': 1, **fields}
        profile_path = self.root / f'{name}.json'
        profile_path.write_bytes((json.dumps({'kind': 'train', **fields}, indent=2) + '\n').encode('utf-8'))
        profile = run_profile.load(profile_path)
        run = self.root / f'run_{name}'
        inputs = self.inputs_for(fields)
        runners.write_stats(profile, run, inputs=inputs, constants=CPU)
        runners.train(profile, run, inputs=inputs, constants=CPU, probe=quiet_probe)
        ledger = run_ledger.Ledger.create(run, make_header(identity=IDENTITY_A, profile_sha256=profile.sha256, runner={}))
        ledger.begin('stats', inputs=[profile_path.as_posix()])
        ledger.finish('stats', [runners.STATS_FILE, runners.STATS_RECORD_FILE])
        ledger.begin('train', inputs=[profile_path.as_posix(), run / runners.STATS_FILE, run / runners.STATS_RECORD_FILE])
        if finish_train:
            ledger.finish('train', [runners.TRAIN_RECORD_FILE, runners.E6_FILE])
        self.runs[name] = run
        return run


@pytest.fixture(scope='module')
def factory(tmp_path_factory, world, reps_sha):
    return Factory(tmp_path_factory.mktemp('evaluate_runs'), world, reps_sha)


S4B_FIELDS = {key: value_ for key, value_ in S4B.items() if key != 'kind'}


def seed_run(factory, number, **over):
    """The cached run of seed ``number`` of the S4b-like configuration; ``over`` (settings that differ) makes another cached run, named after them."""
    name = f'seed{number}' + ''.join(f'_{key}_{val}' for key, val in sorted(over.items()))
    return factory.make(name, **{**S4B_FIELDS, 'seed': SEEDS[number - 1], **over})


# ---- one trained run, read through its ledger and its record -------------------------------------------------------------------------------------
def test_a_training_run_is_read_through_its_ledger_and_its_record(factory, reps_sha):
    folder = seed_run(factory, 1)
    trained = run_evaluate.load_trained('seed1', folder)
    assert (trained.name, trained.seed, trained.folder) == ('seed1', SEEDS[0], folder)
    assert dict(trained.settings) == {'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'terrain_smooth_weight': 0.0, 'swap_mode': 'random_view',
                                      'eval_average': True, 'on_wall_cap': 'best_stop', 'zone_mark': True, 'column_drop': 'store_and_subset',
                                      'stats_population': 'selected_fit', 'k_reps_sha256': reps_sha}
    assert trained.record_sha256 == run_io.sha_file(folder / runners.TRAIN_RECORD_FILE)
    assert trained.record['schema'] == runners.TRAIN_RECORD_SCHEMA
    best = trained.head['best']
    assert trained.best_path == folder / 'train' / 'ckpt' / best['folder'] / 'best.pt' and run_io.sha_file(trained.best_path) == best['sha256']
    assert trained.stats.sha256 == trained.record['stats']['sha256'] and trained.encoder == trained.head['identity']['encoder']
    bound = trained.bound_files()
    assert bound == sorted(bound) and all(Path(p).is_absolute() and '\\' not in p for p in bound)
    assert {Path(p).name for p in bound} == {'RECORD.json', 'STATS.json', 'head.json', 'best.pt'}


def test_the_settings_that_the_identity_leaves_out_are_the_old_defaults_and_a_raw_run_has_no_representatives(factory):
    trained = run_evaluate.load_trained('plain', factory.make('plain'))
    assert dict(trained.settings) == {'kill_law': 'poisson', 'k_condition': 'raw', 'decay_groups': 'all', 'terrain_smooth_weight': 0.0, 'swap_mode': 'average',
                                      'eval_average': True, 'on_wall_cap': 'fail', 'zone_mark': True, 'column_drop': 'none', 'stats_population': 'whole_fit',
                                      'k_reps_sha256': None}
    assert run_evaluate.settings_of({}) == dict(trained.settings) and 'kill_law' not in trained.identity                       # absent in the identity means the default


def copy_run(factory, tmp_path, name='seed1', **fields):
    import shutil
    target = tmp_path / 'copy'
    shutil.copytree(seed_run(factory, 1) if not fields else factory.make(name, **fields), target)
    return target


@pytest.mark.parametrize('relative, word', [(runners.TRAIN_RECORD_FILE, 'changed'), (runners.STATS_FILE, 'changed'), (runners.STATS_RECORD_FILE, 'changed'),
                                            (runners.E6_FILE, 'changed')])
def test_a_file_of_the_training_run_that_changed_after_the_ledger_hashed_it_is_refused(factory, tmp_path, relative, word):
    copy = copy_run(factory, tmp_path)
    path = copy / relative
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(run_evaluate.CheckpointRefused, match=word):
        run_evaluate.load_trained('seed1', copy)


def test_a_best_checkpoint_or_a_head_that_differs_from_the_record_is_refused(factory, tmp_path):
    copy = copy_run(factory, tmp_path)
    trained = run_evaluate.load_trained('seed1', copy)
    head_file = trained.best_path.parents[0] / 'head.json'
    heads = sorted((copy / 'train' / 'ckpt').glob('epoch_*/head.json'))
    last = heads[-1]
    last.write_bytes(last.read_bytes() + b'\n')                                                      # the head: still valid JSON, but not the bytes of the record
    with pytest.raises(run_evaluate.CheckpointRefused, match='head'):
        run_evaluate.load_trained('seed1', copy)
    last.write_bytes(last.read_bytes()[:-1])                                                         # undone
    trained = run_evaluate.load_trained('seed1', copy)
    trained.best_path.write_bytes(trained.best_path.read_bytes() + b'0')                             # the weights
    with pytest.raises(run_evaluate.CheckpointRefused, match='best'):
        run_evaluate.load_trained('seed1', copy)
    assert head_file.name == 'head.json'


def test_a_training_run_without_a_finished_train_stage_or_without_a_ledger_is_refused(factory, tmp_path):
    with pytest.raises(run_evaluate.CheckpointRefused, match='finished'):
        run_evaluate.load_trained('x', factory.make('unfinished', finish_train=False, **S4B_FIELDS))
    with pytest.raises(run_evaluate.CheckpointRefused, match='ledger'):
        run_evaluate.load_trained('x', tmp_path / 'nowhere')
    copy = copy_run(factory, tmp_path)
    (copy / 'stages' / 'train.json').unlink()                                                        # a finished stage whose stage file is gone: the ledger is tampered with
    with pytest.raises(run_evaluate.CheckpointRefused, match='ledger'):
        run_evaluate.load_trained('x', copy)


def test_an_identity_key_that_nobody_classified_is_refused(factory, monkeypatch):
    monkeypatch.setattr(eval_guard, 'CLASSIFIED_IDENTITY_KEYS', frozenset(eval_guard.CLASSIFIED_IDENTITY_KEYS - {'swap_mode'}))
    with pytest.raises(run_evaluate.CheckpointRefused, match='not classified'):
        run_evaluate.load_trained('seed1', seed_run(factory, 1))


# ---- models and checkpoints, in both directions ----------------------------------------------------------------------------------------------------
def test_checkpoints_and_models_must_match_in_both_directions(factory, tmp_path):
    folder = str(seed_run(factory, 1))
    ok = evaluate_profile(tmp_path, 'ok.json', role='STOP', models=['seed1'], checkpoints={'seed1': folder})
    assert [t.name for t in run_evaluate.load_models(ok)] == ['seed1']
    missing = evaluate_profile(tmp_path, 'missing.json', role='STOP', models=['seed1', 'seed2'], checkpoints={'seed1': folder})
    with pytest.raises(run_evaluate.CheckpointRefused, match="model 'seed2' has no checkpoint"):
        run_evaluate.load_models(missing)
    extra = evaluate_profile(tmp_path, 'extra.json', role='STOP', models=['seed1'], checkpoints={'seed1': folder, 'other': folder})
    with pytest.raises(run_evaluate.CheckpointRefused, match="checkpoint 'other' belongs to no model"):
        run_evaluate.load_models(extra)
    none = evaluate_profile(tmp_path, 'none.json', role='STOP', models=['seed1'])
    with pytest.raises(run_evaluate.CheckpointRefused, match='no checkpoint'):
        run_evaluate.load_models(none)


def test_the_models_are_ordered_by_training_seed_and_two_models_of_one_seed_are_refused(factory, tmp_path):
    folders = {f'seed{n}': str(seed_run(factory, n)) for n in (3, 1, 2)}
    profile = evaluate_profile(tmp_path, 'all.json', role='STOP', models=['seed3', 'seed1', 'seed2'], checkpoints=folders)
    loaded = run_evaluate.load_models(profile)
    assert [t.seed for t in loaded] == list(SEEDS) and [t.name for t in loaded] == ['seed1', 'seed2', 'seed3']
    twice = evaluate_profile(tmp_path, 'twice.json', role='STOP', models=['a', 'b'], checkpoints={'a': folders['seed1'], 'b': folders['seed1']})
    with pytest.raises(run_evaluate.CheckpointRefused, match='same training seed'):
        run_evaluate.load_models(twice)

# ---- the checks of a set of models that are evaluated together ----------------------------------------------------------------------------------
def eval_inputs(factory, fields=None, **extra):
    base = factory.inputs_for(S4B_FIELDS if fields is None else fields)
    return run_evaluate.EvalInputs(base=base, **extra)


def trained_set(factory, numbers=(1, 2, 3)):
    return [run_evaluate.load_trained(f'seed{n}', seed_run(factory, n)) for n in numbers]


def test_three_seeds_of_one_configuration_agree_and_the_agreed_conditions_are_returned(factory, reps_sha):
    trained = trained_set(factory)
    inputs = eval_inputs(factory)
    checked = run_evaluate.check_models(trained, inputs)
    assert checked['settings'] == dict(trained[0].settings) and checked['settings']['k_reps_sha256'] == reps_sha
    reps = value.fit_gold_bin_means(factory.world.s1, factory.world.labels, population='in_scope')
    assert checked['reps']['sha256'] == reps_sha and checked['reps']['fallback'] == [int(b) for b in reps.fallback] and np.array_equal(checked['reps']['array'], reps.means)
    assert checked['scales'] == {'file_sha256': run_io.sha_file(inputs.base.scales_path),
                                 'condition': {'kill_law': 'shifted', 'k_condition': 'binned', 'k_reps_sha256': reps_sha}}
    dropped = list(trained[0].stats.dropped)
    assert checked['dropped'] == {'columns': dropped, 'sha256': run_io.canonical_sha256(dropped)}   # no matched B1 was given: the models' own (equal) lists are recorded (K6b I5)


def test_a_raw_configuration_has_no_representatives_to_check(factory):
    plain = [run_evaluate.load_trained('plain', factory.make('plain'))]
    checked = run_evaluate.check_models(plain, eval_inputs(factory, {}))
    assert checked['reps'] is None and checked['settings']['k_reps_sha256'] is None
    assert checked['scales']['condition'] == {'kill_law': 'poisson', 'k_condition': 'raw', 'k_reps_sha256': None}


def test_identity_mixtures_across_the_models_are_refused_and_name_the_settings_that_differ(factory):
    mixed = [run_evaluate.load_trained('seed1', seed_run(factory, 1)), run_evaluate.load_trained('plain', factory.make('plain'))]
    with pytest.raises(run_evaluate.MixedSettings) as raised:
        run_evaluate.check_models(mixed, eval_inputs(factory))
    text = str(raised.value)
    for name in ('kill_law', 'k_condition', 'k_reps_sha256', 'decay_groups', 'swap_mode', 'column_drop', 'stats_population', 'on_wall_cap'):
        assert name in text, name
    for name in ('zone_mark', 'eval_average', 'terrain_smooth_weight'):
        assert name not in text, name                                                                # only the settings that differ are named
    assert isinstance(raised.value, run_evaluate.CheckpointRefused) and isinstance(raised.value, run_io.IORefused)


@pytest.mark.parametrize('over, name', [({'zone_mark': False}, 'zone_mark'), ({'eval_average': False}, 'eval_average'), ({'terrain_smooth_weight': 0.5}, 'terrain_smooth_weight')])
def test_one_setting_that_differs_between_two_otherwise_equal_models_is_refused_by_name(factory, over, name):
    first = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    other = run_evaluate.load_trained('other', seed_run(factory, 2, **over))
    with pytest.raises(run_evaluate.MixedSettings) as raised:
        run_evaluate.check_models([first, other], eval_inputs(factory))
    assert name in str(raised.value) and 'kill_law' not in str(raised.value)


def test_representatives_of_the_training_that_differ_from_the_evaluation_ones_are_refused(factory, monkeypatch):
    trained = trained_set(factory, (1,))
    real = value.fit_gold_bin_means

    def shifted(*args, **kwargs):
        found = real(*args, **kwargs)
        return found._replace(means=found.means + 1.0)
    monkeypatch.setattr(value, 'fit_gold_bin_means', shifted)
    with pytest.raises(run_evaluate.CheckpointRefused, match='representatives'):
        run_evaluate.check_models(trained, eval_inputs(factory))


def test_loss_scales_of_another_condition_than_the_training_are_refused(factory, reps_sha):
    trained = trained_set(factory, (1,))
    wrong = dataclasses.replace(factory.inputs_for(S4B_FIELDS), scales_path=factory.scales_for({}))        # the scales of the plain law and the raw gold
    with pytest.raises(runners.ScalesConditionRefused, match='kill_law'):
        run_evaluate.check_models(trained, run_evaluate.EvalInputs(base=wrong))
    none = dataclasses.replace(factory.inputs_for(S4B_FIELDS), scales_path=None)
    with pytest.raises(runners.ScalesConditionRefused, match='no scales file'):
        run_evaluate.check_models(trained, run_evaluate.EvalInputs(base=none))


def test_a_static_input_or_loss_scales_that_are_not_the_ones_of_the_training_are_refused(factory):
    trained = trained_set(factory, (1,))
    base = factory.inputs_for(S4B_FIELDS)
    with pytest.raises(run_evaluate.CheckpointRefused, match='static'):
        run_evaluate.check_models(trained, run_evaluate.EvalInputs(base=dataclasses.replace(base, static=toy_static(walls=False))))
    with pytest.raises(run_evaluate.CheckpointRefused, match='scales'):
        run_evaluate.check_models(trained, run_evaluate.EvalInputs(base=dataclasses.replace(base, scales={**SCALES, loss.BLOCKS[0]: 99.0})))


def test_the_dropped_columns_of_the_matched_b1_must_equal_those_of_every_model(factory):
    """M5 hand-off H3: C4 compares an EPT with a B1 that was fitted on the same selected FIT; the run checks it, it does not assume it."""
    from types import SimpleNamespace
    trained = trained_set(factory)
    dropped = list(trained[0].stats.dropped)
    assert dropped, 'the toy world drops columns under store_and_subset'
    checked = run_evaluate.check_models(trained, eval_inputs(factory, b1=SimpleNamespace(dropped=tuple(dropped))))
    assert checked['dropped'] == {'columns': dropped, 'sha256': run_io.canonical_sha256(dropped)}
    for wrong in (tuple(dropped[:-1]), (*dropped, 'extra_column'), tuple(reversed(dropped))):
        with pytest.raises(run_evaluate.CheckpointRefused, match='dropped'):
            run_evaluate.check_models(trained, eval_inputs(factory, b1=SimpleNamespace(dropped=wrong)))


def test_an_empty_set_of_models_is_not_checked_into_a_pass(factory):
    with pytest.raises(run_evaluate.CheckpointRefused, match='no model'):
        run_evaluate.check_models([], eval_inputs(factory))

# ---- the C4 statistic (D13): three seed values, paired, a match bootstrap on the first rows of the common matrix --------------------------------------------
IDS = np.arange(10, 14, dtype=np.int64)                                                             # the sorted match numbers of a made-up judging split


def c4_endpoints(seed_shift=(-0.15, -0.10, -0.05), seeds=SEEDS):
    """The endpoint ``M2/t0/H60/occurrence`` of three EPT seeds and of one B1: 4 matches, 3 rows each, B1 always at 0.5, EPT at 0.5 plus a shift toward the outcome."""
    from c_ept.evaluation import endpoints
    match = np.repeat(IDS, 3)
    row = np.arange(len(match), dtype=np.int64)
    y = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int64)
    p_b1 = np.full(len(match), 0.5)
    out = {}
    for seed, shift in zip(seeds, seed_shift):
        p_ept = np.clip(p_b1 + np.where(y == 1, -shift, shift), 0.0, 1.0)
        out[seed] = endpoints.endpoint('M2/t0/H60/occurrence', 'M2', 't0', 60, 'first_kill_ge1', match, row, y, p_ept, p_b1, counts={})
    return out


def test_c4_statistic_pairs_seed_values():
    from c_ept.evaluation import inference
    found = c4_endpoints()
    matrix = inference.resample_matrix(len(IDS), B=300)
    row = run_evaluate.c4_statistic(found, IDS, matrix)
    assert row['name'] == 'C4' and row['endpoint'] == 'M2/t0/H60/occurrence' and row['metric'] == 'brier' and row['bootstrap_B'] == 300
    match = found[SEEDS[0]]['match']
    y, b1 = found[SEEDS[0]]['y'], found[SEEDS[0]]['p_b1']
    weights = 1.0 / np.array([np.count_nonzero(match == m) for m in match])
    brier = lambda p: float(np.sum(weights * (p - y) ** 2) / np.sum(weights))                           # noqa: E731 - 1/n_m weights, the D15 aggregation
    expected = {seed: brier(found[seed]['p_ept']) - brier(b1) for seed in SEEDS}
    assert {int(k): v['diff'] for k, v in row['per_seed'].items()} == pytest.approx(expected, rel=1e-12)
    assert row['delta'] == pytest.approx(sum(expected.values()) / 3, rel=1e-12) and row['delta'] < 0
    for seed in SEEDS:
        assert row['per_seed'][str(seed)]['ept'] == pytest.approx(brier(found[seed]['p_ept'])) and row['per_seed'][str(seed)]['b1'] == pytest.approx(brier(b1))
    # the interval is the percentile interval of the mean of the three paired draws on the one shared matrix, the p the two-sided p of the same draws
    ones = np.ones(len(IDS))
    draws = []
    for seed in SEEDS:
        diff = (found[seed]['p_ept'] - y) ** 2 - (b1 - y) ** 2
        num = inference.match_aggregate(match, diff, IDS)[1]
        draws.append(inference.bootstrap_stat(num, inference.match_presence(match, IDS), matrix))
    mean = inference.paired_seed_mean(draws)
    assert row['ci95'] == pytest.approx(np.quantile(mean, [0.025, 0.975]).tolist(), rel=1e-12)
    assert row['p'] == pytest.approx(inference.two_sided_p(mean, row['delta']), rel=1e-12)
    assert row['adjusted_p'] is None and 'Holm' in row['adjusted_p_note']                              # C4 is outside the D15/D20 Holm family: no family-adjusted p (K6b m6)
    assert 'verdict' not in row and row['rule'] in run_evaluate.C4_RULES                                # the D13 decision is ``rule``; the D15 claim word belongs to the family rows (review concern 5)
    assert row['bootstrap_flag'] is not None and '2,000' in row['bootstrap_flag']                    # B = 300 is not the 2,000 rows that D13 fixes (K6b m5)
    assert json.loads(json.dumps(row)) == row                                                       # strict JSON: no numpy scalar left


def test_c4_pairs_by_the_value_of_the_seed_not_by_the_position_and_refuses_other_seed_values():
    from c_ept.evaluation import inference
    found = c4_endpoints()
    matrix = inference.resample_matrix(len(IDS), B=100)
    reordered = {seed: found[seed] for seed in reversed(SEEDS)}                                     # the same arms in another order
    assert run_evaluate.c4_statistic(reordered, IDS, matrix) == run_evaluate.c4_statistic(found, IDS, matrix)
    with pytest.raises(ValueError, match='seed'):
        run_evaluate.c4_statistic(c4_endpoints(seeds=(SEEDS[0], SEEDS[1], SEEDS[2] + 1)), IDS, matrix)
    with pytest.raises(ValueError, match='seed'):
        run_evaluate.c4_statistic({seed: found[seed] for seed in SEEDS[:2]}, IDS, matrix)


def test_c4_stops_when_the_point_estimate_is_not_better_proceeds_when_the_interval_is_below_zero_and_flags_inconclusive_otherwise():
    from c_ept.evaluation import inference
    matrix = inference.resample_matrix(len(IDS), B=400)
    worse = run_evaluate.c4_statistic(c4_endpoints((0.15, 0.10, 0.05)), IDS, matrix)                # EPT moved away from the outcome: Brier above B1's
    assert worse['delta'] > 0 and worse['rule'] == 'stop'
    same = run_evaluate.c4_statistic(c4_endpoints((0.0, 0.0, 0.0)), IDS, matrix)
    assert same['delta'] == 0.0 and same['rule'] == 'stop'                                           # a difference of exactly 0 is not better (>= 0 stops)
    better = run_evaluate.c4_statistic(c4_endpoints((-0.4, -0.4, -0.4)), IDS, matrix)
    assert better['delta'] < 0 and better['ci95'][1] < 0 and better['rule'] == 'proceed'
    # three matches much better and one match worse: the point is below 0 but a resample that draws the bad match twice is above it, so the interval reaches 0
    mixed = c4_endpoints((-0.4, -0.4, -0.4))
    for seed in SEEDS:
        y = mixed[seed]['y'][:3]
        mixed[seed]['p_ept'][:3] = np.where(y == 1, 0.2, 0.8)                                         # the first match: 0.3 away from 0.5, on the wrong side
    result = run_evaluate.c4_statistic(mixed, IDS, matrix)
    assert result['delta'] < 0 and result['ci95'][1] >= 0 and result['rule'] == 'proceed_inconclusive'


def test_c4_refuses_arms_that_are_not_the_same_rows_and_a_b1_that_differs_between_seeds():
    from c_ept.evaluation import inference
    matrix = inference.resample_matrix(len(IDS), B=50)
    found = c4_endpoints()
    other_rows = {seed: dict(found[seed]) for seed in SEEDS}
    other_rows[SEEDS[1]]['row'] = found[SEEDS[1]]['row'] + 1
    with pytest.raises(ValueError, match='rows'):
        run_evaluate.c4_statistic(other_rows, IDS, matrix)
    other_b1 = {seed: dict(found[seed]) for seed in SEEDS}
    other_b1[SEEDS[2]]['p_b1'] = found[SEEDS[2]]['p_b1'] * 0.9
    with pytest.raises(ValueError, match='B1'):
        run_evaluate.c4_statistic(other_b1, IDS, matrix)


def test_c4_with_a_resample_that_has_no_present_match_is_not_computed_and_says_why():
    """M10 n1: a row quantity that is present in few matches can give a resample with denominator 0; the ratio is undefined there, the row is not computed."""
    from c_ept.evaluation import inference
    ids = np.arange(0, 60, dtype=np.int64)
    found = c4_endpoints()                                                                            # all rows lie in the matches 10 .. 13 of a split of 60 matches
    matrix = inference.resample_matrix(len(ids), B=2000)
    with pytest.raises(run_evaluate.Unavailable, match='denominator'):
        run_evaluate.c4_statistic(found, ids, matrix)

# ---- the evaluation of a profile: the backend, the STOP evaluation, the records ---------------------------------------------------------------------
def eval_profile(factory, tmp_path, role='STOP', numbers=(1,), name='evaluate.json', over=None, **fields):
    folders = {f'seed{n}': str(seed_run(factory, n, **(over or {}))) for n in numbers}
    return evaluate_profile(tmp_path, name, role=role, models=list(folders), checkpoints=folders, **fields)


def structural_records(seed, shift=0.0):
    """The eight structural NLL records of ``run_inference(b0=)`` in the made-up split: ``nll_b0`` of 1.0, 1.1, ...; the model at a seed-dependent fraction of it."""
    from c_ept.evaluation import metrics
    records = []
    for index, name in enumerate(metrics.RELATIVE_HEADS):
        match = np.repeat(IDS, 2)
        base = np.full(len(match), 1.0 + 0.1 * index)
        records.append({'name': name, 'match': match, 'row': np.arange(len(match), dtype=np.int64), 'nll_b0': base,
                        'nll_ept': base * (0.9 + 0.01 * (seed - SEEDS[0]) + shift), 'counts': {}})
    return records


class FakeBackend(run_evaluate.Backend):
    """The real backend with the heavy parts replaced: no model is built (unless ``real_model``), the split is the made-up one, the inference answers with the endpoints
    of ``c4_endpoints`` and the NLL records of ``structural_records``, and every call is logged with the state of the access ledger at that moment."""

    def __init__(self, access=None, real_model=False, real_stop=False, vstart=None, ids=IDS, unavailable=None, endpoint=None):
        self.log, self.access, self.real_model, self.real_stop, self.vstart, self.ids, self.unavailable = [], access, real_model, real_stop, vstart, ids, unavailable
        self.passes, self.endpoint, self.openings = [], endpoint, []

    def evaluates(self, role):
        return True                                                                                   # a fake that answers for every role, TEST too (the default backend cannot until a TEST reader of the stores exists: S8)

    def _note(self, kind, trained, role=None):
        opened = None if self.access is None else [e['event'] for e in self.access.entries()]
        self.log.append((kind, trained.name, role, opened))

    def build_model(self, trained, static, device):
        self._note('build', trained)
        return super().build_model(trained, static, device) if self.real_model else object()

    def split_ids(self, inputs, role):
        return self.ids

    def stop_report(self, model, trained, inputs, constants, check):
        self._note('stop', trained, 'STOP')
        if self.real_stop:
            return super().stop_report(model, trained, inputs, constants, check)
        return {'loss_stop': 1.5, 'blocks': {'M2Rate/masked': {'mean': 1.0, 'scale': 2.0, 'term': 0.5}, 'M2Mark/masked': {'mean': 1.0, 'scale': 2.0, 'term': 0.5}}}

    def inference(self, model, trained, inputs, *, role, profile, reps, constants, check, exposure='observed', population='all', opening=None, progress=None):
        self._note('inference', trained, role)
        self.passes.append((trained.name, exposure, population))
        self.openings.append((role, opening))
        if self.unavailable:
            raise run_evaluate.Unavailable(self.unavailable)
        found = c4_endpoints()[trained.seed]
        if self.endpoint is not None:
            found = self.endpoint(trained, exposure, population, found)
        return {'endpoints': [found], 'nll': structural_records(trained.seed), 'provenance': {'kill_law': trained.settings['kill_law'], 'exposure': exposure, 'population': population}}

    def v_start(self, model, trained, inputs, *, role, profile, reps, constants, check, progress=None):
        self._note('v_start', trained, role)
        if self.vstart is None:
            raise run_evaluate.Unavailable('no composed V_start in this fake')
        return self.vstart(trained)


def fake_sources(**over):
    """The ``sources`` that a record names for an evaluation with ``FakeBackend`` and the made-up loader of ``FAKE``."""
    return {'loader': FAKE['loader'], 'backend': f'{FakeBackend.__module__}:{FakeBackend.__qualname__}', 'manifest_sha256': FAKE['manifest_sha256'], 'allow_custom': True,
            'body': FAKE_BODY, **over}


FAKE_TEST = {'loader': FAKE['loader'], 'manifest_sha256': FAKE['manifest_sha256'], 'body': FAKE_BODY}        # the role TEST never takes allow_custom (review r1); it names its stage body (K6c)


@pytest.fixture
def fake_defaults(monkeypatch):
    """The made-up loader and ``FakeBackend`` stand for the default loader and backend of this module (``LOADER``, ``BACKEND``), so that the role TEST, which refuses the test-only switch
    ``allow_custom``, can run the toy backend. Only the test process is changed: no code path of ``start`` or of a command line does it (review r1)."""
    monkeypatch.setattr(run_evaluate, 'LOADER', FAKE['loader'])
    monkeypatch.setattr(run_evaluate, 'BACKEND', f'{FakeBackend.__module__}:{FakeBackend.__qualname__}')


def real_models(factory, numbers=(1, 2, 3)):
    """``{model: train_record_sha256}`` of the cached training runs: what the training records of a rehearsal must name for the same checkpoints."""
    return {f'seed{n}': run_evaluate.load_trained(f'seed{n}', seed_run(factory, n)).record_sha256 for n in numbers}


def b1_of(factory):
    from types import SimpleNamespace
    return SimpleNamespace(dropped=tuple(run_evaluate.load_trained('seed1', seed_run(factory, 1)).stats.dropped))


def test_stop_evaluation_uses_whole_stop_and_best_checkpoint(factory, tmp_path, monkeypatch):
    """D13: every S4b decision is measured on the whole STOP split with each fit's best checkpoint. The STOP rows are made without a match selection (no proxy
    matches), the model is the ``best.pt`` named by the head (nothing else is loaded) and the blocks are those of ``train.evaluate_stop_blocks`` for the model's own
    kill law, frame average and zone mark."""
    from c_ept.ept import encoder, heads
    profile = eval_profile(factory, tmp_path)
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    seen, loads, selections = {}, [], []
    real_blocks, real_load, real_rows = ept_train.evaluate_stop_blocks, torch.load, data.EngagementRows

    def spy(model, stop, scales, **kwargs):
        seen.update(model=model, stop=stop, scales=scales, kwargs=kwargs)
        return real_blocks(model, stop, scales, **kwargs)
    monkeypatch.setattr(ept_train, 'evaluate_stop_blocks', spy)
    monkeypatch.setattr(torch, 'load', lambda path, *a, **k: loads.append(Path(path)) or real_load(path, *a, **k))

    class Tracking(real_rows):                                                                       # a subclass: ``Batches`` checks ``isinstance`` against the name in ``data``
        def __init__(self, rows, matches=None):
            selections.append(matches)
            super().__init__(rows, matches)
    monkeypatch.setattr(data, 'EngagementRows', Tracking)
    monkeypatch.setattr(data, 'proxy_matches', lambda *a, **k: pytest.fail('the whole STOP split is evaluated: no fraction of the matches is selected'))
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU)
    assert loads == [trained.best_path] and selections == [None]
    assert seen['stop'].rows.role == 'STOP' and seen['stop'].training is False and seen['scales'] == SCALES
    assert seen['kwargs'] == {'device': 'cpu', 'guard': seen['kwargs']['guard'], 'kill_law': 'shifted', 'eval_average': True, 'zone_mark': True}
    reference = heads.EPT(encoder.EncoderConfig(**trained.encoder), toy_static())
    reference.load_state_dict(real_load(trained.best_path, weights_only=True, map_location='cpu'))
    assert all(torch.equal(a, b) for a, b in zip(seen['model'].state_dict().values(), reference.state_dict().values()))
    stop = result['report']['models']['seed1']['stop']
    expected = real_blocks(reference, seen['stop'], SCALES, device='cpu', kill_law='shifted', eval_average=True, zone_mark=True)
    assert stop['loss_stop'] == expected['loss_stop'] and stop['blocks'] == json.loads(json.dumps(expected['blocks']))
    assert set(stop['blocks']) <= set(loss.BLOCKS) and result['report']['role'] == 'STOP' and result['record']['access'] is None
    assert run_io.read_json(tmp_path / 'run' / run_evaluate.REPORT_FILE) == result['report'] and run_io.read_json(tmp_path / 'run' / run_evaluate.RECORD_FILE) == result['record']


@pytest.mark.parametrize('over, key, expected', [({'zone_mark': False}, 'zone_mark', False), ({'eval_average': False}, 'eval_average', False)])
def test_stop_evaluation_gives_the_model_its_own_frame_average_and_zone_mark(factory, tmp_path, monkeypatch, over, key, expected):
    seen = {}
    real_blocks = ept_train.evaluate_stop_blocks

    def spy(model, stop, scales, **kwargs):
        seen.update(model=model, kwargs=kwargs)
        return real_blocks(model, stop, scales, **kwargs)
    monkeypatch.setattr(ept_train, 'evaluate_stop_blocks', spy)
    profile = eval_profile(factory, tmp_path, over=over)
    run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU)
    assert seen['kwargs'][key] is expected and seen['kwargs']['kill_law'] == 'shifted'
    if key == 'zone_mark':
        assert seen['model'].heads.zone_mark is False                                                # the model was built the way it was trained


def test_the_stop_evaluation_says_what_it_cannot_make_and_why(factory, tmp_path):
    """K6d: the STOP rows are read by the inference now, so what is missing here is only what the inputs do not hold: no prices, tier model or matched B1 (the default backend says
    so, not "DEV only"), hence no endpoints, no C4 and no structural metric. The D9 inputs of the STOP report itself are made. Prices and the tier model are not needed on STOP (no Monte Carlo, D30)."""
    from types import SimpleNamespace
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3), b0=True)
    inputs = eval_inputs(factory, b0=SimpleNamespace())                                                      # no matched B1
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=inputs, constants=CPU)
    unavailable = {item['item']: item['reason'] for item in result['record']['unavailable']}
    assert {'inference', 'structural', 'C4'} <= set(unavailable), unavailable
    assert 'no matched B1' in unavailable['inference'] and 'DEV only' not in unavailable['inference'] and 'role' not in unavailable['inference']
    assert 'inference' in unavailable['structural'] and 'matched 25% B1' in unavailable['C4']
    assert result['report']['c4'] is None and result['report']['rows'] == [] and result['record']['bootstrap'] is None            # no bootstrap was needed
    stop = result['report']['models']['seed1']['stop']
    # D9: the real STOP report of the toy world holds both shared M2 blocks (the zone term enters neither); they are reported separately, as the report holds them, and nothing is said
    # to be missing (review m2: no disjunction, the outcome that the real ``stop_report`` has)
    assert set(stop['shared_m2']) == {'M2Rate/masked', 'M2Mark/masked'} and all(stop['shared_m2'][key] == stop['blocks'][key] for key in stop['shared_m2'])
    masked = {key for key in stop['blocks'] if key.startswith('M3') and key.endswith('/masked')}
    assert masked and set(stop['masked_m3']) == masked and 'shared_m2' not in unavailable and 'masked_m3' not in unavailable      # the toy STOP rows score masked M3 blocks too


# ---- STOP through the real inference (K6d, review I3 of K6b): the endpoints, the NLL records, C4 and the structural metric on the whole STOP split ----------------
PLAIN = {'kill_law': 'poisson', 'k_condition': 'raw', 'decay_groups': 'all', 'swap_mode': 'average', 'on_wall_cap': 'fail', 'column_drop': 'none', 'stats_population': 'whole_fit'}


def plain_profile(factory, tmp_path, role='STOP', numbers=(1, 2, 3), name='plain.json', **fields):
    """The evaluate profile of the three seeds of the plain configuration (the plain kill law, the raw K condition): the one that the constant stand-in B1 of the toy world fits."""
    return eval_profile(factory, tmp_path, role=role, numbers=numbers, name=name, over=PLAIN, **fields)


def plain_inputs(factory, **extra):
    """Inputs for the real inference of the plain toy configuration: the stores with an empty objectives table and the labels the inference reads, flat prices, the toy tier model and a
    constant B1 with the dropped columns of the models; ``extra`` replaces fields of the ``EvalInputs`` (B0 and its files, for example)."""
    import dataclasses
    dropped = tuple(run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN)).stats.dropped)
    inputs = bodies.real_eval_inputs(factory.inputs_for({**S4B_FIELDS, **PLAIN}), dropped=dropped)
    return dataclasses.replace(inputs, **extra)


class RecordingBackend(run_evaluate.Backend):
    """The real default backend, remembering every result of its inference: ``(seed, role, exposure, population, result)``."""

    def __init__(self):
        self.results = []

    def inference(self, model, trained, inputs, **kwargs):
        result = super().inference(model, trained, inputs, **kwargs)
        self.results.append((trained.seed, kwargs['role'], kwargs['exposure'], kwargs['population'], result))
        return result


def stop_matches(world):
    return np.flatnonzero(np.asarray(world.s1.array('matches/role')) == C.ROLES.index('STOP'))


def test_the_stop_evaluation_makes_the_endpoints_the_nll_records_and_c4_from_the_real_inference_on_the_stop_rows(factory, world, tmp_path):
    """K6d item 3 and test 1: the default backend runs ``adapters.run_inference(role='STOP')`` for every seed on the whole STOP split; the endpoints and the NLL records are STOP rows, C4 is the row
    that ``c4_statistic`` makes of them, and the record says that both arms used STOP for early stopping."""
    from c_ept.evaluation import inference
    profile = plain_profile(factory, tmp_path)
    backend = RecordingBackend()
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=plain_inputs(factory), constants=CPU, backend=backend)
    assert [(seed, role, exposure, population) for seed, role, exposure, population, _ in backend.results] == [(seed, 'STOP', 'observed', 'all') for seed in SEEDS]
    stop = {int(m) for m in stop_matches(world)}
    assert stop and len(stop_matches(world)) == 1 and len(stop) == 1
    found = {}
    for seed, _, _, _, one in backend.results:
        assert one['provenance']['role'] == 'STOP' and 'for early stopping' in one['provenance']['role_note'] and 'cannot see' in one['provenance']['role_note']
        assert {int(m) for ep in one['endpoints'] for m in ep['match']} <= stop and {int(m) for record in one['nll'] for m in record['match']} <= stop
        assert sum(len(ep['row']) for ep in one['endpoints']) > 0 and len(one['nll']) == 13
        found[seed] = next(ep for ep in one['endpoints'] if ep['name'] == run_evaluate.C4_ENDPOINT)
    ids = run_evaluate.Backend().split_ids(plain_inputs(factory), 'STOP')
    assert [int(m) for m in ids] == sorted(stop)
    expected = run_evaluate.c4_statistic(found, ids, inference.resample_matrix(len(ids))[:2000])
    row = result['report']['c4']
    assert row == {**json.loads(json.dumps(expected)), 'fit_sha256': row['fit_sha256'], 'b1_folder_sha256': row['b1_folder_sha256']} and row['n_matches'] == 1 and row['rule'] in run_evaluate.C4_RULES
    for name in ('seed1', 'seed2', 'seed3'):
        part = result['report']['models'][name]
        assert part['passes']['all']['brier']['diff'] == pytest.approx(part['passes']['all']['brier']['ept'] - part['passes']['all']['brier']['b1'], rel=1e-12)
        assert part['inference_provenance']['role'] == 'STOP'
    unavailable = {item['item'] for item in result['record']['unavailable']}
    assert 'inference' not in unavailable and 'C4' not in unavailable
    for _, _, _, _, one in backend.results:                                                       # D30: closed form on STOP: no Monte Carlo, no M3 sign endpoint, and the record says so
        assert one['provenance']['monte_carlo'] is False and 'M3/open/positive_value' not in {ep['name'] for ep in one['endpoints']}
    assert result['record']['inference_constants']['monte_carlo'] is False
    selection = result['record']['selection_data']
    assert selection['role'] == 'STOP' and 'early stopping' in selection['note'] and selection['ept']['stop_sha256'] == result['record']['agreed']['stop_sha256']
    assert selection['b1']['stop_roles'] == ['STOP']
    # review I1: each arm used the STOP rows its own training run recorded; the runner cannot see B1's row set (a matched B1 stops on the same matched subset as the EPT, B1Heads records the role only)
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN))
    digest = {key: val for key, val in trained.identity['stop'].items() if key in ('role', 'batches', 'rows_sha256')}
    assert selection['ept']['early_stopping_rows'] == digest and digest['role'] == 'STOP' and digest['batches'] > 0 and len(digest['rows_sha256']) == 64
    note = json.dumps(selection)
    assert 'its own training run' in selection['note'] and 'cannot see' in selection['b1']['early_stopping_rows'] and 'B1Heads records the role only' in selection['b1']['early_stopping_rows']
    assert 'the STOP matches (fit_head stop_rows)' not in note and 'a subset of the STOP matches' not in note and 'the matched B1 fit_head(fit_rows, stop_rows)' not in note


def test_the_stop_evaluation_makes_the_relative_structural_metric_from_the_nll_records_with_b0(factory, tmp_path):
    """The structural metric R (D16) on STOP, end to end: a B0 fitted on FIT gives every NLL record its ``nll_b0`` and the metric is made on the common ids of the STOP split."""
    import warnings
    from c_ept.models import baselines_b0
    from c_ept.preprocessing import features
    base = bodies.real_eval_inputs(factory.inputs_for({**S4B_FIELDS, **PLAIN})).base
    means = features.b0_fit_means(base.s2, np.flatnonzero(features.split_views(base.s1, base.s2, 'FIT')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                                                          # the toy heads do not converge: the known warnings of the B0 tests
        b0 = baselines_b0.B0Heads.fit(base.s1, base.s2, base.labels, means)
    folder = b0.save(tmp_path / 'b0')
    files = {name: folder / name for name in (baselines_b0.JSON_NAME, baselines_b0.NPZ_NAME)}
    profile = plain_profile(factory, tmp_path, b0=True)
    backend = RecordingBackend()
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=plain_inputs(factory, b0=b0, b0_files=files, prices=None, tier=None), constants=CPU, backend=backend)       # no prices, no tier model: STOP draws nothing (D30)
    assert not [u for u in result['record']['unavailable'] if u['item'] in ('structural', 'inference', 'C4')], result['record']['unavailable']
    from c_ept.evaluation import metrics
    ids = np.asarray([int(m) for m in run_evaluate.Backend().split_ids(plain_inputs(factory), 'STOP')])
    for seed, name in zip(SEEDS, ('seed1', 'seed2', 'seed3')):
        records = next(one for s, _, _, _, one in backend.results if s == seed)['nll']
        assert all('nll_b0' in record for record in records)
        from c_ept.evaluation import inference
        expected = metrics.relative_structural(records, inference.resample_matrix(len(ids))[:2000], ids=ids)
        assert result['report']['models'][name]['structural'] == json.loads(json.dumps(expected, default=lambda obj: obj.tolist() if hasattr(obj, 'tolist') else float(obj)))


def test_a_test_evaluation_hands_the_opening_of_the_gate_to_the_backend_and_the_other_roles_hand_none(factory, tmp_path, fake_defaults):
    """The ``HoldoutOpening`` that ``HoldoutGate.open`` returns goes with every inference of the role TEST (and only of it): the backend of the run receives the very entry of the ledger."""
    fields = {'models': ['seed1', 'seed2', 'seed3'], 'checkpoints': {f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)}}
    world = GateWorld(tmp_path, **fields)
    world.rehearse(sources=fake_sources(allow_custom=False), models=real_models(factory))
    backend = FakeBackend()
    result = run_evaluate.evaluate(world.test, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=backend, gate=world.gate(), identity=IDENTITY_A,
                                   clock=world.clock, **FAKE_TEST)
    assert [role for role, _ in backend.openings] == ['TEST'] * 3
    entry = world.access.entries()[-1]
    assert entry['event'] == 'test_opened' and all(isinstance(opening, adapters_module().HoldoutOpening) and dict(opening) == entry for _, opening in backend.openings)
    assert result['record']['access']['entry_sha256'] == run_io.canonical_sha256(entry)
    stop = FakeBackend()
    run_evaluate.evaluate(eval_profile(factory, tmp_path, numbers=(1,), name='stop.json'), tmp_path / 'run_stop', inputs=eval_inputs(factory), constants=CPU, backend=stop)
    assert stop.openings == [('STOP', None)]
    (tmp_path / 'dev').mkdir()
    clock, access, profile = dev_setup(factory, tmp_path / 'dev')
    dev = FakeBackend()
    run_evaluate.evaluate(profile, tmp_path / 'run_dev', inputs=eval_inputs(factory), constants=CPU, backend=dev, access=access, identity=IDENTITY_A, clock=clock, **FAKE)
    assert [role for role, _ in dev.openings] == ['DEV'] * 3 and all(opening is None for _, opening in dev.openings)


def test_the_default_backend_refuses_the_role_test_without_the_opening_and_every_other_role_and_the_gate_can_open_test_only_for_the_stores_that_read_it(factory, tmp_path):
    """Unavailable (not an error of the run) for TEST without an opening and for a role that is none of DEV, STOP, TEST; with a genuine opening the capability check of ``run_inference``
    passes and the refusal comes from the stores, which have no reader of a holdout role (the second lock)."""
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN))
    inputs = plain_inputs(factory)
    profile = plain_profile(factory, tmp_path, numbers=(1,))
    backend = run_evaluate.Backend()
    for role, word in (('TEST', 'HoldoutGate.open'), ('FIT', 'not FIT'), ('CAL', 'not CAL')):
        with pytest.raises(run_evaluate.Unavailable, match=word):
            backend.inference(object(), trained, inputs, role=role, profile=profile, reps=None, constants=CPU, check=None, exposure='observed', population='all')
    world = GateWorld(tmp_path, models=['seed1', 'seed2', 'seed3'], checkpoints={f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)})
    world.rehearse(models=real_models(factory))
    opening = world.open(models=real_models(factory))
    assert isinstance(opening, adapters_module().HoldoutOpening)
    model = backend.build_model(trained, inputs.base.static, 'cpu')
    with pytest.raises(ValueError, match='holdout role'):
        backend.inference(model, trained, inputs, role='TEST', profile=profile, reps=None, constants=CPU, check=None, exposure='observed', population='all', opening=opening)


# ---- the durable per-stage progress and timing record (lead's addition to K6d): one flushed line per stage boundary, counts only, no metric ------------------
class StepClock:
    """A clock that a test advances by hand: every call returns a time 0.5 s after the call before."""

    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        self.now += 0.5
        return self.now


def progress_lines(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / run_evaluate.PROGRESS_FILE).read_text(encoding='utf-8').splitlines()]


def stage_log(tmp_path, name='run'):
    return run_evaluate.StageLog(tmp_path / name / run_evaluate.PROGRESS_FILE, wall=StepClock(1000.0), monotonic=StepClock(0.0))


def test_a_toy_run_writes_the_stage_lines_in_order_with_times_and_counts_and_no_metric(factory, tmp_path):
    """Every stage of an evaluation (identity and checks, the access, per model: the model, the STOP report, each inference pass, the structural metric, the V_start; then C4, the records and the
    whole run) leaves a ``begin`` and an ``end`` line in ``PROGRESS_FILE``, in order. The end line has the stage, the model key, the wall start and end, the elapsed seconds, the status and the counts;
    the counts are whole numbers and no line holds a number of a metric."""
    from types import SimpleNamespace
    log = stage_log(tmp_path)
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3), b0=True, bootstrap_B=30)
    inputs = eval_inputs(factory, b1=b1_of(factory), b0=SimpleNamespace(), b0_files={})
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=inputs, constants=CPU, backend=FakeBackend(), progress=log)
    lines = progress_lines(tmp_path / 'run')
    leaf = [('identity', None)] + [(stage, model) for model in ('seed1', 'seed2', 'seed3') for stage in ('build_model', 'stop_report', 'inference', 'structural', 'v_start')]
    sequence = [('begin', 'evaluate', None), ('begin', 'identity', None), ('end', 'identity', None)]
    sequence += [(event, stage, model) for stage, model in leaf[1:] for event in ('begin', 'end')]
    sequence += [(event, stage, None) for stage in ('c4', 'records') for event in ('begin', 'end')] + [('end', 'evaluate', None)]
    assert [(line['event'], line['stage'], line['model']) for line in lines] == sequence
    ends = [line for line in lines if line['event'] == 'end']
    assert all(line['schema'] == run_evaluate.PROGRESS_SCHEMA and line['pid'] == os.getpid() and line['start'] < line['end'] and line['elapsed_s'] > 0 for line in ends)
    assert all(line['elapsed_s'] == pytest.approx(0.5) for line in ends if line['stage'] != 'evaluate')                       # a stage that holds no other stage: two samples of the monotonic clock
    begins = [line['start'] for line in lines if line['event'] == 'begin']
    assert begins == sorted(begins) and len(set(begins)) == len(begins)
    by = {(line['stage'], line['model']): line for line in ends}
    assert by[('inference', 'seed1')]['counts'] == {'endpoint_rows': len(c4_endpoints()[SEEDS[0]]['row']), 'block_records': len(structural_records(SEEDS[0]))}          # no 'draws': the fake's provenance does not say how many (review M2: the counts come from the call)
    assert by[('inference', 'seed1')]['detail'] == {'role': 'STOP', 'pass': 'all', 'exposure': 'observed', 'population': 'all'} and by[('inference', 'seed1')]['status'] == 'ok'
    assert by[('v_start', 'seed2')]['status'] == 'unavailable' and by[('v_start', 'seed2')]['error'] == 'Unavailable'                       # reported as unavailable, as in the record
    assert by[('identity', None)]['counts'] == {'models': 3} and by[('c4', None)]['counts'] == {'bootstrap_rows': 30, 'matches': len(IDS)} and by[('evaluate', None)]['status'] == 'ok'
    assert by[('evaluate', None)]['elapsed_s'] > sum(by[(stage, model)]['elapsed_s'] for stage, model in leaf)
    for line in lines:                                                                            # counts only: whole numbers, and nothing that names or holds a number of a metric
        assert all(isinstance(val, int) and not isinstance(val, bool) and val >= 0 for val in line.get('counts', {}).values())
        assert all(isinstance(val, (str, int)) and not isinstance(val, bool) for val in line['detail'].values())
        assert not any(word in key for key in (*line, *line.get('counts', {}), *line['detail']) for word in ('nll', 'brier', 'crps', 'loss', 'delta', 'ci95', 'p_value'))
    assert result['report']['c4'] is not None and json.dumps(lines, allow_nan=False)


def test_a_run_that_stops_in_a_stage_leaves_the_lines_up_to_that_stage_and_names_the_failure(factory, tmp_path):
    """A stage that raises (here the inference of the second seed) leaves its begin line and an end line with the status ``failed`` and the exception class (never the message, which may name a path
    or a key), then the enclosing stage fails too; nothing of the later stages is there. A kill that no code sees (``begin`` without ``end``) is the same file one line shorter."""
    class Breaking(FakeBackend):
        def inference(self, model, trained, inputs, **kwargs):
            if trained.name == 'seed2':
                raise RuntimeError('boom at KR_1234567890')
            return super().inference(model, trained, inputs, **kwargs)
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3))
    with pytest.raises(RuntimeError, match='boom'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=Breaking(), progress=stage_log(tmp_path))
    lines = progress_lines(tmp_path / 'run')
    assert [(line['event'], line['stage'], line['model'], line['status'] if line['event'] == 'end' else None) for line in lines] == [
        ('begin', 'evaluate', None, None), ('begin', 'identity', None, None), ('end', 'identity', None, 'ok'),
        ('begin', 'build_model', 'seed1', None), ('end', 'build_model', 'seed1', 'ok'), ('begin', 'stop_report', 'seed1', None), ('end', 'stop_report', 'seed1', 'ok'),
        ('begin', 'inference', 'seed1', None), ('end', 'inference', 'seed1', 'ok'), ('begin', 'v_start', 'seed1', None), ('end', 'v_start', 'seed1', 'unavailable'),
        ('begin', 'build_model', 'seed2', None), ('end', 'build_model', 'seed2', 'ok'), ('begin', 'stop_report', 'seed2', None), ('end', 'stop_report', 'seed2', 'ok'),
        ('begin', 'inference', 'seed2', None), ('end', 'inference', 'seed2', 'failed'), ('end', 'evaluate', None, 'failed')]
    failed = [line for line in lines if line['event'] == 'end' and line['status'] == 'failed']
    assert [line['error'] for line in failed] == ['RuntimeError', 'RuntimeError'] and 'KR_1234567890' not in (tmp_path / 'run' / run_evaluate.PROGRESS_FILE).read_text(encoding='utf-8')
    with pytest.raises(KeyboardInterrupt):                                                         # an interrupt is a failure of the stage too, and the file is kept
        run_evaluate.evaluate(profile, tmp_path / 'run2', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, progress=stage_log(tmp_path, 'run2'),
                              backend=type('Interrupted', (FakeBackend,), {'build_model': lambda self, *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())})())
    assert [line['error'] for line in progress_lines(tmp_path / 'run2') if line['event'] == 'end' and line['status'] == 'failed'] == ['KeyboardInterrupt', 'KeyboardInterrupt']


def test_every_line_is_flushed_when_it_is_written_so_that_a_killed_run_keeps_its_progress(tmp_path):
    """The ``begin`` line is on disk while the stage is still running (read back from inside it); the ``end`` line follows at the exit; the file is appended to, never rewritten (a resumed run
    adds to it)."""
    log = stage_log(tmp_path)
    with log.stage('inference', 'seed1', detail={'role': 'STOP'}, rows=7) as counts:
        inside = progress_lines(tmp_path / 'run')
        counts['draws'] = 12
    assert [(line['event'], line['stage'], line['model']) for line in inside] == [('begin', 'inference', 'seed1')] and inside[0]['detail'] == {'role': 'STOP'}
    after = progress_lines(tmp_path / 'run')
    assert after[0] == inside[0] and after[1]['event'] == 'end' and after[1]['counts'] == {'rows': 7, 'draws': 12} and after[1]['elapsed_s'] == pytest.approx(0.5)
    again = run_evaluate.StageLog(tmp_path / 'run' / run_evaluate.PROGRESS_FILE)
    with again.stage('records'):
        pass
    assert [line['stage'] for line in progress_lines(tmp_path / 'run')] == ['inference', 'inference', 'records', 'records']


@pytest.mark.parametrize('counts, detail', [({'rows': 1.5}, {}), ({'rows': True}, {}), ({'rows': -1}, {}), ({'rows': float('nan')}, {}), ({'rows': '7'}, {}), ({}, {'nll': 0.25}), ({}, {'role': None}), ({}, {'role': True})])
def test_a_progress_line_takes_whole_counts_and_plain_details_only(tmp_path, counts, detail):
    """The log is no place for a metric: counts are whole numbers of at least 0 and details are texts or whole numbers; anything else is refused before a line is written."""
    log = stage_log(tmp_path)
    with pytest.raises(ValueError, match='progress'):
        with log.stage('inference', detail=detail, **counts):
            pass
    assert not (tmp_path / 'run' / run_evaluate.PROGRESS_FILE).exists()
    with pytest.raises(ValueError, match='progress'):                                              # also a count that is set inside the stage
        with log.stage('inference') as inside:
            inside['rows'] = 2.5
    assert [line['status'] for line in progress_lines(tmp_path / 'run') if line['event'] == 'end'] == ['failed']


def test_an_unavailable_inference_is_a_status_of_its_line_and_the_run_goes_on(factory, tmp_path):
    result = run_evaluate.evaluate(eval_profile(factory, tmp_path, numbers=(1,)), tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU,
                                   backend=FakeBackend(unavailable='no prices'), progress=stage_log(tmp_path))
    ends = {(line['stage'], line['model']): line for line in progress_lines(tmp_path / 'run') if line['event'] == 'end'}
    assert ends[('inference', 'seed1')]['status'] == 'unavailable' and ends[('inference', 'seed1')]['counts'] == {} and ends[('evaluate', None)]['status'] == 'ok'
    assert [item['reason'] for item in result['record']['unavailable'] if item['item'] == 'inference'] == ['no prices']


def test_the_real_inference_leaves_one_progress_line_pair_per_head_block_inside_its_inference_stage(factory, tmp_path):
    """Review M3: ``run_inference`` reports each of its 13 head blocks to the log (begin and end, whole-number counts rows and views), so that a run that hits a cap can be attributed to a block."""
    run_evaluate_log = stage_log(tmp_path)
    profile = plain_profile(factory, tmp_path, numbers=(1,))
    run_evaluate.evaluate(profile, tmp_path / 'run', inputs=plain_inputs(factory), constants=CPU, backend=RecordingBackend(), progress=run_evaluate_log)
    lines = progress_lines(tmp_path / 'run')
    names = [loss.block_key(head, kind) for head, kind in data.ROW_FIELDS]
    blocks = [line for line in lines if line['stage'] == 'block']
    assert [(line['event'], line['detail']['block']) for line in blocks] == [(event, name) for name in names for event in ('begin', 'end')]
    assert all(line['model'] == 'seed1' and line['detail']['role'] == 'STOP' and line['detail']['pass'] == 'all' for line in blocks)
    ends = [line for line in blocks if line['event'] == 'end']
    assert all(line['status'] == 'ok' and {'rows', 'views'} <= set(line['counts']) and 'draws' not in line['counts'] for line in ends) and ends[0]['counts']['rows'] > 0
    order = [(line['event'], line['stage']) for line in lines if line['stage'] in ('inference', 'block')]
    assert order[0] == ('begin', 'inference') and order[-1] == ('end', 'inference') and len(order) == 2 + 26
    inference_end = next(line for line in lines if line['stage'] == 'inference' and line['event'] == 'end')
    assert inference_end['counts']['draws'] == 0 and inference_end['counts']['endpoint_rows'] > 0                       # STOP: the call says that it drew nothing


def test_a_failure_inside_a_head_block_leaves_the_failed_block_line_and_the_stages_around_it(factory, tmp_path):
    log = stage_log(tmp_path)
    path = tmp_path / 'run' / run_evaluate.PROGRESS_FILE

    def check():                                                                                  # the guard of the inference: fails in the first batch of the block M2Rate/masked, after its begin line
        last = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()][-1] if path.exists() else {}
        if last.get('stage') == 'block' and last.get('event') == 'begin' and last['detail']['block'] == 'M2Rate/masked':
            raise RuntimeError('stop here')
    with pytest.raises(RuntimeError, match='stop here'):
        run_evaluate.evaluate(plain_profile(factory, tmp_path, numbers=(1,)), tmp_path / 'run', inputs=plain_inputs(factory), constants=CPU, backend=RecordingBackend(), progress=log,
                              check=check)
    lines = progress_lines(tmp_path / 'run')
    tail = [(line['event'], line['stage'], line['detail'].get('block'), line.get('status')) for line in lines[-4:]]
    assert tail == [('begin', 'block', 'M2Rate/masked', None), ('end', 'block', 'M2Rate/masked', 'failed'), ('end', 'inference', None, 'failed'), ('end', 'evaluate', None, 'failed')]
    assert [line['error'] for line in lines[-3:]] == ['RuntimeError'] * 3


def test_a_guard_that_fails_at_a_block_boundary_leaves_the_finished_block_ok_and_fails_the_inference_stage(factory, tmp_path):
    """K6d re-check R1: the guard at the boundary between two head blocks runs after the block before has been reported finished and before the next one begins, so its failure closes no block as failed:
    the finished block keeps its ``ok`` line with its counts, the next block has no line at all, and the failure is the one of the ``inference`` stage and of the run.

    해석: 블록 경계의 점검이 실패해도 이미 끝난 블록은 성공 줄과 개수를 그대로 가진다. 실패는 추론 단계와 실행 전체의 줄에만 남는다."""
    def run(folder, check):
        return run_evaluate.evaluate(plain_profile(factory, tmp_path, numbers=(1,)), tmp_path / folder, inputs=plain_inputs(factory), constants=CPU, backend=RecordingBackend(),
                                     progress=stage_log(tmp_path, folder), check=check)
    heard = []                                                                                    # the dry run: the number of progress lines that each guard call found on disk
    path = tmp_path / 'dry' / run_evaluate.PROGRESS_FILE
    run('dry', lambda: heard.append(len(path.read_text(encoding='utf-8').splitlines()) if path.exists() else 0))
    dry = progress_lines(tmp_path / 'dry')
    first, second = [loss.block_key(head, kind) for head, kind in data.ROW_FIELDS][:2]
    begin = next(i for i, line in enumerate(dry) if line['stage'] == 'block' and line['event'] == 'begin' and line['detail']['block'] == second)
    boundary = sum(1 for lines in heard if lines <= begin)                                           # the guards that ran before the begin line of the second block was written: the last one is the boundary
    calls = []

    def check():
        calls.append(1)
        if len(calls) == boundary:
            raise RuntimeError('stop at the boundary')
    with pytest.raises(RuntimeError, match='stop at the boundary'):
        run('wet', check)
    lines = progress_lines(tmp_path / 'wet')
    blocks = [line for line in lines if line['stage'] == 'block']
    assert [(line['event'], line['detail']['block'], line.get('status')) for line in blocks] == [('begin', first, None), ('end', first, 'ok')]
    assert blocks[1]['counts']['rows'] > 0 and blocks[1]['error'] is None
    assert [(line['stage'], line['status'], line['error']) for line in lines[-2:]] == [('inference', 'failed', 'RuntimeError'), ('evaluate', 'failed', 'RuntimeError')] and lines[-3] == blocks[1]


class ProvenanceBackend(FakeBackend):
    """The fake backend with a provenance of its own for the inference: what the call says about its Monte Carlo is what the record and the log must show."""

    def __init__(self, provenance, **kwargs):
        super().__init__(**kwargs)
        self.provenance = provenance

    def inference(self, *args, **kwargs):
        result = super().inference(*args, **kwargs)
        return {**result, 'provenance': {**result['provenance'], **self.provenance}}


@pytest.mark.parametrize('provenance, role, draws_in_counts, recorded', [({'monte_carlo': False, 'draws': 0}, 'STOP', 0, {'monte_carlo': False, 'draws': 0}),
                                                                     ({'draws': 12}, 'STOP', 12, {'monte_carlo': True, 'draws': 12}),
                                                                     ({'monte_carlo': False, 'draws': 0}, 'DEV', 0, {'monte_carlo': False, 'draws': 0}),
                                                                     ({'draws': 1000}, 'DEV', 1000, {'monte_carlo': True, 'draws': 1000})])
def test_the_draws_of_the_log_and_the_record_are_those_of_the_call_not_those_of_the_role(factory, tmp_path, provenance, role, draws_in_counts, recorded):
    """Review M2: a backend decides whether it draws; the line of the inference stage and ``inference_constants`` take ``monte_carlo`` and ``draws`` from the provenance of the call."""
    clock, access, profile = dev_setup(factory, tmp_path) if role == 'DEV' else (None, None, eval_profile(factory, tmp_path, numbers=(1,)))
    extra = dict(access=access, identity=IDENTITY_A, clock=clock, **FAKE) if role == 'DEV' else {}
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=ProvenanceBackend(provenance), progress=stage_log(tmp_path), **extra)
    inference = next(line for line in progress_lines(tmp_path / 'run') if line['stage'] == 'inference' and line['event'] == 'end' and line['model'] == 'seed1')
    assert inference['counts']['draws'] == draws_in_counts
    assert {key: result['record']['inference_constants'][key] for key in ('monte_carlo', 'draws')} == recorded


@pytest.mark.parametrize('backend, seed', [(lambda: FakeBackend(unavailable='no prices'), None),                                 # no inference call was made: nothing was drawn
                                           (lambda: ProvenanceBackend({'monte_carlo': False, 'draws': 0}), None),                # a call that drew nothing
                                           (lambda: ProvenanceBackend({'draws': 12}), run_evaluate.MC_SEED)])                    # a call that drew
def test_the_record_seed_is_that_of_the_draws_and_none_when_no_call_drew_anything(factory, tmp_path, backend, seed):
    """K6d re-check R2: ``inference_constants.seed`` is the seed of the Monte Carlo draws and None when no inference call drew anything, also when no call was made at all (every call unavailable).

    해석: 기록의 seed는 추첨에 쓴 씨앗이다. 추론 호출이 하나도 없었거나 호출이 아무것도 뽑지 않았으면 쓴 씨앗이 없으므로 None이다."""
    result = run_evaluate.evaluate(eval_profile(factory, tmp_path, numbers=(1,)), tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=backend(),
                                   progress=stage_log(tmp_path))
    assert result['record']['inference_constants']['seed'] == seed


def test_the_record_has_the_schema_version_that_names_its_new_keys(factory, tmp_path):
    """Review M7: ``agreed.stop_sha256``, ``selection_data`` and ``inference_constants.monte_carlo`` are new keys: the record schema is v2."""
    assert run_evaluate.RECORD_SCHEMA == 'ept-run-evaluate-record-v2'
    result = run_evaluate.evaluate(eval_profile(factory, tmp_path, numbers=(1,)), tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())
    assert result['record']['schema'] == 'ept-run-evaluate-record-v2' and {'selection_data', 'inference_constants'} <= set(result['record'])


def test_every_attempt_of_a_run_folder_starts_with_an_evaluate_line_that_has_its_counter_and_the_profile(factory, tmp_path):
    """Review M8: attempts append to one file; each line carries ``attempt`` (1, 2, ...: the number of ``evaluate`` begin lines in the file before it, plus one) and the ``evaluate`` begin line names the
    profile and the role, so two attempts can be told apart."""
    profile = eval_profile(factory, tmp_path, numbers=(1,))
    for _ in range(2):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())                  # the default log: one StageLog for each call
    lines = progress_lines(tmp_path / 'run')
    begins = [line for line in lines if line['stage'] == 'evaluate' and line['event'] == 'begin']
    assert [line['attempt'] for line in begins] == [1, 2] and all(line['detail'] == {'role': 'STOP', 'profile_sha256': profile.sha256} for line in begins)
    assert {line['attempt'] for line in lines[:len(lines) // 2]} == {1} and {line['attempt'] for line in lines[len(lines) // 2:]} == {2}


def test_a_supervised_run_writes_its_progress_in_its_own_folder_and_the_ledger_does_not_list_it(factory, world, tmp_path):
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='progress')
    assert rig.start() == 0, rig.stream.getvalue()
    lines = progress_lines(rig.run_dir)
    assert lines[0]['stage'] == 'evaluate' and lines[-1]['stage'] == 'evaluate' and lines[-1]['status'] == 'ok' and {line['model'] for line in lines} == {None, 'seed1', 'seed2', 'seed3'}
    assert run_evaluate.PROGRESS_FILE not in rig.ledger().stage('evaluate').outputs                  # an extra file, not a declared output: the stage hashes only its record and its report


def test_the_record_holds_the_effective_values_of_every_checkpoint(factory, tmp_path, reps_sha):
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3), winner_truth='binned', bootstrap_B=40)
    inputs = eval_inputs(factory, b1=b1_of(factory))
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=inputs, constants=CPU, backend=FakeBackend())
    record = result['record']
    assert record['schema'] == run_evaluate.RECORD_SCHEMA and record['role'] == 'STOP' and record['profile_sha256'] == profile.sha256
    assert record['profile_without_role_sha256'] == run_evaluate.profile_without_role_sha256(profile) and record['runner'] == CPU.to_record()
    assert record['winner_truth'] == 'binned' and record['exposure'] == {'exposure': 'observed', 'population': 'all', 'passes': [{'name': 'all', 'exposure': 'observed', 'population': 'all'}]}
    assert record['reps']['source'] == 'in_scope_fit' and record['reps']['sha256'] == reps_sha and isinstance(record['reps']['fallback'], list)
    assert record['scales']['condition']['k_reps_sha256'] == reps_sha and record['dropped']['columns'] == list(b1_of(factory).dropped)
    assert list(record['models']) == ['seed1', 'seed2', 'seed3'] and [m['seed'] for m in record['models'].values()] == list(SEEDS)
    for name, model in record['models'].items():
        trained = run_evaluate.load_trained(name, seed_run(factory, int(name[-1])))
        assert model['settings'] == dict(trained.settings) and model['train_record_sha256'] == trained.record_sha256
        assert model['best'] == trained.head['best'] and model['decay_groups'] == trained.record['decay_groups'] and model['termination'] == trained.record['termination']
        assert model['settings']['kill_law'] == 'shifted' and model['settings']['k_condition'] == 'binned' and 'identity' not in model
    assert record['agreed']['settings'] == dict(trained.settings) and record['agreed']['encoder'] == trained.encoder
    assert json.loads(json.dumps(record)) == record                                                  # strict JSON: no array, no numpy scalar


def test_a_raw_default_evaluation_records_gold_truth_and_no_representatives(factory, tmp_path):
    profile = eval_profile(factory, tmp_path, over=None)
    plain = evaluate_profile(tmp_path, 'plain.json', role='STOP', models=['plain'], checkpoints={'plain': str(factory.make('plain'))})
    result = run_evaluate.evaluate(plain, tmp_path / 'run', inputs=eval_inputs(factory, {}), constants=CPU, backend=FakeBackend())
    dropped = list(run_evaluate.load_trained('plain', factory.make('plain')).stats.dropped)
    assert result['record']['winner_truth'] == 'gold' and result['record']['reps'] is None
    assert result['record']['dropped'] == {'columns': dropped, 'sha256': run_io.canonical_sha256(dropped)}      # the models' own dropped lists are recorded also without a B1 (K6b I5)
    assert result['record']['models']['plain']['settings']['k_reps_sha256'] is None and profile['winner_truth'] == 'gold'


def test_the_winner_truth_of_a_raw_run_hangs_on_the_label_stores_representatives_and_the_record_says_so():
    """Under ``winner_truth='binned'`` the truth hangs on the 31 representatives; for the raw K condition they are ``labels/gold_reps``, and their hash goes to the record."""
    from types import SimpleNamespace
    reps = np.arange(31, dtype=np.float64) * 100.0
    labels = SimpleNamespace(array=lambda name: reps if name == 'labels/gold_reps' else pytest.fail(name))
    inputs = run_evaluate.EvalInputs(base=SimpleNamespace(labels=labels))
    assert run_evaluate.reps_record('binned', {'reps': None}, inputs) == {'source': 'labels', 'sha256': value.gold_reps_sha256(reps), 'fallback': None}
    assert run_evaluate.reps_record('gold', {'reps': None}, inputs) is None                           # nothing hangs on them
    checked = {'reps': {'sha256': 'a' * 64, 'fallback': [3], 'array': reps}}
    assert run_evaluate.reps_record('gold', checked, inputs) == {'source': 'in_scope_fit', 'sha256': 'a' * 64, 'fallback': [3]}

# ---- DEV: every access is logged (D4), before a number is made ----------------------------------------------------------------------------------
def dev_setup(factory, tmp_path, **fields):
    clock = FakeClock()
    access = run_evaluate.AccessLedger.create(tmp_path / 'access', clock=clock)
    profile = eval_profile(factory, tmp_path, role='DEV', numbers=(1, 2, 3), **fields)
    return clock, access, profile


def test_dev_access_is_logged(factory, tmp_path):
    """D4, D20: a DEV evaluation writes one entry to the access ledger before it makes a number; the entry names the profile, the code and the checkpoints and says that
    the numbers were computed, not opened."""
    clock, access, profile = dev_setup(factory, tmp_path)
    backend = FakeBackend(access=access)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=backend, access=access, identity=IDENTITY_A,
                                   clock=clock, **FAKE)
    entries = access.entries()
    assert [e['event'] for e in entries] == ['created', 'dev_access'] and entries[1]['seq'] == 1
    record = entries[1]['record']
    assert record['role'] == 'DEV' and record['profile_sha256'] == profile.sha256 and record['opened'] is False
    assert record['profile_without_role_sha256'] == run_evaluate.profile_without_role_sha256(profile)
    assert record['identity_sha256'] == run_io.canonical_sha256(IDENTITY_A.to_record()) and record['identity_commit'] == IDENTITY_A.commit
    assert record['models'] == {name: {'seed': seed, 'train_record_sha256': run_evaluate.load_trained(name, seed_run(factory, n)).record_sha256}
                                for n, (name, seed) in enumerate(zip(('seed1', 'seed2', 'seed3'), SEEDS), start=1)}
    assert result['record']['access'] == {'event': 'dev_access', 'seq': 1, 'entry_sha256': run_io.canonical_sha256(entries[1])}
    assert backend.log and all(call[3] == ['created', 'dev_access'] for call in backend.log)         # the line was there before the first model was built
    second = run_evaluate.evaluate(profile, tmp_path / 'run2', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(), access=access,
                                   identity=IDENTITY_A, clock=clock, **FAKE)
    assert [e['event'] for e in access.entries()] == ['created', 'dev_access', 'dev_access'] and second['record']['access']['seq'] == 2


def test_a_dev_evaluation_without_the_access_ledger_or_the_code_identity_is_refused_before_anything_is_read(factory, tmp_path):
    clock, access, profile = dev_setup(factory, tmp_path)
    backend = FakeBackend()
    with pytest.raises(run_evaluate.AccessRefused, match='access ledger'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=backend, identity=IDENTITY_A)
    with pytest.raises(run_evaluate.AccessRefused, match='identity'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=backend, access=access, **FAKE)
    assert backend.log == [] and [e['event'] for e in access.entries()] == ['created'] and not (tmp_path / 'run').exists()


def test_a_dev_access_that_ends_in_a_failure_stays_in_the_ledger_and_a_stop_evaluation_writes_none(factory, tmp_path):
    clock, access, profile = dev_setup(factory, tmp_path)

    class Failing(FakeBackend):
        def inference(self, *args, **kwargs):
            raise RuntimeError('boom')
    with pytest.raises(RuntimeError, match='boom'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=Failing(), access=access, identity=IDENTITY_A, clock=clock, **FAKE)
    assert [e['event'] for e in access.entries()] == ['created', 'dev_access']                       # the access happened, whatever came of it
    stop = eval_profile(factory, tmp_path, role='STOP', numbers=(1, 2, 3), name='stop.json')
    run_evaluate.evaluate(stop, tmp_path / 'run3', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend(), access=access, identity=IDENTITY_A, clock=clock, **FAKE)
    assert [e['event'] for e in access.entries()] == ['created', 'dev_access']                       # D13: DEV is not opened by a STOP evaluation


# ---- TEST: the gate (D20) ---------------------------------------------------------------------------------------------------------------------------
def test_the_test_role_evaluation_goes_through_the_gate_and_opens_once(factory, tmp_path, fake_defaults):
    fields = {'models': ['seed1', 'seed2', 'seed3'], 'checkpoints': {f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)}}
    world = GateWorld(tmp_path, **fields)
    world.rehearse(sources=fake_sources(allow_custom=False), models=real_models(factory))
    backend = FakeBackend(access=world.access)
    inputs = eval_inputs(factory, b1=b1_of(factory))
    result = run_evaluate.evaluate(world.test, tmp_path / 'run', inputs=inputs, constants=CPU, backend=backend, gate=world.gate(), identity=IDENTITY_A, clock=world.clock, **FAKE_TEST)
    assert world.access.test_opened() is True and result['record']['role'] == 'TEST'
    assert result['record']['access']['event'] == 'test_opened' and backend.log and all(call[3][-1] == 'test_opened' for call in backend.log)      # opened before the first model
    assert all(call[2] == 'TEST' for call in backend.log if call[0] == 'inference')
    again = FakeBackend()
    with pytest.raises(run_evaluate.AccessRefused, match='opened'):
        run_evaluate.evaluate(world.test, tmp_path / 'run2', inputs=inputs, constants=CPU, backend=again, gate=world.gate(), identity=IDENTITY_A, clock=world.clock, **FAKE_TEST)
    assert again.log == [] and [e['event'] for e in world.access.entries()].count('test_opened') == 1
    assert result['record']['sources'] == fake_sources(allow_custom=False)


def test_the_test_role_is_refused_without_the_gate_with_a_wrong_lock_or_other_code_and_nothing_is_opened(factory, tmp_path, fake_defaults):
    fields = {'models': ['seed1', 'seed2', 'seed3'], 'checkpoints': {f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)}}
    world = GateWorld(tmp_path, **fields)
    world.rehearse(sources=fake_sources(allow_custom=False), models=real_models(factory))
    backend = FakeBackend()
    inputs = eval_inputs(factory, b1=b1_of(factory))
    cases = [(dict(), 'gate'), (dict(gate=world.gate(lock_path=tmp_path / 'absent.json'), identity=IDENTITY_A), 'lock'), (dict(gate=world.gate()), 'identity'),
             (dict(gate=world.gate(), identity=IDENTITY_B), 'identity')]
    for extra, word in cases:
        with pytest.raises(run_evaluate.AccessRefused, match=word):
            run_evaluate.evaluate(world.test, tmp_path / 'run', inputs=inputs, constants=CPU, backend=backend, clock=world.clock, **extra, **FAKE_TEST)
    # nothing was computed or opened; since the progress record exists the run folder may hold that file (a refused run is a failed 'identity' or 'access' stage in it), and nothing else
    assert backend.log == [] and world.access.test_opened() is False and [path.name for path in (tmp_path / 'run').rglob('*') if path.is_file()] == ['PROGRESS.jsonl']
    ends = [line for line in progress_lines(tmp_path / 'run') if line['event'] == 'end']
    assert {line['stage'] for line in ends} <= {'identity', 'access', 'evaluate'} and all(line['status'] == 'failed' and line['error'] == 'AccessRefused' for line in ends if line['stage'] != 'identity')
    assert any(line['stage'] == 'identity' and line['status'] == 'failed' for line in ends) and any(line['stage'] == 'access' for line in ends)         # the absent lock fails the first, the code identity the second


def test_the_role_test_refuses_the_test_only_switch_and_a_made_up_loader_and_backend(factory, tmp_path, monkeypatch):
    """Review r1: the switch ``allow_custom`` is for DEV and STOP fixtures. A TEST run refuses it, however well the gate is served, and takes the default loader and backend only."""
    fields = {'models': ['seed1', 'seed2', 'seed3'], 'checkpoints': {f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)}}
    world = GateWorld(tmp_path, **fields)
    world.rehearse(sources=fake_sources(allow_custom=True), models=real_models(factory))
    inputs = eval_inputs(factory, b1=b1_of(factory))
    backend = FakeBackend()
    with pytest.raises(run_evaluate.EvaluationRefused, match='allow_custom'):                          # the switch, and the fake is not the default
        run_evaluate.evaluate(world.test, tmp_path / 'run', inputs=inputs, constants=CPU, backend=backend, gate=world.gate(), identity=IDENTITY_A, clock=world.clock, **FAKE)
    monkeypatch.setattr(run_evaluate, 'LOADER', FAKE['loader'])                                       # now the fake is the default: the switch is still refused for TEST
    monkeypatch.setattr(run_evaluate, 'BACKEND', f'{FakeBackend.__module__}:{FakeBackend.__qualname__}')
    with pytest.raises(run_evaluate.EvaluationRefused, match='allow_custom'):
        run_evaluate.evaluate(world.test, tmp_path / 'run2', inputs=inputs, constants=CPU, backend=backend, gate=world.gate(), identity=IDENTITY_A, clock=world.clock, **FAKE)
    assert backend.log == [] and world.access.test_opened() is False
    with pytest.raises(run_evaluate.AccessRefused, match='allow_custom'):                              # the gate itself refuses a rehearsal or a run that had the switch
        world.check(sources=fake_sources(allow_custom=True))


# ---- C4, the structural metric, V_start and B0 through the evaluation ----------------------------------------------------------------------------------
def test_c4_runs_on_the_first_rows_of_the_common_matrix_and_the_record_names_the_matrix(factory, tmp_path):
    from c_ept.evaluation import inference
    clock, access, profile = dev_setup(factory, tmp_path, bootstrap_B=60)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(), access=access,
                                   identity=IDENTITY_A, clock=clock, **FAKE)
    full = inference.resample_matrix(len(IDS))
    assert full.shape[0] == 10_000
    expected = run_evaluate.c4_statistic(c4_endpoints(), IDS, full[:60])
    expected.update(fit_sha256=result['record']['agreed']['fit_sha256'], b1_folder_sha256=result['record']['b1']['folder_sha256'])        # the two hashes that the runner puts beside the row
    assert result['report']['c4'] == json.loads(json.dumps(expected)) and result['report']['rows'] == [result['report']['c4']] and result['report']['c4']['bootstrap_B'] == 60
    boot = result['record']['bootstrap']
    assert boot['numpy'] == np.__version__ and boot['seed'] == inference.SEED and boot['matrix_rows'] == 10_000 and boot['B'] == 60
    assert boot['matrix_sha256'] == hashlib.sha256(full.tobytes()).hexdigest() and boot['used_sha256'] == hashlib.sha256(full[:60].tobytes()).hexdigest()
    assert boot['split'] == {'n_matches': 4, 'ids_sha256': hashlib.sha256(IDS.astype('<i8').tobytes()).hexdigest()}
    assert not [u for u in result['record']['unavailable'] if u['item'] in ('C4', 'structural')]


def test_the_default_b_of_c4_is_2000_the_first_rows_of_the_b_10000_matrix(factory, tmp_path):
    from c_ept.evaluation import inference
    assert evaluate_profile(tmp_path, role='STOP', models=['a'])['bootstrap_B'] == 2000
    clock, access, profile = dev_setup(factory, tmp_path)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(), access=access,
                                   identity=IDENTITY_A, clock=clock, **FAKE)
    assert result['report']['c4']['bootstrap_B'] == 2000 and result['record']['bootstrap']['matrix_rows'] == inference.B_DEFAULT == 10_000
    assert result['report']['c4']['bootstrap_flag'] is None                                          # the 2,000 rows of D13: nothing to flag (review m5)


def test_c4_needs_a_matched_b1_and_the_three_seed_values_and_says_why_otherwise(factory, tmp_path):
    clock, access, profile = dev_setup(factory, tmp_path)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend(), access=access, identity=IDENTITY_A, clock=clock, **FAKE)
    reasons = {u['item']: u['reason'] for u in result['record']['unavailable']}
    assert result['report']['c4'] is None and 'matched' in reasons['C4']
    one = eval_profile(factory, tmp_path, role='STOP', numbers=(1,), name='one.json')
    result = run_evaluate.evaluate(one, tmp_path / 'run2', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend())
    assert result['report']['c4'] is None and 'seed values' in {u['item']: u['reason'] for u in result['record']['unavailable']}['C4']


def test_the_relative_structural_metric_is_made_per_model_on_the_common_ids_and_b0_files_are_hashed_from_the_disk(factory, tmp_path):
    from types import SimpleNamespace
    from c_ept.evaluation import inference, metrics
    files = {'b0_heads.json': tmp_path / 'b0_heads.json', 'b0_heads.npz': tmp_path / 'b0_heads.npz'}
    files['b0_heads.json'].write_bytes(b'{"b0": 1}\n')
    files['b0_heads.npz'].write_bytes(b'npz bytes')
    clock, access, profile = dev_setup(factory, tmp_path, b0=True, bootstrap_B=30)
    inputs = eval_inputs(factory, b1=b1_of(factory), b0=SimpleNamespace(), b0_files=files)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=inputs, constants=CPU, backend=FakeBackend(), access=access, identity=IDENTITY_A, clock=clock, **FAKE)
    full = inference.resample_matrix(len(IDS))
    for seed, name in zip(SEEDS, ('seed1', 'seed2', 'seed3')):
        expected = metrics.relative_structural(structural_records(seed), full[:30], ids=IDS)
        assert result['report']['models'][name]['structural'] == json.loads(json.dumps(expected))
    assert result['record']['b0']['files_sha256'] == {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    with pytest.raises(run_evaluate.EvaluationRefused, match='B0'):                                   # the profile asks for B0 and there is none
        run_evaluate.evaluate(profile, tmp_path / 'run2', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend(), access=access, identity=IDENTITY_A, clock=clock, **FAKE)


def test_a_file_of_the_b0_set_that_cannot_be_read_is_refused(factory, tmp_path):
    from types import SimpleNamespace
    clock, access, profile = dev_setup(factory, tmp_path, b0=True)
    inputs = eval_inputs(factory, b0=SimpleNamespace(), b0_files={'b0_heads.json': tmp_path / 'absent.json'})
    with pytest.raises(run_evaluate.EvaluationRefused, match='b0_heads.json'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=inputs, constants=CPU, backend=FakeBackend(), access=access, identity=IDENTITY_A, clock=clock, **FAKE)


def test_the_v_start_crps_of_a_model_is_the_match_balanced_crps19_of_its_rows(factory, tmp_path):
    from c_ept.evaluation import proper
    q19 = np.linspace(0.0, 90.0, 19)[None, :] + np.arange(12)[:, None]
    realized = np.linspace(-20.0, 140.0, 12)
    match = np.repeat(IDS, 3)

    def vstart(trained):
        return {'q19': q19 + (trained.seed - SEEDS[0]), 'realized': realized, 'match': match}
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3))
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend(vstart=vstart))
    for seed, name in zip(SEEDS, ('seed1', 'seed2', 'seed3')):
        expected = proper.crps19(q19 + (seed - SEEDS[0]), realized, np.full(12, 1.0 / 3.0))
        assert result['report']['models'][name]['v_start'] == {'crps19': expected, 'n_rows': 12, 'n_matches': 4}


def test_an_inference_that_is_unavailable_is_reported_with_its_reason_and_the_other_parts_still_run(factory, tmp_path):
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3))
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(unavailable='no prices'))
    reasons = [u for u in result['record']['unavailable'] if u['item'] == 'inference']
    assert len(reasons) == 3 and all(u['reason'] == 'no prices' for u in reasons) and result['report']['models']['seed1']['stop']['loss_stop'] == 1.5


# ---- the default backend hands every setting of the checkpoint to run_inference ---------------------------------------------------------------------
def test_the_default_backend_passes_the_checkpoints_own_settings_to_run_inference(factory, tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(adapters_module(), 'run_inference', lambda *args, **kwargs: calls.append((args, kwargs)) or {'endpoints': [], 'nll': [], 'provenance': {}})
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    base = factory.inputs_for(S4B_FIELDS)
    b1, b0, prices, tier = SimpleNamespace(tag='b1'), SimpleNamespace(tag='b0'), {'p': 1.0}, SimpleNamespace(tag='tier')
    inputs = run_evaluate.EvalInputs(base=base, prices=prices, tier=tier, b1=b1, b0=b0)
    for exposure, population, b0_given, role in (('observed', 'all', False, 'DEV'), ('nominal', 'primary', True, 'STOP')):
        profile = evaluate_profile(tmp_path, f'{exposure}.json', role=role, models=['seed1'], checkpoints={'seed1': str(trained.folder)}, exposure=exposure, winner_truth='binned', b0=b0_given)
        marker = object()
        run_evaluate.Backend().inference(marker, trained, inputs, role=role, profile=profile, reps=None, constants=CPU, check=None, exposure=exposure, population=population)
        args, kwargs = calls[-1]
        assert args[1] is marker and args[2] is trained.stats and args[3] is b1 and args[0].role == role and args[0].prices is prices and args[0].tier is tier
        assert (kwargs['role'], kwargs['holdout']) == (role, None) and 'b1_swap_average' not in kwargs                  # the role of the call; the M13 field stays unwired (refused before)
        assert kwargs['monte_carlo'] is (role != 'STOP')                                                               # D30: the closed-form STOP path draws nothing, the DEV rehearsal and TEST do
        assert (kwargs['kill_law'], kwargs['k_condition'], kwargs['gold_reps'], kwargs['winner_truth']) == ('shifted', 'binned', 'in_scope_fit', 'binned')
        assert (kwargs['eval_average'], kwargs['zone_mark'], kwargs['exposure'], kwargs['population']) == (True, True, exposure, population)
        assert kwargs['b0'] is (b0 if b0_given else None) and kwargs['scales'] is base.scales and kwargs['device'] == 'cpu' and kwargs['batch_size'] == 32
        assert (kwargs['draws'], kwargs['seed']) == (run_evaluate.MC_DRAWS, run_evaluate.MC_SEED) == (1000, 20261022) and callable(kwargs['check'])
    plain = run_evaluate.load_trained('plain', factory.make('plain'))
    profile = evaluate_profile(tmp_path, 'plain.json', role='DEV', models=['plain'], checkpoints={'plain': str(plain.folder)})
    run_evaluate.Backend().inference(object(), plain, run_evaluate.EvalInputs(base=base, prices=prices, tier=tier, b1=b1), role='DEV', profile=profile, reps=None, constants=CPU,
                                     check=None, exposure='observed', population='all')
    kwargs = calls[-1][1]
    assert (kwargs['kill_law'], kwargs['k_condition'], kwargs['gold_reps'], kwargs['winner_truth']) == ('poisson', 'raw', 'labels', 'gold')


def adapters_module():
    from c_ept.evaluation import adapters
    return adapters


def test_the_composer_settings_follow_the_checkpoint_and_the_profile_and_name_real_parameters():
    import inspect
    from c_ept.valuation import compose
    settings = {'kill_law': 'shifted', 'zone_mark': False}
    assert run_evaluate.composer_settings(settings, 'nominal') == {'kill_law': 'shifted', 'zone_mark': False, 'exposure': 'nominal'}
    assert set(run_evaluate.composer_settings(settings, 'observed')) <= set(inspect.signature(compose.Composer).parameters)


def test_the_default_backend_leaves_the_composed_v_start_and_the_inference_of_other_roles_unavailable(factory, tmp_path):
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    profile = eval_profile(factory, tmp_path)
    inputs = run_evaluate.EvalInputs(base=factory.inputs_for(S4B_FIELDS), prices={'p': 1.0}, tier=object(), b1=object())            # everything the DEV inference needs
    with pytest.raises(run_evaluate.Unavailable, match='Composer'):
        run_evaluate.Backend().v_start(object(), trained, inputs, role='STOP', profile=profile, reps=None, constants=CPU, check=None)
    with pytest.raises(run_evaluate.Unavailable, match='HoldoutGate.open'):                                         # TEST without the opening of the gate
        run_evaluate.Backend().inference(object(), trained, inputs, role='TEST', profile=profile, reps=None, constants=CPU, check=None, exposure='observed', population='all')
    with pytest.raises(run_evaluate.Unavailable, match='no matched B1'):
        run_evaluate.Backend().inference(object(), trained, eval_inputs(factory), role='DEV', profile=profile, reps=None, constants=CPU, check=None, exposure='observed',
                                         population='all')



# ---- the evaluation inputs manifest and its loader ---------------------------------------------------------------------------------------------------
def eval_manifest_document(world, scales_path, **changes) -> dict:
    document = {'schema': 'ept-run-eval-inputs-v1', 's1': str(world.s1.root), 's2': str(world.s2.root), 'labels': str(world.labels.root), 'terrain': None,
                'scales': str(scales_path), 'scales_provenance': None, 'prices': None, 'tier': None, 'b1': None, 'b0': None}
    document.update(changes)
    return document


def write_eval_manifest(folder, document, name='EVAL_INPUTS.json'):
    path = Path(folder) / name
    path.write_bytes(json.dumps(document, indent=1).encode('utf-8'))
    return path


def test_the_eval_manifest_names_the_stores_the_scales_the_prices_the_tier_and_the_b1_and_b0_folders(world, tmp_path):
    scales_path = write_scales(tmp_path, 'scales.json')
    path = write_eval_manifest(tmp_path, eval_manifest_document(world, scales_path))
    manifest, digest = run_evaluate.read_eval_manifest(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest() and manifest['prices'] is None and manifest['b0'] is None
    full = eval_manifest_document(world, scales_path, prices=str(tmp_path / 'PRICES.json'), tier=str(tmp_path / 'tier'), b1=str(tmp_path / 'b1'), b0=str(tmp_path / 'b0'))
    assert run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, full, 'full.json'))[0]['b1'] == str(tmp_path / 'b1')


@pytest.mark.parametrize('change, word', [({'schema': 'other'}, 'schema'), ({'extra': 1}, 'fields'), ({'s1': 'relative/path'}, 's1'), ({'prices': 'rel.json'}, 'prices'),
                                          ({'b1': 7}, 'b1'), ({'b0': '../b0'}, 'b0'), ({'tier': 'a\x00b'}, 'tier'), ({'scales_provenance': [1]}, 'scales_provenance')])
def test_an_eval_manifest_that_is_not_acceptable_is_refused(world, tmp_path, change, word):
    scales_path = write_scales(tmp_path, 'scales.json')
    with pytest.raises(run_io.IORefused, match=word):
        run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, eval_manifest_document(world, scales_path, **change)))
    document = eval_manifest_document(world, scales_path)
    del document['b1']
    with pytest.raises(run_io.IORefused, match='fields'):
        run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, document, 'missing.json'))


def test_the_files_that_an_evaluation_binds_are_the_stores_the_scales_the_prices_and_every_file_of_the_tier_b1_and_b0_folders(world, tmp_path):
    for folder, names in (('tier', ('head.json', 'tier.npz')), ('b1', ('b1_heads.json', 'sub/head.json')), ('b0', ('b0_heads.json', 'b0_heads.npz'))):
        for name in names:
            target = tmp_path / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(name.encode())
    (tmp_path / 'PRICES.json').write_bytes(b'{}')
    scales_path = write_scales(tmp_path, 'scales.json')
    plain = run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, eval_manifest_document(world, scales_path)))[0]
    assert run_evaluate.manifest_bound_files(plain) == runners.manifest_files(plain)                 # nothing else to bind without prices, tier, B1 and B0
    full = run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, eval_manifest_document(
        world, scales_path, prices=str(tmp_path / 'PRICES.json'), tier=str(tmp_path / 'tier'), b1=str(tmp_path / 'b1'), b0=str(tmp_path / 'b0')), 'full.json'))[0]
    bound = run_evaluate.manifest_bound_files(full)
    assert bound == sorted(bound) and set(runners.manifest_files(full)) <= set(bound) and '\\' not in ''.join(bound)
    names = [p.rsplit('/', 1)[-1] for p in bound]
    for name in ('PRICES.json', 'tier.npz', 'b1_heads.json', 'b0_heads.npz'):
        assert name in names
    assert (tmp_path / 'b1' / 'sub' / 'head.json').as_posix() in bound and len(bound) == len(set(bound))


def test_the_extras_are_loaded_by_the_loaders_of_the_stacks_and_the_b0_files_are_named(world, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from c_ept.models import baselines_b0, baselines_b1
    from c_ept.valuation import towertier
    calls = []
    monkeypatch.setattr(value, 'load_prices', lambda path: calls.append(('prices', Path(path))) or {'tower': 1.0})
    monkeypatch.setattr(towertier.TierModel, 'load', classmethod(lambda cls, path: calls.append(('tier', Path(path))) or SimpleNamespace(tag='tier')))
    monkeypatch.setattr(baselines_b1.B1Heads, 'load', classmethod(lambda cls, path: calls.append(('b1', Path(path))) or SimpleNamespace(tag='b1')))
    monkeypatch.setattr(baselines_b0.B0Heads, 'load', classmethod(lambda cls, path: calls.append(('b0', Path(path))) or SimpleNamespace(tag='b0')))
    scales_path = write_scales(tmp_path, 'scales.json')
    plain = run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, eval_manifest_document(world, scales_path)))[0]
    assert run_evaluate.load_extras(plain) == {'prices': None, 'tier': None, 'b1': None, 'b0': None, 'b0_files': {}, 'b1_files': {}} and calls == []
    full = run_evaluate.read_eval_manifest(write_eval_manifest(tmp_path, eval_manifest_document(
        world, scales_path, prices=str(tmp_path / 'PRICES.json'), tier=str(tmp_path / 'tier'), b1=str(tmp_path / 'b1'), b0=str(tmp_path / 'b0')), 'full.json'))[0]
    extras = run_evaluate.load_extras(full)
    assert [name for name, _ in calls] == ['prices', 'tier', 'b1', 'b0'] and extras['prices'] == {'tower': 1.0} and extras['b1'].tag == 'b1' and extras['b0'].tag == 'b0'
    assert extras['b0_files'] == {baselines_b0.JSON_NAME: tmp_path / 'b0' / baselines_b0.JSON_NAME, baselines_b0.NPZ_NAME: tmp_path / 'b0' / baselines_b0.NPZ_NAME}
    assert extras['b1_files'] == {}                                                                  # the folder of the monkeypatched loader does not exist: nothing to hash
    (tmp_path / 'b1' / 'sub').mkdir(parents=True)
    (tmp_path / 'b1' / 'b1_heads.json').write_bytes(b'{}')
    (tmp_path / 'b1' / 'sub' / 'head.json').write_bytes(b'{}')
    assert run_evaluate.load_extras(full)['b1_files'] == {'b1_heads.json': tmp_path / 'b1' / 'b1_heads.json', 'sub/head.json': tmp_path / 'b1' / 'sub' / 'head.json'}


def test_the_real_loader_needs_a_terrain_like_the_training_loader(world, tmp_path):
    path = write_eval_manifest(tmp_path, eval_manifest_document(world, write_scales(tmp_path, 'scales.json')))
    with pytest.raises(run_io.IORefused, match='terrain'):
        run_evaluate.load_eval_inputs(path)                                                          # the toy manifest has no terrain


# ---- the stage body and the plan ---------------------------------------------------------------------------------------------------------------
TOY_LOADER = 'c_ept.tests.run_evaluate_bodies:toy_eval_inputs'
SYNTHETIC = 'c_ept.tests.run_evaluate_bodies:SyntheticBackend'
TOY_BODY = 'c_ept.tests.run_evaluate_bodies:evaluate_stage_with_toy_defaults'
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def toy_defaults(monkeypatch):
    """In this process the toy loader and backend are the defaults of ``run_evaluate`` (``LOADER``, ``BACKEND``): the parent of a supervised run checks the role against them. The child
    is given ``TOY_BODY``, which sets the same two names before it runs the stage body. The role TEST refuses ``allow_custom``, so this is how a toy TEST run is made; ``start()`` has no
    such path (review r1)."""
    monkeypatch.setattr(run_evaluate, 'LOADER', TOY_LOADER)
    monkeypatch.setattr(run_evaluate, 'BACKEND', SYNTHETIC)


def test_the_stage_body_is_an_importable_function_of_a_run_folder_and_twelve_texts():
    """The caller supplies no hash of the S5 lock any more (decision D28: the profile is the anchor): the stage has the profile file and its lock, the manifest, the loader and the
    backend, the device, the threads and the batch size, the access folder, the lock file, the rehearsal run and the test-only switch."""
    import importlib
    import inspect
    parameters = list(inspect.signature(run_evaluate.evaluate_stage).parameters)
    assert parameters == ['run_dir', 'profile_path', 'profile_sha256', 'manifest_path', 'loader', 'backend', 'device', 'threads', 'batch_size', 'access_dir', 'lock_path',
                          'rehearsal_run', 'allow_custom']
    module, _, name = run_evaluate.BODY.partition(':')
    assert getattr(importlib.import_module(module), name) is run_evaluate.evaluate_stage and run_evaluate.EVALUATE_MODULES == ('c_ept.run.supervisor', 'c_ept.run.evaluate')


def test_a_stage_body_checks_the_lock_of_the_profile_and_its_text_arguments_before_it_does_anything(factory, world, tmp_path):
    profile = eval_profile(factory, tmp_path, role='STOP', numbers=(1,))
    manifest_path = write_eval_manifest(tmp_path, eval_manifest_document(world, factory.scales_for(S4B_FIELDS)))
    run_dir = tmp_path / 'run'
    good = (str(tmp_path / 'evaluate.json'), profile.sha256, str(manifest_path), TOY_LOADER, SYNTHETIC, 'cpu', '1', '32',
            '-', '-', '-', '-')
    for number, change in enumerate(({1: 'f' * 64}, {1: 'nothex'}, {5: 'tpu'}, {6: 'x'}, {6: '0'}, {7: '-3'}, {7: ''}, {3: 'not a loader'}, {3: 'c_ept.run.evaluate:missing'},
                                     {4: 'not a backend'}, {4: 'c_ept.run.evaluate:missing'}, {9: ''}, {11: 'maybe'})):
        arguments = tuple(change.get(i, text) for i, text in enumerate(good))
        with pytest.raises(run_io.IORefused):
            run_evaluate.evaluate_stage(run_dir, *arguments)
    assert not run_dir.exists()
    other = tmp_path / 'train.json'
    other.write_bytes(b'{"kind": "train"}\n')
    with pytest.raises(run_io.IORefused, match='evaluate profile'):
        run_evaluate.evaluate_stage(run_dir, str(other), run_profile.load(other).sha256, *good[2:])
    assert not run_dir.exists()

class EvalRig:
    """One supervised evaluation run's world: a profile file (role aside, the same for a DEV rehearsal and the TEST run), the evaluation inputs manifest, a decision record
    that agrees with the code constants, the injected gates of the supervisor (identity, E6, Git: the working tree of a test run is not a clean checkout) and a large budget."""

    def __init__(self, tmp_path, factory, world, *, role, name, numbers=(1, 2, 3), identity=IDENTITY_A, over=None, toy=False, **fields):
        """``toy``: the toy loader and backend stand for the defaults (the test uses the ``toy_defaults`` fixture; the child is given the body that does the same), so the run needs no
        ``allow_custom`` and a TEST run can be made."""
        from c_ept.tests.test_run_ledger import real_spec
        import io
        self.toy = toy
        self.tmp = tmp_path / name
        self.tmp.mkdir()
        folders = {f'seed{n}': str(seed_run(factory, n, **(over or {}))) for n in numbers}
        self.profile_path = self.tmp / 'evaluate.json'
        self.profile_path.write_bytes((json.dumps({'kind': 'evaluate', 'role': role, 'models': list(folders), 'checkpoints': folders, **fields}, indent=2) + '\n').encode('utf-8'))
        self.manifest = write_eval_manifest(self.tmp, eval_manifest_document(world, factory.scales_for(S4B_FIELDS)))
        dropped = list(run_evaluate.load_trained('seed1', seed_run(factory, 1)).stats.dropped)
        run_io.write_json_atomic(self.tmp / 'B1_DROPPED.json', {'dropped': dropped})
        self.spec, self.run_dir, self.identity, self.stream, self.log = real_spec(self.tmp), self.tmp / 'run', identity, io.StringIO(), []

    def e6(self):
        self.log.append('e6')
        return dict(E6_RECORD)

    def capture(self, modules, *, repo=None):
        self.log.append(('capture', tuple(modules)))
        return self.identity

    def start(self, **over):
        from c_ept.run import guards
        values = dict(manifest_path=self.manifest, spec_path=self.spec, budget=guards.Budget(10 ** 7, 2 ** 40, measure=lambda: 10 ** 6), constants=CPU, scrub_keys=[KEY],
                      stream=self.stream, loader=TOY_LOADER, backend=SYNTHETIC, allow_custom=True, e6=self.e6, capture=self.capture, verify=lambda saved, repo=None: None,
                      start_gate=lambda repo, identity, **kwargs: None, poll_s=0.05, repo=REPO)
        if self.toy:
            values.update(allow_custom=False, bodies={run_evaluate.STAGE: TOY_BODY}, allow_custom_body=True)             # m4: the library takes another body for DEV and TEST only with the fixture switch
        values.update(over)
        return run_evaluate.start(self.profile_path, self.run_dir, **values)

    def ledger(self):
        return run_ledger.Ledger(self.run_dir)


def sealed(tmp_path):
    """``(access ledger, lock file, hash of the lock)``: a new access ledger and the S5 lock that names its folder and its created entry. The hash goes into the profiles."""
    access = run_evaluate.AccessLedger.create(tmp_path / 'access')
    lock = tmp_path / 'S5_LOCK.json'
    return access, lock, write_lock(lock, access)


def test_a_dev_rehearsal_and_then_the_test_run_go_through_the_supervisor_and_test_opens_once(factory, world, tmp_path, toy_defaults):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', toy=True, holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    ledger = dev.ledger()
    assert ledger.stages == (('evaluate', 'finished'),) and sorted(ledger.stage('evaluate').outputs) == sorted([run_evaluate.RECORD_FILE, run_evaluate.REPORT_FILE])
    profile = run_profile.load(dev.profile_path)
    evaluation = ledger.header.runner['evaluation']
    assert evaluation['role'] == 'DEV' and evaluation['profile_sha256'] == profile.sha256 and ledger.header.profile_sha256 == profile.sha256
    assert evaluation['profile_without_role_sha256'] == run_evaluate.profile_without_role_sha256(profile) and set(evaluation['models']) == {'seed1', 'seed2', 'seed3'}
    assert ledger.header.runner['device'] == 'cpu' and ('capture', run_evaluate.EVALUATE_MODULES) in dev.log
    record = run_io.read_json(dev.run_dir / run_evaluate.RECORD_FILE)
    assert record['access']['event'] == 'dev_access' and record['role'] == 'DEV' and record['access']['seq'] == 1
    report = run_io.read_json(dev.run_dir / run_evaluate.REPORT_FILE)
    assert report['c4']['name'] == 'C4' and report['c4']['bootstrap_B'] == 2000 and report['c4']['delta'] < 0
    assert record['sources'] == {'loader': TOY_LOADER, 'backend': SYNTHETIC, 'manifest_sha256': run_io.sha_file(dev.manifest), 'allow_custom': False,
                                 'body': run_evaluate.body_record(TOY_BODY)}
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', toy=True, holdout_lock_sha256=lock_sha)
    assert test.start(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir) == 0, test.stream.getvalue()
    assert [e['event'] for e in access.entries()] == ['created', 'dev_access', 'test_opened']
    assert access.entries()[2]['record']['lock_sha256'] == lock_sha and run_io.read_json(test.run_dir / run_evaluate.RECORD_FILE)['sources']['allow_custom'] is False
    inputs = test.ledger().stage('evaluate').inputs
    assert lock.as_posix() in inputs and (dev.run_dir / run_evaluate.RECORD_FILE).as_posix() in inputs                  # the evidence of the gate is bound by hash
    again = EvalRig(tmp_path, factory, world, role='TEST', name='again', toy=True, holdout_lock_sha256=lock_sha)
    assert again.start(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir) == 2
    assert not again.run_dir.exists() and 'opened' in again.stream.getvalue()                                         # a refusal before the reservation: TEST is never reopened
    assert [e['event'] for e in access.entries()].count('test_opened') == 1


def test_the_stage_of_an_evaluation_binds_the_files_of_the_checkpoints_and_of_the_inputs(factory, world, tmp_path):
    access = run_evaluate.AccessLedger.create(tmp_path / 'access')
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev')
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    inputs = dev.ledger().stage('evaluate').inputs
    expected = {dev.profile_path.as_posix(), dev.manifest.as_posix()}
    for name in ('seed1', 'seed2', 'seed3'):
        expected |= set(run_evaluate.load_trained(name, seed_run(factory, int(name[-1]))).bound_files())
    assert expected <= set(inputs) and set(runners.manifest_files(run_io.read_json(dev.manifest))) <= set(inputs)
    assert all(inputs[path] == run_io.sha_file(path) for path in expected)


def test_a_dev_evaluation_without_an_access_ledger_folder_is_exit_2_before_anything_is_reserved(factory, world, tmp_path):
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev')
    assert dev.start() == 2 and not dev.run_dir.exists() and 'access ledger' in dev.stream.getvalue()
    missing = tmp_path / 'nowhere'
    dev.stream.truncate(0)
    assert dev.start(access_dir=missing) == 2 and not dev.run_dir.exists() and not missing.exists() and 'access ledger' in dev.stream.getvalue()


def test_a_test_evaluation_with_a_wrong_lock_a_missing_rehearsal_or_no_gate_is_exit_2_before_anything_is_reserved(factory, world, tmp_path, toy_defaults):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', toy=True, holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', toy=True, holdout_lock_sha256=lock_sha)
    good = dict(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir)
    for change, word in (({'lock_path': tmp_path / 'absent.json'}, 'lock'), ({'rehearsal_run': tmp_path / 'nowhere'}, 'rehearsal'), ({'lock_path': None}, 'gate'),
                         ({'rehearsal_run': None}, 'gate'), ({'access_dir': None}, 'access ledger'), ({'access_dir': tmp_path / 'another_folder'}, 'access ledger')):
        test.stream.truncate(0)
        assert test.start(**{**good, **change}) == 2 and not test.run_dir.exists() and word in test.stream.getvalue(), (change, test.stream.getvalue())
    other = EvalRig(tmp_path, factory, world, role='TEST', name='other_lock', toy=True, holdout_lock_sha256='0' * 64)                 # the profile fixes another lock than the file
    assert other.start(**good) == 2 and 'lock' in other.stream.getvalue() and not other.run_dir.exists()
    assert access.test_opened() is False


def test_a_test_run_of_other_code_than_the_rehearsal_is_refused_in_the_child_and_does_not_open_test(factory, world, tmp_path, toy_defaults):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', toy=True, holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', toy=True, identity=IDENTITY_B, holdout_lock_sha256=lock_sha)
    assert test.start(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir) == 2
    stops = [dict(note.record) for note in test.ledger().notes if note.kind == 'stop']
    assert len(stops) == 1 and stops[0]['reason'] == 'refused' and stops[0]['exit_code'] == 2 and 'identity' in stops[0]['detail']
    assert test.ledger().stages == (('evaluate', 'begun'),) and access.test_opened() is False


def test_models_that_are_mixed_are_exit_2_before_anything_is_reserved(factory, world, tmp_path):
    plain = factory.make('plain_seed2', seed=SEEDS[1])                                                    # the plain law and the raw gold, another seed value
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='mixed', numbers=(1,))
    document = run_io.read_json(rig.profile_path)
    document['models'], document['checkpoints'] = ['seed1', 'plain'], {**document['checkpoints'], 'plain': str(plain)}
    rig.profile_path.write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    assert rig.start() == 2 and not rig.run_dir.exists() and 'kill_law' in rig.stream.getvalue()


def test_a_profile_that_is_not_an_evaluate_profile_or_a_manifest_that_is_not_acceptable_is_exit_2_and_its_text_holds_no_key(factory, world, tmp_path):
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='bad', numbers=(1,))
    train = rig.tmp / 'train.json'
    train.write_bytes(b'{"kind": "train"}\n')
    rig.profile_path = train
    assert rig.start() == 2 and 'evaluate profile' in rig.stream.getvalue() and not rig.run_dir.exists()
    rig.profile_path = rig.tmp / 'evaluate.json'
    rig.stream.truncate(0)
    assert rig.start(manifest_path=rig.tmp / f'absent_{KEY}.json') == 2 and KEY not in rig.stream.getvalue() and not rig.run_dir.exists()


def test_a_stop_evaluation_runs_through_the_supervisor_without_an_access_ledger(factory, world, tmp_path):
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='stop')
    assert rig.start() == 0, rig.stream.getvalue()
    record = run_io.read_json(rig.run_dir / run_evaluate.RECORD_FILE)
    assert record['access'] is None and record['role'] == 'STOP' and rig.ledger().stages == (('evaluate', 'finished'),)
    assert run_io.read_json(rig.run_dir / run_evaluate.REPORT_FILE)['models']['seed1']['stop']['loss_stop'] == 1.5

def test_runners_evaluate_is_the_entry_point_that_the_plan_names_and_delegates_to_this_module(monkeypatch):
    """The plan's interface is ``runners.evaluate(profile, run_dir, *, inputs=...)``; the work is in this module (``runners`` is the training runner of K6a)."""
    import inspect
    parameters = inspect.signature(runners.evaluate).parameters
    assert list(parameters)[:2] == ['profile', 'run_dir'] and parameters['inputs'].kind is inspect.Parameter.KEYWORD_ONLY and parameters['inputs'].default is inspect.Parameter.empty
    seen = []
    monkeypatch.setattr(run_evaluate, 'evaluate', lambda *args, **kwargs: seen.append((args, kwargs)) or {'record': {}, 'report': {}})
    assert runners.evaluate('profile', 'run', inputs='inputs', constants='c', backend='b') == {'record': {}, 'report': {}}
    assert seen == [(('profile', 'run'), {'inputs': 'inputs', 'constants': 'c', 'backend': 'b'})]

def test_the_d9_inputs_are_the_two_shared_m2_blocks_and_the_masked_m3_blocks_separately(factory, tmp_path):
    """Decision D9 judges the zone mark by the shared M2 NLL (M2Rate and M2Mark, not added), the V_start CRPS and the masked M3 NLL: the STOP report names them apart."""
    class WithM3(FakeBackend):
        def stop_report(self, model, trained, inputs, constants, check):
            blocks = {key: {'mean': 1.0, 'scale': 2.0, 'term': 0.5} for key in ('M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked', 'M3Size/masked', 'M3Kills/masked', 'M3Size/open')}
            return {'loss_stop': 3.0, 'blocks': blocks}
    result = run_evaluate.evaluate(eval_profile(factory, tmp_path), tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=WithM3())
    stop = result['report']['models']['seed1']['stop']
    assert set(stop['shared_m2']) == {'M2Rate/masked', 'M2Mark/masked'} and set(stop['masked_m3']) == {'M3Size/masked', 'M3Kills/masked'}
    assert not [u for u in result['record']['unavailable'] if u['item'] in ('shared_m2', 'masked_m3')]
    bare = run_evaluate.evaluate(eval_profile(factory, tmp_path), tmp_path / 'run2', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())
    reasons = [u for u in bare['record']['unavailable'] if u['item'] == 'masked_m3']
    assert bare['report']['models']['seed1']['stop']['masked_m3'] == {} and len(reasons) == 1 and reasons[0]['model'] == 'seed1'


# ======================================================================================================================================================
# K6b fix round 1 (review of 18d4406): I1, I2 (with m4), I4, I5, I6, m1, m2, m5, m6, m7. C1 (the anchor of the gate) and I3 (K6d) are not here.
# ======================================================================================================================================================
def with_identity(trained, **changes):
    """A copy of a ``TrainedRun`` whose training identity (the final head's) has ``changes`` (what a run of another environment or configuration would carry)."""
    head = copy.deepcopy(dict(trained.head))
    head['identity'] = {**head['identity'], **changes}
    return dataclasses.replace(trained, head=head)


# ---- I1: the role TEST uses up its one opening only when the backend can evaluate it ------------------------------------------------------------------
def test_the_default_backend_declares_dev_and_stop_and_not_test():
    backend = run_evaluate.Backend()
    assert [backend.evaluates(role) for role in ('DEV', 'STOP', 'TEST', 'FIT', 'CAL')] == [True, True, False, False, False]


def test_test_is_refused_before_the_opening_when_the_backend_cannot_evaluate_it(factory, tmp_path):
    """Review I1: the default backend answers nothing for TEST (until a TEST reader of the stores exists, S8), so a TEST run on it would only burn D20's single opening: it is refused first, with no marker, no entry."""
    fields = {'models': ['seed1', 'seed2', 'seed3'], 'checkpoints': {f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)}}
    world = GateWorld(tmp_path, **fields)
    world.rehearse(sources=fake_sources(), models=real_models(factory))
    before = world.access.entries()
    with pytest.raises(run_evaluate.EvaluationRefused, match='cannot evaluate'):
        run_evaluate.evaluate(world.test, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=run_evaluate.Backend(), gate=world.gate(),
                              identity=IDENTITY_A, clock=world.clock, **FAKE)
    assert world.access.test_opened() is False and world.access.entries() == before and not world.access.marker.exists() and not (tmp_path / 'run').exists()


def test_a_supervised_test_run_on_the_default_backend_is_exit_2_before_anything_is_reserved(factory, world, tmp_path):
    access, lock, lock_sha = sealed(tmp_path)
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', holdout_lock_sha256=lock_sha)
    code = test.start(access_dir=access.folder, lock_path=lock, rehearsal_run=tmp_path / 'nowhere', loader=run_evaluate.LOADER, backend=run_evaluate.BACKEND, allow_custom=False)
    assert code == 2 and 'cannot evaluate' in test.stream.getvalue() and not test.run_dir.exists() and access.test_opened() is False


# ---- I6: profile fields that nothing in the runner reads are refused, not ignored -----------------------------------------------------------------------
@pytest.mark.parametrize('field, given, task', [('strata', True, 'M17'), ('price_sensitivity', True, 'M17'), ('b1_swap_average', True, 'M17')])
def test_a_profile_field_that_the_runner_does_not_read_yet_is_refused_and_names_the_task(factory, tmp_path, field, given, task):
    backend = FakeBackend()
    profile = eval_profile(factory, tmp_path, **{field: given})
    with pytest.raises(run_evaluate.EvaluationRefused, match=f'{field}.*{task}'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=backend)
    assert backend.log == [] and not (tmp_path / 'run').exists()
    default = eval_profile(factory, tmp_path, name='default.json')                                    # the defaults are not refused
    run_evaluate.evaluate(default, tmp_path / 'run2', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())


def test_a_supervised_evaluation_of_a_profile_with_an_unread_field_is_exit_2_before_anything_is_reserved(factory, world, tmp_path):
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='unread', numbers=(1,), b1_swap_average=True)
    assert rig.start() == 2 and 'b1_swap_average' in rig.stream.getvalue() and 'M17' in rig.stream.getvalue() and not rig.run_dir.exists()


# ---- m5, m6, m7: the bootstrap rows, the adjusted p and the inputs of the C4 row ----------------------------------------------------------------------
def test_a_bootstrap_of_more_rows_than_the_common_matrix_is_refused_and_a_c4_with_other_than_2000_rows_is_flagged(factory, tmp_path):
    backend = FakeBackend()
    too_many = eval_profile(factory, tmp_path, bootstrap_B=10_001)
    with pytest.raises(run_evaluate.EvaluationRefused, match='10,000'):
        run_evaluate.evaluate(too_many, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=backend)
    assert backend.log == [] and not (tmp_path / 'run').exists()
    full = eval_profile(factory, tmp_path, numbers=(1, 2, 3), bootstrap_B=10_000, name='full.json')                # the whole matrix is allowed, and it is flagged: D13 fixes B = 2,000 for C4
    result = run_evaluate.evaluate(full, tmp_path / 'run2', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend())
    assert result['report']['c4']['bootstrap_B'] == 10_000 and '2,000' in result['report']['c4']['bootstrap_flag']


def test_the_c4_row_has_no_family_adjusted_p_and_names_the_fit_rows_and_the_b1_files_it_rests_on(factory, tmp_path):
    files = {'b1_heads.json': tmp_path / 'b1_heads.json', 'sub/head.json': tmp_path / 'head.json'}
    files['b1_heads.json'].write_bytes(b'{"b1": 1}\n')
    files['sub/head.json'].write_bytes(b'{"head": 1}\n')
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3))
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory), b1_files=files), constants=CPU, backend=FakeBackend())
    row, record = result['report']['c4'], result['record']
    assert row['adjusted_p'] is None and 'Holm' in row['adjusted_p_note'] and row['bootstrap_flag'] is None and row['p'] > 0
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    fit = {key: val for key, val in trained.identity['fit'].items() if key != 'seed'}
    assert row['fit_sha256'] == run_io.canonical_sha256(fit) and record['agreed']['fit_sha256'] == row['fit_sha256']                 # the membership, the same for the three seeds
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    assert record['b1'] == {'files_sha256': hashes, 'folder_sha256': run_io.canonical_sha256(hashes)} and row['b1_folder_sha256'] == record['b1']['folder_sha256']
    bare = run_evaluate.evaluate(profile, tmp_path / 'run2', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend())
    assert bare['record']['b1'] == {'files_sha256': {}, 'folder_sha256': run_io.canonical_sha256({})}               # a B1 without files: its (empty) record
    none = run_evaluate.evaluate(profile, tmp_path / 'run3', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())
    assert none['record']['b1'] is None and none['report']['c4'] is None                                          # no B1, no record of it (review r2)


# ---- I5: one configuration means one training identity, the seed and the execution keys aside --------------------------------------------------------
@pytest.mark.parametrize('over, name', [({'lr': 0.002}, 'lr'), ({'weight_decay': 0.5}, 'weight_decay'), ({'warmup': 3}, 'warmup_steps'), ({'patience': 9}, 'patience'),
                                        ({'epochs': 1}, 'epochs'), ({'selection_seed': 20261030}, 'fit')])
def test_models_that_differ_in_a_training_key_other_than_the_seed_are_refused_and_the_key_is_named(factory, over, name):
    """Review I5: lr, weight decay, warmup, patience, epochs and the FIT membership are part of the identity; three runs of other learning rates are not three seeds."""
    first = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    other = run_evaluate.load_trained('other', seed_run(factory, 2, **over))
    with pytest.raises(run_evaluate.MixedSettings) as raised:
        run_evaluate.check_agreement([first, other])
    named = str(raised.value).split(':')[0]                                                            # 'the models differ in <keys>'
    assert name in named and 'seed' not in named
    with pytest.raises(run_evaluate.MixedSettings, match=name):
        run_evaluate.check_models([first, other], eval_inputs(factory))


@pytest.mark.parametrize('key, other', [('torch', '0.0.0'), ('dtype', 'torch.float64'), ('deterministic_algorithms', 'flipped'), ('lr', 0.5), ('s2_store_json_sha256', '0' * 64),
                                        ('static_sha256', '0' * 64), ('model_spec', {'architecture': 'ept_s_v1', 'other': 1})])
def test_the_environment_and_the_data_identity_of_the_models_must_be_equal(factory, key, other):
    trained = trained_set(factory)
    value_ = (not trained[1].identity[key]) if other == 'flipped' else other
    with pytest.raises(run_evaluate.MixedSettings, match=key):
        run_evaluate.check_agreement([trained[0], with_identity(trained[1], **{key: value_}), trained[2]])


def test_the_seed_and_the_execution_keys_may_differ_and_are_listed_in_the_agreement(factory):
    trained = trained_set(factory)
    moved = [with_identity(trained[0], device='cuda:0', threads=4), trained[1], trained[2]]
    checked = run_evaluate.check_models(moved, eval_inputs(factory))
    assert checked['execution'] == {'seed1': {'device': 'cuda:0', 'threads': 4},
                                    'seed2': {'device': trained[1].identity['device'], 'threads': trained[1].identity['threads']},
                                    'seed3': {'device': trained[2].identity['device'], 'threads': trained[2].identity['threads']}}
    assert checked['settings'] == dict(trained[0].settings)


def test_the_fit_rows_of_the_models_must_be_equal_apart_from_the_batch_seed(factory):
    """The digest of the FIT batches holds the shuffle seed, which is the training seed: the three seeds differ in it and in nothing else. Another row digest is another membership."""
    trained = trained_set(factory)
    assert trained[0].identity['fit']['seed'] != trained[1].identity['fit']['seed']
    run_evaluate.check_agreement(trained)                                                              # the real seeds agree
    other_rows = with_identity(trained[1], fit={**trained[1].identity['fit'], 'rows_sha256': '0' * 64})
    with pytest.raises(run_evaluate.MixedSettings, match='fit'):
        run_evaluate.check_agreement([trained[0], other_rows])
    other_stop = with_identity(trained[1], stop={**trained[1].identity['stop'], 'rows_sha256': '0' * 64})
    with pytest.raises(run_evaluate.MixedSettings, match='stop'):
        run_evaluate.check_agreement([trained[0], other_stop])


def test_the_s2_store_of_the_evaluation_must_be_the_one_of_the_training(factory, tmp_path):
    trained = trained_set(factory, (1,))
    inputs = eval_inputs(factory)
    checked = run_evaluate.check_models(trained, inputs)
    assert checked['s2_store_json_sha256'] == run_io.sha_file(Path(inputs.base.s2.file)) == trained[0].identity['s2_store_json_sha256']
    other = tmp_path / 'STORE.json'
    other.write_bytes(b'{"another": "store"}\n')
    wrong = run_evaluate.EvalInputs(base=dataclasses.replace(inputs.base, s2=SimpleNamespace(file=other)))
    with pytest.raises(run_evaluate.CheckpointRefused, match='S2 store'):
        run_evaluate.check_models(trained, wrong)


def test_the_dropped_columns_of_the_models_must_be_equal_also_without_a_matched_b1(factory):
    trained = trained_set(factory)
    odd = dataclasses.replace(trained[1], stats=SimpleNamespace(dropped=(*trained[1].stats.dropped, 'extra_column'), sha256=trained[1].stats.sha256))
    with pytest.raises(run_evaluate.CheckpointRefused, match='dropped'):
        run_evaluate.check_models([trained[0], odd, trained[2]], eval_inputs(factory))


# ---- I4: the nominal exposure makes the primary and the secondary pass (D14) ------------------------------------------------------------------------
def secondary_endpoint(trained, exposure, population, found):
    counts = {'candidate': 12, 'population_candidate': 12, 'population_selected': 7} if population == 'primary' else {'candidate': 12, 'population_candidate': 12, 'population_selected': 5}
    out = {**found, 'counts': counts}
    if population == 'secondary':
        out['p_ept'] = np.clip(found['p_b1'] + 0.05, 0.0, 1.0)                                         # another probability: the secondary pass must not feed C4
    return out


def test_the_nominal_exposure_makes_a_primary_and_a_secondary_pass_and_reports_both(factory, tmp_path):
    from c_ept.evaluation import inference, proper
    clock, access, profile = dev_setup(factory, tmp_path, exposure='nominal')
    backend = FakeBackend(endpoint=secondary_endpoint)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=backend, access=access, identity=IDENTITY_A,
                                   clock=clock, **FAKE)
    assert backend.passes == [(name, exposure, population) for name in ('seed1', 'seed2', 'seed3') for exposure, population in (('nominal', 'primary'), ('observed', 'secondary'))]
    assert result['record']['exposure'] == {'exposure': 'nominal', 'population': 'primary', 'passes': [{'name': 'primary', 'exposure': 'nominal', 'population': 'primary'},
                                                                                                    {'name': 'secondary', 'exposure': 'observed', 'population': 'secondary'}]}
    found = c4_endpoints()
    weight = np.full(12, 1.0 / 3.0)
    for seed, name in zip(SEEDS, ('seed1', 'seed2', 'seed3')):
        passes = result['report']['models'][name]['passes']
        assert list(passes) == ['primary', 'secondary']
        assert passes['primary']['counts'] == {'candidate': 12, 'population_candidate': 12, 'population_selected': 7}
        assert passes['secondary']['counts'] == {'candidate': 12, 'population_candidate': 12, 'population_selected': 5}
        assert (passes['secondary']['exposure'], passes['secondary']['population']) == ('observed', 'secondary')
        secondary = secondary_endpoint(None, 'observed', 'secondary', found[seed])
        ept, b1 = proper.brier(secondary['p_ept'], found[seed]['y'], weight), proper.brier(found[seed]['p_b1'], found[seed]['y'], weight)
        assert passes['secondary']['brier'] == pytest.approx({'ept': ept, 'b1': b1, 'diff': ept - b1}, rel=1e-12)
        assert result['report']['models'][name]['inference_provenance'] == {'kill_law': 'shifted', 'exposure': 'nominal', 'population': 'primary'}     # the primary pass is the main one
    expected = run_evaluate.c4_statistic({seed: secondary_endpoint(None, 'nominal', 'primary', found[seed]) for seed in SEEDS}, IDS, inference.resample_matrix(len(IDS))[:2000])
    assert result['report']['c4'] == {**json.loads(json.dumps(expected)), 'fit_sha256': result['report']['c4']['fit_sha256'], 'b1_folder_sha256': result['report']['c4']['b1_folder_sha256']}
    assert not [u for u in result['record']['unavailable'] if u['item'] == 'inference']


def test_a_pass_that_is_unavailable_is_reported_with_its_name_and_the_observed_exposure_makes_one_pass(factory, tmp_path):
    profile = eval_profile(factory, tmp_path, numbers=(1,), exposure='nominal')
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU)                      # the default backend on STOP without a B1, prices and tier: both passes say so
    items = [u for u in result['record']['unavailable'] if u['item'] == 'inference']
    assert [u['pass'] for u in items] == ['primary', 'secondary'] and all('no matched B1' in u['reason'] for u in items)
    legacy = eval_profile(factory, tmp_path, numbers=(1,), name='legacy.json')
    backend = FakeBackend()
    run_evaluate.evaluate(legacy, tmp_path / 'run2', inputs=eval_inputs(factory), constants=CPU, backend=backend)
    assert backend.passes == [('seed1', 'observed', 'all')]


# ---- I2: the loader, the backend and the inputs of a DEV or TEST evaluation, and what the gate compares ---------------------------------------------
def test_the_record_and_the_dev_access_line_name_the_loader_the_backend_and_the_inputs_manifest(factory, tmp_path):
    clock, access, profile = dev_setup(factory, tmp_path)
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(), access=access, identity=IDENTITY_A,
                                   clock=clock, **FAKE)
    assert result['record']['sources'] == fake_sources() and access.entries()[1]['record']['sources'] == fake_sources()
    default = run_evaluate.evaluate(profile, tmp_path / 'run2', inputs=eval_inputs(factory), constants=CPU, backend=run_evaluate.Backend(), access=access, identity=IDENTITY_A,
                                    clock=clock, loader=run_evaluate.LOADER, manifest_sha256='c' * 64)                              # the defaults need no switch
    assert default['record']['sources'] == {'loader': run_evaluate.LOADER, 'backend': run_evaluate.BACKEND, 'manifest_sha256': 'c' * 64, 'allow_custom': False}
    stop = run_evaluate.evaluate(eval_profile(factory, tmp_path, name='stop.json'), tmp_path / 'run3', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())
    assert stop['record']['sources']['allow_custom'] is False and stop['record']['sources']['loader'] is None                       # STOP needs no switch for a made-up backend


def test_a_loader_or_backend_other_than_the_defaults_is_refused_for_dev_and_test_unless_the_test_switch_is_set(factory, tmp_path):
    clock, access, profile = dev_setup(factory, tmp_path)
    inputs = eval_inputs(factory, b1=b1_of(factory))
    for loader, backend in ((FAKE['loader'], FakeBackend()), (None, run_evaluate.Backend()), (FAKE['loader'], run_evaluate.Backend())):
        with pytest.raises(run_evaluate.EvaluationRefused, match='allow_custom'):
            run_evaluate.evaluate(profile, tmp_path / 'run', inputs=inputs, constants=CPU, backend=backend, access=access, identity=IDENTITY_A, clock=clock, loader=loader,
                                  manifest_sha256='c' * 64)
    assert [e['event'] for e in access.entries()] == ['created'] and not (tmp_path / 'run').exists()                                # nothing was logged, nothing read
    folder = tmp_path / 'gate_world'
    folder.mkdir()
    world = GateWorld(folder, models=['seed1', 'seed2', 'seed3'], checkpoints={f'seed{n}': str(seed_run(factory, n)) for n in (1, 2, 3)})
    world.rehearse(sources=fake_sources(allow_custom=False), models=real_models(factory))
    with pytest.raises(run_evaluate.EvaluationRefused, match='allow_custom'):
        run_evaluate.evaluate(world.test, tmp_path / 'run2', inputs=inputs, constants=CPU, backend=FakeBackend(), gate=world.gate(), identity=IDENTITY_A, clock=world.clock,
                              loader=FAKE['loader'], manifest_sha256=FAKE['manifest_sha256'])
    assert world.access.test_opened() is False


def test_a_supervised_dev_evaluation_with_a_made_up_loader_is_exit_2_unless_the_switch_is_set_and_test_never_takes_the_switch(factory, world, tmp_path):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder, allow_custom=False) == 2 and 'allow_custom' in dev.stream.getvalue() and not dev.run_dir.exists()
    assert [e['event'] for e in access.entries()] == ['created']
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()                                                          # the switch of the rig: test fixtures only
    record = run_io.read_json(dev.run_dir / run_evaluate.RECORD_FILE)
    assert record['sources'] == {'loader': TOY_LOADER, 'backend': SYNTHETIC, 'manifest_sha256': run_io.sha_file(dev.manifest), 'allow_custom': True,
                                 'body': run_evaluate.body_record(run_evaluate.BODY)}                                                # the default stage body: the real one
    assert dev.ledger().header.runner['evaluation']['sources'] == record['sources']
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', holdout_lock_sha256=lock_sha)
    for switch in (False, True):                                                                                                     # a made-up loader, with and without the switch
        test.stream.truncate(0)
        assert test.start(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir, allow_custom=switch) == 2 and 'allow_custom' in test.stream.getvalue()
        assert not test.run_dir.exists()
    assert access.test_opened() is False


def test_a_test_run_that_reads_other_inputs_than_the_rehearsal_is_refused(factory, world, tmp_path, toy_defaults):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', toy=True, holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', toy=True, holdout_lock_sha256=lock_sha)
    document = run_io.read_json(test.manifest)
    document['scales_provenance'] = {'another': 'manifest'}                                                                          # the TEST run reads other inputs than the rehearsal did
    test.manifest.write_bytes(json.dumps(document, indent=1).encode('utf-8'))
    assert test.start(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir) == 2 and 'manifest_sha256' in test.stream.getvalue()
    assert not test.run_dir.exists() and access.test_opened() is False


@pytest.mark.parametrize('field, other', [('loader', 'c_ept.tests.other:loader'), ('backend', 'c_ept.tests.other:Backend'), ('manifest_sha256', 'c' * 64), ('allow_custom', True)])
def test_the_test_run_must_have_the_loader_the_backend_and_the_inputs_of_the_rehearsal(tmp_path, field, other):
    world = GateWorld(tmp_path)
    world.rehearse()
    assert world.check()['rehearsal']['sources'] == SOURCES
    with pytest.raises(run_evaluate.AccessRefused, match=field):
        world.check(sources={**SOURCES, field: other})
    with pytest.raises(run_evaluate.AccessRefused, match=field):
        world.open(sources={**SOURCES, field: other})
    assert world.access.test_opened() is False


def test_the_checkpoints_of_the_test_run_must_be_the_training_runs_of_the_rehearsal(tmp_path):
    """Review m4 (folded into I2): the same folders could hold re-made training runs; each model's ``train_record_sha256`` must be the rehearsal's."""
    world = GateWorld(tmp_path)
    world.rehearse()
    assert world.check()['rehearsal']['models'] == world.models
    with pytest.raises(run_evaluate.AccessRefused, match='seed2'):
        world.check(models={**world.models, 'seed2': 'f' * 64})
    with pytest.raises(run_evaluate.AccessRefused, match='models'):
        world.check(models={name: digest for name, digest in world.models.items() if name != 'seed3'})
    with pytest.raises(run_evaluate.AccessRefused, match='models'):
        world.check(models={**world.models, 'seed4': 'e' * 64})
    with pytest.raises(run_evaluate.AccessRefused, match='seed2'):
        world.open(models={**world.models, 'seed2': 'f' * 64})
    assert world.access.test_opened() is False


def test_the_rehearsal_must_be_the_logged_entry_itself_and_not_any_line_with_the_same_two_hashes(tmp_path):
    """Review I2: the record of the rehearsal names its own ``dev_access`` entry (seq and hash); another line for the same profile and code does not stand for it."""
    world = GateWorld(tmp_path)
    world.rehearse(log=False)
    profile_hash = run_evaluate.profile_without_role_sha256(world.dev)
    world.access.append('dev_access', {'role': 'DEV', 'profile_without_role_sha256': profile_hash, 'identity_sha256': run_io.canonical_sha256(IDENTITY_A.to_record()),
                                       'sources': SOURCES}, clock=world.clock)
    with pytest.raises(run_evaluate.AccessRefused, match='logged'):
        world.check()
    for number, override in enumerate(({'event': 'dev_access', 'seq': 1, 'entry_sha256': '0' * 64}, {'event': 'dev_access', 'seq': 99, 'entry_sha256': '0' * 64},
                                       {'event': 'dev_access', 'seq': 'one', 'entry_sha256': '0' * 64}, {'event': 'dev_access'})):
        folder = tmp_path / f'w{number}'
        folder.mkdir()
        other = GateWorld(folder)
        other.rehearse(access_override=override)
        with pytest.raises(run_evaluate.AccessRefused, match='logged'):
            other.check()


def test_the_rehearsal_of_another_ledger_is_not_found_in_the_gates_ledger(tmp_path):
    first = GateWorld(tmp_path)
    first.rehearse()
    second_dir = tmp_path / 'second'
    second_dir.mkdir()
    second = GateWorld(second_dir, **first.fields)
    second.clock = FakeClock(7000.0)                                                                                          # its entries are made at other times than the first ledger's
    second.rehearse(name='other_rehearsal')
    with pytest.raises(run_evaluate.AccessRefused, match='logged'):
        first.check(gate=first.gate(rehearsal_run=second.tmp / 'other_rehearsal'))                                             # its entry hash is not an entry of this ledger


def test_the_test_opening_names_the_sources_and_the_models(tmp_path):
    world = GateWorld(tmp_path)
    world.rehearse()
    record = world.open()['record']
    assert record['sources'] == SOURCES and record['models'] == world.models


# ---- m1: the fail-closed branches that no test inverted -----------------------------------------------------------------------------------------------
def patch_latest(monkeypatch, change):
    real = ept_checkpoints.latest_metadata

    def patched(out_dir):
        found = real(out_dir)
        return change(found[0], copy.deepcopy(found[1]), found[2])
    monkeypatch.setattr(ept_checkpoints, 'latest_metadata', patched)


def test_a_head_that_is_not_done_a_model_that_is_not_an_ept_and_an_identity_that_differs_from_the_record_are_refused(factory, monkeypatch):
    folder = seed_run(factory, 1)

    def not_done(final, head, extra):
        head['done'] = False
        return final, head, extra

    def other_architecture(final, head, extra):
        head['identity']['model_spec'] = {'architecture': 'other_net'}
        return final, head, extra

    def other_law(final, head, extra):
        head['identity']['kill_law'] = 'poisson'                                                                                 # the record says shifted
        return final, head, extra

    def other_final(final, head, extra):
        return final.parent / 'epoch_99', head, extra

    for change, word in ((not_done, 'not done'), (other_architecture, 'not an EPT'), (other_law, 'explicit identity'), (other_final, 'newest checkpoint')):
        patch_latest(monkeypatch, change)
        with pytest.raises(run_evaluate.CheckpointRefused, match=word):
            run_evaluate.load_trained('seed1', folder)
        monkeypatch.undo()
    monkeypatch.setattr(ept_checkpoints, 'latest_metadata', lambda out_dir: None)
    with pytest.raises(run_evaluate.CheckpointRefused, match='no readable committed checkpoint'):
        run_evaluate.load_trained('seed1', folder)


def test_statistics_other_than_the_ones_of_the_record_are_refused(factory, monkeypatch):
    monkeypatch.setattr(data.Stats, 'load', classmethod(lambda cls, path: SimpleNamespace(sha256='0' * 64)))
    with pytest.raises(run_evaluate.CheckpointRefused, match='statistics other than'):
        run_evaluate.load_trained('seed1', seed_run(factory, 1))


def test_a_ledger_whose_file_is_gone_but_whose_marker_stays_is_not_created_again_and_the_gate_and_the_identity_check_refuse_what_is_not_theirs(tmp_path):
    ledger = run_evaluate.AccessLedger.create(tmp_path / 'access')
    ledger.append('test_opened', {'run': 'first'})
    ledger.path.unlink()                                                                                                         # the ledger file is deleted, the marker remains
    with pytest.raises(run_evaluate.AccessRefused, match='already'):
        run_evaluate.AccessLedger.create(tmp_path / 'access')
    with pytest.raises(run_evaluate.AccessRefused, match='AccessLedger'):
        run_evaluate.HoldoutGate(tmp_path / 'lock.json', tmp_path / 'run', 'not a ledger')
    with pytest.raises(run_evaluate.AccessRefused, match='Identity'):
        run_evaluate.HoldoutGate.check_identity({'rehearsal': {'identity_sha256': '0' * 64}}, object())


def test_a_match_list_that_is_not_sorted_unique_and_non_empty_is_refused(factory, tmp_path):
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3))
    for ids in (np.array([12, 10, 11, 13], dtype=np.int64), np.array([], dtype=np.int64), np.array([10, 10, 11], dtype=np.int64)):
        with pytest.raises(run_evaluate.EvaluationRefused, match='sorted'):
            run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(ids=ids))


def test_endpoints_that_do_not_make_a_c4_row_are_a_refusal_not_a_silent_gap(factory, tmp_path):
    def other_b1(trained, exposure, population, found):
        if trained.seed == SEEDS[2]:
            return {**found, 'p_b1': found['p_b1'] * 0.9}                                                                       # the third seed was scored against another B1
        return found
    profile = eval_profile(factory, tmp_path, numbers=(1, 2, 3))
    with pytest.raises(run_evaluate.EvaluationRefused, match='C4 cannot be made'):
        run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory, b1=b1_of(factory)), constants=CPU, backend=FakeBackend(endpoint=other_b1))


# ---- K6c: the numeric policy (D10 P6), the stage body that the gate compares (K6b review r4) and the approval text ---------------------------------------------------------
TWIN_BODY = 'c_ept.tests.run_evaluate_bodies:twin_stage_with_toy_defaults'
APPROVAL = 'Approved by the author: evaluate STOP, "no GPU".\nSecond line with a backslash \\ and 한글 승인 '


def test_the_stage_body_is_named_by_module_name_and_the_hash_of_its_file():
    default = run_evaluate.body_record(run_evaluate.BODY)
    assert default == {'module': 'c_ept.run.evaluate', 'name': 'evaluate_stage', 'file_sha256': run_io.sha_file(Path(run_evaluate.__file__))}
    toy = run_evaluate.body_record(TOY_BODY)
    assert (toy['module'], toy['name']) == ('c_ept.tests.run_evaluate_bodies', 'evaluate_stage_with_toy_defaults') and toy['file_sha256'] == run_io.sha_file(Path(bodies.__file__))
    assert run_evaluate.body_record(TWIN_BODY)['file_sha256'] == toy['file_sha256'] and run_evaluate.body_record(TWIN_BODY) != toy                 # the same file, another name
    assert run_evaluate.body_record('c_ept.tests.run_evaluate_bodies:SyntheticBackend.evaluates')['name'] == 'SyntheticBackend.evaluates'          # a dotted name stays whole
    for bad in ('c_ept.run.evaluate', 'no.such.module:body', 'c_ept.run.evaluate:', ':name', 'two:colons:here', 5, None):
        with pytest.raises(run_evaluate.EvaluationRefused):
            run_evaluate.body_record(bad)


def test_the_child_checks_that_the_file_of_its_body_is_the_one_that_the_supervisor_hashed():
    record = run_evaluate.body_record(TOY_BODY)
    run_evaluate.verify_body(record)
    for change in ({'file_sha256': '0' * 64}, {'module': 'c_ept.tests.not_there'}, {'name': 5}, {'extra': 1}):
        with pytest.raises(run_evaluate.EvaluationRefused, match='body'):
            run_evaluate.verify_body({**record, **change})
    for odd in (None, 'c_ept.run.evaluate:evaluate_stage', {}, {'module': 'm'}):
        with pytest.raises(run_evaluate.EvaluationRefused, match='body'):
            run_evaluate.verify_body(odd)


@pytest.mark.parametrize('field, other', [('module', 'c_ept.tests.other'), ('name', 'other_body'), ('file_sha256', 'f' * 64)])
def test_the_test_run_must_have_the_stage_body_of_the_rehearsal(tmp_path, field, other):
    world = GateWorld(tmp_path)
    world.rehearse()
    assert world.check()['rehearsal']['sources']['body'] == FAKE_BODY
    different = {**SOURCES, 'body': {**FAKE_BODY, field: other}}
    with pytest.raises(run_evaluate.AccessRefused, match='body'):
        world.check(sources=different)
    with pytest.raises(run_evaluate.AccessRefused, match='body'):
        world.open(sources=different)
    assert world.access.test_opened() is False


def test_the_test_gate_refuses_a_run_that_does_not_name_its_stage_body_even_when_the_rehearsal_names_none_either(tmp_path):
    without = {key: value_ for key, value_ in SOURCES.items() if key != 'body'}
    world = GateWorld(tmp_path)
    world.rehearse()
    with pytest.raises(run_evaluate.AccessRefused, match='body'):
        world.check(sources=without)                                                                   # the rehearsal names its body, this run names none
    for odd in ('c_ept.run.evaluate:evaluate_stage', {}, {'module': 'm'}, {**FAKE_BODY, 'file_sha256': 'short'}, {**FAKE_BODY, 'extra': 1}, {**FAKE_BODY, 'name': 5}):
        with pytest.raises(run_evaluate.AccessRefused, match='body'):
            world.check(sources={**SOURCES, 'body': odd})                                              # not a record of a stage body
    folder = tmp_path / 'bare'
    folder.mkdir()
    bare = GateWorld(folder)
    bare.rehearse(sources=without)                                                                     # a rehearsal that recorded no body (made before K6c)
    with pytest.raises(run_evaluate.AccessRefused, match='body'):
        bare.check(sources=without)                                                                    # equal by omission is not equal: the gate wants the body named
    with pytest.raises(run_evaluate.AccessRefused, match='body'):
        bare.check()
    assert world.access.test_opened() is False and bare.access.test_opened() is False


def test_a_supervised_evaluation_records_its_stage_body_in_the_record_the_access_line_and_the_header(factory, world, tmp_path, toy_defaults):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', toy=True, holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    toy = run_evaluate.body_record(TOY_BODY)
    assert run_io.read_json(dev.run_dir / run_evaluate.RECORD_FILE)['sources']['body'] == toy and access.entries()[1]['record']['sources']['body'] == toy
    assert dev.ledger().header.runner['evaluation']['sources']['body'] == toy
    stop = EvalRig(tmp_path, factory, world, role='STOP', name='stop')                                 # no bodies argument: the default stage body, the real one
    assert stop.start() == 0, stop.stream.getvalue()
    assert run_io.read_json(stop.run_dir / run_evaluate.RECORD_FILE)['sources']['body'] == run_evaluate.body_record(run_evaluate.BODY)


def test_a_test_run_with_another_stage_body_than_its_rehearsal_is_exit_2_before_the_reservation_and_opens_nothing(factory, world, tmp_path, toy_defaults):
    access, lock, lock_sha = sealed(tmp_path)
    dev = EvalRig(tmp_path, factory, world, role='DEV', name='dev', toy=True, holdout_lock_sha256=lock_sha)
    assert dev.start(access_dir=access.folder) == 0, dev.stream.getvalue()
    test = EvalRig(tmp_path, factory, world, role='TEST', name='test', toy=True, holdout_lock_sha256=lock_sha)
    given = dict(access_dir=access.folder, lock_path=lock, rehearsal_run=dev.run_dir)
    assert test.start(**given, bodies={run_evaluate.STAGE: TWIN_BODY}) == 2                              # the same code under another name is another stage body
    assert 'body' in test.stream.getvalue() and not test.run_dir.exists() and access.test_opened() is False
    assert test.start(**given) == 0, test.stream.getvalue()                                            # the body of the rehearsal opens it
    assert access.test_opened() is True and access.entries()[-1]['record']['sources']['body'] == run_evaluate.body_record(TOY_BODY)


def test_the_evaluation_stage_holds_the_numeric_policy_before_the_inputs_are_loaded(factory, world, tmp_path):
    """m3 of the K6c review: the fitted inputs (B1, B0, tier objects) are loaded after the policy, not before."""
    from c_ept.run import supervisor
    from c_ept.tests.test_run_ledger import make_header
    from c_ept.tests.test_run_numeric import OPPOSITE, write_flags
    profile = eval_profile(factory, tmp_path, role='STOP', numbers=(1,))
    manifest_path = write_eval_manifest(tmp_path, eval_manifest_document(world, factory.scales_for(S4B_FIELDS)))
    run_dir = tmp_path / 'run'
    run_ledger.Ledger.create(run_dir, make_header(identity=IDENTITY_A, profile_sha256=profile.sha256, runner={}))
    probe = 'c_ept.tests.run_evaluate_bodies:flags_probe_loader'
    arguments = (str(tmp_path / 'evaluate.json'), profile.sha256, str(manifest_path), probe, SYNTHETIC, 'cpu', '1', '32', '-', '-', '-', '-')
    write_flags(OPPOSITE)
    bodies.FLAGS_AT_LOAD.clear()
    with pytest.raises(RuntimeError, match='flags_probe_loader') as raised:
        run_evaluate.evaluate_stage(run_dir, *arguments)
    assert supervisor.exit_code_for(raised.value) == 3
    assert bodies.FLAGS_AT_LOAD == [{'cudnn_deterministic': True, 'deterministic_algorithms': True, 'float32_matmul_precision': 'highest'}]


@pytest.mark.parametrize('role', ['DEV', 'TEST'])
def test_the_library_start_refuses_another_stage_body_for_dev_and_test_unless_the_fixture_switch_is_set(factory, world, tmp_path, toy_defaults, role):
    """m4 of the K6c review: as the command line does, ``evaluate.start`` takes the default stage body only for DEV and TEST, before anything is reserved or opened."""
    access, lock, lock_sha = sealed(tmp_path)
    rig = EvalRig(tmp_path, factory, world, role=role, name=role.lower(), toy=True, holdout_lock_sha256=lock_sha)
    given = dict(access_dir=access.folder, lock_path=lock, allow_custom_body=False)
    assert rig.start(**given) == 2
    assert 'default stage body' in rig.stream.getvalue() and 'allow_custom_body' in rig.stream.getvalue()
    assert not rig.run_dir.exists() and access.test_opened() is False and len(access.entries()) == 1       # only the created line: nothing was logged


def test_a_custom_stage_body_is_free_for_the_role_stop(factory, world, tmp_path, toy_defaults):
    stop = EvalRig(tmp_path, factory, world, role='STOP', name='stop')
    assert stop.start(bodies={run_evaluate.STAGE: TOY_BODY}) == 0, stop.stream.getvalue()                 # the role STOP has no rehearsal to be equal to: any body


def test_the_stage_body_refuses_a_file_that_changed_between_the_hash_of_the_supervisor_and_its_own_start(factory, world, tmp_path):
    from c_ept.tests.test_run_ledger import make_header
    profile = eval_profile(factory, tmp_path, role='STOP', numbers=(1,))
    manifest_path = write_eval_manifest(tmp_path, eval_manifest_document(world, factory.scales_for(S4B_FIELDS)))
    wrong = {**run_evaluate.body_record(TOY_BODY), 'file_sha256': '0' * 64}                             # what the header says the file was
    run_dir = tmp_path / 'run'
    run_ledger.Ledger.create(run_dir, make_header(identity=IDENTITY_A, profile_sha256=profile.sha256, runner={'evaluation': {'sources': {'body': wrong}}}))
    arguments = (str(tmp_path / 'evaluate.json'), profile.sha256, str(manifest_path), TOY_LOADER, SYNTHETIC, 'cpu', '1', '32', '-', '-', '-', '-')
    with pytest.raises(run_evaluate.EvaluationRefused, match='body'):
        run_evaluate.evaluate_stage(run_dir, *arguments)
    assert not (run_dir / run_evaluate.RECORD_FILE).exists()


def test_the_evaluation_sets_the_numeric_policy_before_it_builds_a_model_and_records_the_state(factory, tmp_path):
    from c_ept.tests.test_run_numeric import OPPOSITE, read_flags, write_flags
    seen = []

    class Spy(FakeBackend):
        def build_model(self, trained, static, device):
            seen.append(read_flags())
            return super().build_model(trained, static, device)
    write_flags(OPPOSITE)
    result = run_evaluate.evaluate(eval_profile(factory, tmp_path), tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=Spy())
    assert seen == [{name: run_numeric.REQUIRED[name] for name in OPPOSITE}]
    numeric = result['record']['numeric']
    assert numeric['requested'] == dict(run_numeric.REQUIRED) and numeric['measured']['device'] == 'cpu'
    assert all(numeric['measured'][name] == run_numeric.REQUIRED[name] for name in run_numeric.REQUIRED)
    assert run_io.read_json(tmp_path / 'run' / run_evaluate.RECORD_FILE)['numeric'] == numeric and torch.cuda.is_initialized() is False


def test_a_state_that_is_not_the_policy_refuses_the_evaluation_before_anything_is_built_or_written(factory, tmp_path):
    from c_ept.run import supervisor
    backend = FakeBackend()
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(run_numeric.NumericRefused, match='default_dtype') as raised:
            run_evaluate.evaluate(eval_profile(factory, tmp_path), tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=backend)
    finally:
        torch.set_default_dtype(torch.float32)
    assert supervisor.exit_code_for(raised.value) == 2 and backend.log == [] and not (tmp_path / 'run').exists()


def test_the_header_of_an_evaluation_holds_the_requested_policy_and_the_approval_as_given(factory, world, tmp_path):
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='stop')
    assert rig.start(approval=APPROVAL) == 0, rig.stream.getvalue()
    runner = rig.ledger().header.runner
    assert runner['numeric'] == run_numeric.requested_record()
    assert runner['approval'] == {'text': APPROVAL, 'sha256': hashlib.sha256(APPROVAL.encode('utf-8')).hexdigest()}
    assert run_io.read_json(rig.run_dir / run_evaluate.RECORD_FILE)['numeric']['requested_sha256'] == runner['numeric']['requested_sha256']
    plain = EvalRig(tmp_path, factory, world, role='STOP', name='plain')
    assert plain.start() == 0 and plain.ledger().header.runner['approval'] is None
    for bad in ('', '   ', 5, ['text']):
        refused = EvalRig(tmp_path, factory, world, role='STOP', name=f'bad{abs(hash(str(bad)))}')
        assert refused.start(approval=bad) == 2 and not refused.run_dir.exists() and 'approval' in refused.stream.getvalue()


# ---- the composed V_start pass on a sample of STOP (K6e; decisions D9, D14, D30; the hand-offs of M11, M12, M14) ---------------------------------------
from c_ept.ept import compose_heads                                                                     # noqa: E402
from c_ept.evaluation import adapters as eval_adapters                                                  # noqa: E402
from c_ept.evaluation import inference as eval_inference                                                # noqa: E402
from c_ept.evaluation import proper as eval_proper                                                      # noqa: E402
from c_ept.run import vstart as run_vstart                                                              # noqa: E402
from c_ept.valuation import compose as val_compose                                                      # noqa: E402

COMPOSED = {'sample_matches': 1, 'sample_seed': 3, 'draws': 8, 'mc_batches': 4, 'draw_seed': 20261022}      # the toy world has one STOP match; 8 draws in 4 batches of 2
HORIZON_KEYS = ('60', '120', '180')


def v_start_labels(factory) -> dict:
    """``labels/v_start`` of the toy world (the heads world has none): made-up realised values of the three horizons for every frame row, from a fixed seed."""
    return {'labels/v_start': np.random.default_rng(5).normal(0.0, 300.0, (len(factory.world.s1.array('frame_rows/t0')), 3))}


def composed_inputs(factory, **extra):
    """Inputs of the plain toy configuration with ``labels/v_start`` added: what the composed pass reads besides the plain STOP inference."""
    dropped = tuple(run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN)).stats.dropped)
    return dataclasses.replace(bodies.real_eval_inputs(factory.inputs_for({**S4B_FIELDS, **PLAIN}), dropped=dropped, labels_extra=v_start_labels(factory)), **extra)


def composed_profile(factory, tmp_path, name='composed.json', numbers=(1,), over=None, composed=None, exposure='nominal', **fields):
    """The evaluate profile of the composed pass on the plain toy configuration: the role STOP, the nominal exposure and the block ``COMPOSED`` (``composed`` replaces some of its keys; False leaves the block out)."""
    block = {} if composed is False else {'composed': {**COMPOSED, **(composed or {})}}
    return eval_profile(factory, tmp_path, numbers=numbers, name=name, over={**PLAIN, **(over or {})}, exposure=exposure, **block, **fields)


def composed_run(factory, tmp_path, name='run', backend=None, **kwargs):
    profile = composed_profile(factory, tmp_path, name=f'{name}.json', **kwargs)
    backend = backend if backend is not None else RecordingBackend()
    result = run_evaluate.evaluate(profile, tmp_path / name, inputs=composed_inputs(factory), constants=CPU, backend=backend)
    return SimpleNamespace(result=result, backend=backend, profile=profile, folder=tmp_path / name)


@pytest.fixture(scope='module')
def composed_default(factory, tmp_path_factory):
    """One composed run of the plain toy configuration, seed 1, with ``COMPOSED``: read by several tests."""
    return composed_run(factory, tmp_path_factory.mktemp('composed_default'))


def rows_of(run, name='seed1'):
    """The per-row arrays of a composed run (checked against the content hash that its report names) and the meta of the file."""
    report = run.result['report']
    meta, payloads = run_vstart.read_rows(run.folder / run_vstart.ROWS_FILE, report['v_start_rows']['sha256'])
    return meta, payloads[name]


def reference_pass(trained, inputs, block, exposure, table=None, chunk_views=None):
    """The pass of the brief done by hand: a Composer of the settings of the checkpoint on ``table`` (all STOP rows when None), the EPT chain of the model, ``simulate``, and the row loss of its ``q19``."""
    model = run_evaluate.Backend().build_model(trained, inputs.base.static, 'cpu')
    table = stop_table(inputs) if table is None else table
    realized = np.asarray(inputs.base.labels.array('labels/v_start'))[table.frame_row]
    composer = val_compose.Composer(compose_heads.EPTComposerHeads(model, trained.stats, chunk_views=chunk_views), inputs.prices, inputs.tier,
                                    np.asarray(inputs.base.labels.array('labels/gold_reps'), dtype=np.float64), table, inputs.base.s2, S=block['draws'], seed=block['draw_seed'],
                                    **run_evaluate.composer_settings(trained.settings, exposure))
    q19 = composer.simulate(np.arange(len(table))).q19
    return q19, run_vstart.row_loss(q19, realized), realized


def stop_table(inputs):
    return val_compose.row_table(inputs.base.s1, inputs.base.s2, ('STOP',))


def sub_table(table, rows):
    return dataclasses.replace(table, **{name: getattr(table, name)[rows] for name in ('frame_row', 'match', 't0_ms', 'exposure_ms', 'T_ms', 'views', 'destroyed')})


def test_the_composed_pass_is_off_by_default_and_a_null_block_leaves_every_hash_and_record_as_they_were(factory, tmp_path):
    """The default (``composed`` null) is the evaluation of every earlier profile: no new key in the record or the report, the text of the unavailable V_start is the old one, the declared outputs of the stage
    are the record and the report, and the profile hash without the role is the hash of the profile as it was before the field existed (the null block is left out of it)."""
    profile = eval_profile(factory, tmp_path)
    assert profile['composed'] is None
    result = run_evaluate.evaluate(profile, tmp_path / 'run', inputs=eval_inputs(factory), constants=CPU, backend=FakeBackend())
    assert 'composed' not in result['record'] and 'v_start_rows' not in result['report'] and not (tmp_path / 'run' / run_vstart.ROWS_FILE).exists()
    with pytest.raises(run_evaluate.Unavailable) as raised:
        run_evaluate.Backend().v_start(object(), run_evaluate.load_trained('seed1', seed_run(factory, 1)), eval_inputs(factory), role='STOP', profile=profile, reps=None, constants=CPU, check=None)
    assert raised.value.reason == run_evaluate.V_START_UNAVAILABLE
    assert run_evaluate.stage_outputs(profile) == (run_evaluate.RECORD_FILE, run_evaluate.REPORT_FILE)
    old = run_io.canonical_sha256({key: val for key, val in profile.as_dict().items() if key not in ('role', 'composed')})
    assert run_evaluate.profile_without_role_sha256(profile) == old
    on = composed_profile(factory, tmp_path)
    assert run_evaluate.profile_without_role_sha256(on) == run_io.canonical_sha256({key: val for key, val in on.as_dict().items() if key != 'role'}) != old
    assert run_evaluate.stage_outputs(on) == (run_evaluate.RECORD_FILE, run_evaluate.REPORT_FILE, run_vstart.ROWS_FILE)
    other = composed_profile(factory, tmp_path, name='other.json', composed={'draws': 16})
    assert run_evaluate.profile_without_role_sha256(other) != run_evaluate.profile_without_role_sha256(on)                 # every parameter of the block is part of the lock


def test_the_composed_pass_is_the_composer_of_the_checkpoint_on_every_row_of_the_sample(factory, composed_default):
    """The nominal pass: the losses of the rows file are, bit for bit, the row loss of ``q19`` of the Composer that the test builds by hand (EPT chain of the model, the kill law and the zone mark of the identity,
    the nominal exposure, ``labels/gold_reps`` for a raw checkpoint, S and the seed of the block). The observed pass is the same on the rows that the longest horizon cuts."""
    run, inputs = composed_default, composed_inputs(factory)
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN))
    meta, payload = rows_of(run)
    table = stop_table(inputs)
    q19, loss, realized = reference_pass(trained, inputs, COMPOSED, 'nominal', table)
    assert np.array_equal(payload['frame_row'], table.frame_row) and np.array_equal(payload['match'], table.match)
    masks = eval_adapters.primary_masks(inputs.base.s1, table.frame_row)
    primary = np.column_stack([masks[h] for h in (60, 120, 180)])
    assert np.array_equal(payload['primary'], primary) and primary.any(axis=0).all() and (~primary).any(axis=0).all()                  # both populations exist at every horizon of the toy world
    assert np.array_equal(payload['loss'], loss)
    cut = np.flatnonzero(~primary[:, -1])
    _, observed, _ = reference_pass(trained, inputs, COMPOSED, 'observed', sub_table(table, cut))
    assert np.array_equal(payload['loss_observed'][cut], observed) and np.isnan(np.delete(payload['loss_observed'], cut, axis=0)).all()
    assert payload['loss_batches'].shape == (4, len(table), 3) and payload['loss_observed_batches'].shape == (4, len(table), 3)
    part = run.result['report']['models']['seed1']['v_start']
    match = table.match
    for j, h in enumerate(HORIZON_KEYS):
        first, second = primary[:, j], ~primary[:, j]
        weights = features_weights(match[first])
        assert part['by_horizon'][h]['primary']['crps19'] == eval_proper.crps19(q19[first, j], realized[first, j], weights)
        assert part['by_horizon'][h]['primary']['n_rows'] == int(first.sum()) and part['by_horizon'][h]['secondary']['n_rows'] == int(second.sum())
    assert part['n_rows'] == len(table) and part['n_matches'] == 1 and json.loads(json.dumps(part)) == part


def features_weights(match):
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse.reshape(-1)]


def test_the_report_names_the_rows_file_by_its_content_and_the_record_names_the_sample_and_the_settings(factory, composed_default):
    run = composed_default
    report, record = run.result['report'], run.result['record']
    assert report['v_start_rows']['file'] == run_vstart.ROWS_FILE and len(report['v_start_rows']['sha256']) == 64 and report['v_start_rows']['models'] == ['seed1']
    meta, _ = rows_of(run)
    assert meta['models']['seed1']['seed'] == SEEDS[0] and (meta['draws'], meta['mc_batches'], meta['draw_seed']) == (8, 4, 20261022)
    composed = record['composed']
    sample = composed['sample']
    stop = [int(m) for m in run_evaluate.Backend().split_ids(composed_inputs(factory), 'STOP')]
    assert (sample['matches_requested'], sample['matches_selected'], sample['pool_matches'], sample['seed']) == (1, 1, 1, 3) and sample['ids_sha256'] == run_vstart.ids_sha256(stop)
    assert set(sample) == {'matches_requested', 'matches_selected', 'pool_matches', 'seed', 'ids_sha256', 'rule', 'numpy'} and meta['sample'] == sample         # the record names the sample by its hash, not by its ids
    assert set(np.unique(rows_of(run)[1]['match'])) == set(stop)                                                                          # the match numbers are in the rows file only
    assert composed['draws'] == 8 and composed['mc_batches'] == 4 and composed['draw_seed'] == 20261022 and composed['exposure'] == {'primary': 'nominal', 'secondary': 'observed'}
    assert composed['mc_error'] == 'batch means: sd of the statistic over the 4 batches of 2 draws / sqrt(4)' and composed['bootstrap_B'] == 2000
    settings = record['composed']['models']['seed1']
    assert (settings['kill_law'], settings['zone_mark'], settings['k_condition'], settings['eval_average']) == ('poisson', True, 'raw', True) and settings['gold_reps']['source'] == 'labels'
    assert settings['device'] == 'cpu' and 'chunk_views' not in settings and 'batch_size' not in settings                                  # chunk_views is recorded only when it is set
    assert not [u for u in record['unavailable'] if u['item'] in ('v_start', 'occurrence')]


def test_every_ept_chain_of_the_pass_gets_the_chunk_views_of_the_profile_and_never_the_batch_size(factory, tmp_path, monkeypatch):
    """M14 hand-off: both chains (the nominal one and the one of the observed pass) are ``EPTComposerHeads(model, stats, chunk_views=profile.chunk_views)``; the value is recorded only when it is not None; the
    batch size of the inference (32) is another thing; a chunk of 4 views and no chunking give the same pass within the tolerance of the chunked encoding."""
    made = []

    class Spy(compose_heads.EPTComposerHeads):
        def __init__(self, model, stats, **kwargs):
            made.append((model, stats, kwargs))
            super().__init__(model, stats, **kwargs)
    monkeypatch.setattr(compose_heads, 'EPTComposerHeads', Spy)
    chunked = composed_run(factory, tmp_path, name='chunked', chunk_views=4)
    assert len(made) == 2 and all(kwargs == {'chunk_views': 4} for _, _, kwargs in made) and all(stats is made[0][1] for _, stats, _ in made)
    assert chunked.result['record']['composed']['models']['seed1']['chunk_views'] == 4 and 'batch_size' not in chunked.result['record']['composed']['models']['seed1']
    made.clear()
    plain = composed_run(factory, tmp_path, name='unchunked')
    assert len(made) == 2 and all(kwargs == {'chunk_views': None} for _, _, kwargs in made)
    a, b = rows_of(chunked)[1], rows_of(plain)[1]
    assert np.allclose(a['loss'], b['loss'], rtol=1e-3, atol=1e-6)                                                                         # the chunked encoding moves the heads at the last digits only


def test_a_model_without_the_frame_average_is_composed_without_it(factory, tmp_path, monkeypatch):
    """The model object carries no identity: ``eval_average`` False of the checkpoint reaches the chain as ``FrameAveraged(model, average=False)``, as it reaches ``run_inference``."""
    from c_ept.ept import heads as ept_heads
    made = []

    class Spy(compose_heads.EPTComposerHeads):
        def __init__(self, model, stats, **kwargs):
            made.append(model)
            super().__init__(model, stats, **kwargs)
    monkeypatch.setattr(compose_heads, 'EPTComposerHeads', Spy)
    composed_run(factory, tmp_path, over={'eval_average': False})
    assert len(made) == 2 and all(isinstance(chain, ept_heads.FrameAveraged) and chain.average is False for chain in made)


def test_the_same_seed_gives_the_same_pass_and_another_draw_seed_another_one(factory, tmp_path, composed_default):
    again = composed_run(factory, tmp_path, name='again')
    first, second = rows_of(composed_default)[1], rows_of(again)[1]
    assert all(np.array_equal(first[f], second[f], equal_nan=first[f].dtype.kind == 'f') for f in run_vstart.ROW_FIELDS)
    assert composed_default.result['report']['v_start_rows']['sha256'] == again.result['report']['v_start_rows']['sha256']
    assert composed_default.result['report']['models'] == again.result['report']['models']
    other = composed_run(factory, tmp_path, name='other', composed={'draw_seed': 20261023})
    assert not np.array_equal(first['loss'], rows_of(other)[1]['loss'])


def test_the_monte_carlo_error_of_the_toy_pass_falls_with_the_draws(factory, tmp_path):
    """The same sample and seeds with 8 and with 128 draws (four batches each): the standard error of the primary CRPS, averaged over the three horizons, is smaller with more draws (about one over
    sqrt(16) = a quarter in expectation; the test asks for less than 0.8, with the fixed seeds)."""
    ses = {}
    for draws in (8, 128):
        part = composed_run(factory, tmp_path, name=f'draws{draws}', composed={'draws': draws}).result['report']['models']['seed1']['v_start']
        ses[draws] = np.mean([part['by_horizon'][h]['primary']['mc']['se'] for h in HORIZON_KEYS])
        assert part['by_horizon']['60']['primary']['mc']['batches'] == 4
    assert ses[8] > 0 and ses[128] > 0 and ses[128] / ses[8] < 0.8


def test_the_occurrence_calibration_is_the_citl_of_the_closed_form_probability_on_the_sample_rows(factory, composed_default):
    """Next to the CRPS the report holds, per horizon, the mean composed occurrence probability, the observed rate and their difference (CITL) of the EPT and of the matched B1, on the sampled rows and on
    the whole primary population of the split; the numbers are those of the endpoints ``M2/t0/H*/occurrence`` of the primary inference pass."""
    part = composed_default.result['report']['models']['seed1']['v_start']
    primary = next(result for seed, role, exposure, population, result in composed_default.backend.results if seed == SEEDS[0] and population == 'primary')
    for h in (60, 120, 180):
        ep = next(e for e in primary['endpoints'] if e['name'] == f'M2/t0/H{h}/occurrence')
        entry = part['occurrence'][str(h)]['sample']['ept']
        assert entry['n_rows'] == len(ep['row']) and entry['mean_p'] == pytest.approx(float(np.mean(ep['p_ept']))) and entry['observed_rate'] == pytest.approx(float(np.mean(ep['y'])))
        assert entry['citl'] == pytest.approx(entry['mean_p'] - entry['observed_rate']) and part['occurrence'][str(h)]['sample']['b1']['mean_p'] == pytest.approx(float(np.mean(ep['p_b1'])))
        assert part['occurrence'][str(h)]['stop']['ept'] == entry                                                                           # the toy split has the one sampled match


def test_the_pass_writes_a_begin_and_end_line_for_every_block_with_counts_only(factory, composed_default):
    lines = progress_lines(composed_default.folder)
    blocks = [line for line in lines if line['stage'] == 'block' and line['detail'].get('pass') == 'v_start']
    assert blocks and {line['detail']['block'].split('/')[0] for line in blocks} <= {'bind', 'nominal', 'observed'} and {line['event'] for line in blocks} == {'begin', 'end'}
    ends = [line for line in blocks if line['event'] == 'end']
    table = stop_table(composed_inputs(factory))
    assert sum(line['counts']['rows'] for line in ends if line['detail']['block'].startswith('nominal')) == len(table)
    assert all(line['counts'].get('draws') in (None, 8) for line in ends) and all(line['status'] == 'ok' for line in ends)
    assert len([line for line in lines if line['stage'] == 'v_start' and line['event'] == 'end']) == 1
    for line in lines:
        assert not any(word in key for key in (*line, *line.get('counts', {}), *line['detail']) for word in ('crps', 'loss', 'share', 'citl'))


def test_the_zone_contrast_of_two_runs_is_the_difference_of_their_crps_on_the_same_rows(factory, tmp_path, composed_default):
    """A7: the run with the zone mark and the run without it (two configurations, so two records) share the sample, the seeds and the streams; ``paired_contrast`` of their rows files is the difference of
    the CRPS of their primary rows, and the record of each run names the same sample hash."""
    off = composed_run(factory, tmp_path, name='zone_off', over={'zone_mark': False})
    on_meta, on_rows = run_vstart.read_rows(composed_default.folder / run_vstart.ROWS_FILE)
    off_meta, off_rows = run_vstart.read_rows(off.folder / run_vstart.ROWS_FILE)
    assert on_meta['sample'] == off_meta['sample'] and on_meta['models']['seed1']['zone_mark'] is True and off_meta['models']['seed1']['zone_mark'] is False
    ids = run_evaluate.Backend().split_ids(composed_inputs(factory), 'STOP')
    matrix = eval_inference.resample_matrix(len(ids))[:2000]
    out = run_vstart.contrast_rows_files(composed_default.folder / run_vstart.ROWS_FILE, off.folder / run_vstart.ROWS_FILE, ids, matrix,
                                         zone_on_sha256=composed_default.result['report']['v_start_rows']['sha256'], zone_off_sha256=off.result['report']['v_start_rows']['sha256'])
    assert out['sample_ids_sha256'] == on_meta['sample']['ids_sha256'] and out == {**out, **run_vstart.paired_contrast(run_vstart.by_seed(on_meta, on_rows), run_vstart.by_seed(off_meta, off_rows), ids, matrix)}
    with pytest.raises(run_vstart.VStartRefused, match='zone mark'):                                       # the same side twice is no zone comparison
        run_vstart.contrast_rows_files(composed_default.folder / run_vstart.ROWS_FILE, composed_default.folder / run_vstart.ROWS_FILE, ids, matrix)
    for h in HORIZON_KEYS:
        got = out['by_horizon'][h]
        want = composed_default.result['report']['models']['seed1']['v_start']['by_horizon'][h]['primary']['crps19'] - off.result['report']['models']['seed1']['v_start']['by_horizon'][h]['primary']['crps19']
        assert got['delta'] == pytest.approx(want, rel=1e-12, abs=1e-12) and got['n_rows'] == off.result['report']['models']['seed1']['v_start']['by_horizon'][h]['primary']['n_rows']
        assert got['mc']['batches'] == 4 and got['bootstrap']['B'] == 2000
    assert not np.array_equal(on_rows['seed1']['loss'], off_rows['seed1']['loss'])
    assert np.array_equal(on_rows['seed1']['frame_row'], off_rows['seed1']['frame_row'])


def test_the_composed_pass_runs_on_the_role_stop_and_the_nominal_exposure_only_and_chunk_views_needs_the_pass(factory, tmp_path):
    """DEV and TEST stay out of this pass in S4b (D30): the role must be STOP, whatever the backend. The nominal exposure is the T-blind forecast of D14. ``chunk_views`` has no other reader than the pass: set
    without it, it would be recorded but used by nothing, so the profile is refused."""
    backend = FakeBackend()
    for fields, word in (({'exposure': 'observed'}, 'nominal'), ({'chunk_views': 4, 'composed': False}, 'chunk_views')):
        profile = composed_profile(factory, tmp_path, name=f'{word}.json', **fields)
        with pytest.raises(run_evaluate.EvaluationRefused, match=word):
            run_evaluate.evaluate(profile, tmp_path / f'run_{word}', inputs=eval_inputs(factory), constants=CPU, backend=backend)
    assert backend.log == [] and not (tmp_path / 'run_nominal').exists()
    for role in ('DEV', 'TEST'):
        dev_profile = evaluate_profile(tmp_path, f'{role}.json', role=role, models=['seed1'], checkpoints={'seed1': str(seed_run(factory, 1))}, exposure='nominal', composed=COMPOSED,
                                       **({'holdout_lock_sha256': 'ab' * 32} if role == 'TEST' else {}))
        with pytest.raises(run_evaluate.EvaluationRefused, match='STOP'):
            run_evaluate.check_composed(dev_profile)
        with pytest.raises(run_evaluate.EvaluationRefused, match='STOP'):
            run_evaluate.Backend().v_start(object(), run_evaluate.load_trained('seed1', seed_run(factory, 1)), eval_inputs(factory), role=role, profile=dev_profile, reps=None, constants=CPU, check=None)
    run_evaluate.check_composed(eval_profile(factory, tmp_path, name='off.json'))                           # the default profile is not refused
    run_evaluate.check_composed(composed_profile(factory, tmp_path, name='on.json', chunk_views=4))


def test_a_supervised_start_refuses_a_composed_profile_for_the_wrong_role_before_anything_is_reserved(factory, world, tmp_path):
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='wrong', numbers=(1,), exposure='observed', composed=COMPOSED)
    assert rig.start() == 2 and 'nominal' in rig.stream.getvalue() and not rig.run_dir.exists()


def test_the_pass_says_what_it_lacks_and_refuses_a_sample_that_the_pool_cannot_give(factory, tmp_path):
    """No prices, no tier model or no ``labels/v_start`` is an unavailable item with its reason (the other parts still run); a sample larger than the pool of valued matches is a refusal (the run is
    not made with fewer matches than the profile asked for), and so is a ``labels/v_start`` that is NaN on a whole match only when no match is left."""
    trained = run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN))
    profile = composed_profile(factory, tmp_path)
    model = run_evaluate.Backend().build_model(trained, composed_inputs(factory).base.static, 'cpu')
    for field, word in (('prices', 'prices'), ('tier', 'tier')):
        with pytest.raises(run_evaluate.Unavailable, match=word):
            run_evaluate.Backend().v_start(model, trained, composed_inputs(factory, **{field: None}), role='STOP', profile=profile, reps=None, constants=CPU, check=None)
    without = plain_inputs(factory)                                                                          # no labels/v_start
    with pytest.raises(run_evaluate.Unavailable, match='labels/v_start'):
        run_evaluate.Backend().v_start(model, trained, without, role='STOP', profile=profile, reps=None, constants=CPU, check=None)
    too_big = composed_profile(factory, tmp_path, name='big.json', composed={'sample_matches': 2})
    with pytest.raises(run_evaluate.EvaluationRefused, match='1 valued matches'):
        run_evaluate.Backend().v_start(model, trained, composed_inputs(factory), role='STOP', profile=too_big, reps=None, constants=CPU, check=None)
    nan_inputs = bodies.real_eval_inputs(factory.inputs_for({**S4B_FIELDS, **PLAIN}), dropped=tuple(trained.stats.dropped),
                                         labels_extra={'labels/v_start': np.full((len(factory.world.s1.array('frame_rows/t0')), 3), np.nan)})
    with pytest.raises(run_evaluate.EvaluationRefused, match='0 valued matches'):
        run_evaluate.Backend().v_start(model, trained, nan_inputs, role='STOP', profile=profile, reps=None, constants=CPU, check=None)


def test_the_gold_representatives_of_the_composer_are_the_evaluations_when_binned_and_the_label_stores_when_raw(factory, tmp_path, monkeypatch):
    """Binned checkpoints compose with the evaluation's 31 representatives (``evaluation_reps``); a record whose sha256 is not the training's ``k_reps_sha256`` (or a missing one, or an array that does not have
    the hash it carries) is refused before any composer is made; raw checkpoints compose with ``labels/gold_reps``. The kill law and the zone mark come from the identity."""
    seen = []

    class Spy:
        def __init__(self, heads_, prices, tier, gold_reps, rows, s2, **kwargs):
            self.S, self.rows = kwargs['S'], rows
            seen.append((np.array(gold_reps), kwargs, len(rows)))

        def simulate_onsets(self, row):
            return SimpleNamespace(sim=np.empty(0, dtype=np.int64), q=np.empty(0, dtype=np.int64), value=np.empty(0))
    monkeypatch.setattr(val_compose, 'Composer', Spy)
    backend = run_evaluate.Backend()
    binned = run_evaluate.load_trained('seed1', seed_run(factory, 1))
    inputs = bodies.real_eval_inputs(factory.inputs_for(S4B_FIELDS), dropped=tuple(binned.stats.dropped), labels_extra=v_start_labels(factory))
    reps = run_evaluate.evaluation_reps(inputs)
    profile = eval_profile(factory, tmp_path, name='gold.json', exposure='nominal', composed=COMPOSED)
    model = backend.build_model(binned, inputs.base.static, 'cpu')
    backend.v_start(model, binned, inputs, role='STOP', profile=profile, reps=reps, constants=CPU, check=None)
    labels_reps = np.asarray(inputs.base.labels.array('labels/gold_reps'), dtype=np.float64)
    assert [kwargs for _, kwargs, _ in seen] == [{'S': 8, 'seed': 20261022, 'kill_law': 'shifted', 'zone_mark': True, 'exposure': 'nominal'},
                                                  {'S': 8, 'seed': 20261022, 'kill_law': 'shifted', 'zone_mark': True, 'exposure': 'observed'}]
    assert all(np.array_equal(array, reps['array']) and not np.array_equal(array, labels_reps) for array, _, _ in seen)
    other = reps['array'] + 1.0
    for bad in (None, {**reps, 'sha256': 'f' * 64}, {**reps, 'array': reps['array'] + 1.0}, {**reps, 'sha256': value.gold_reps_sha256(other), 'array': other}):
        seen.clear()
        with pytest.raises(run_evaluate.EvaluationRefused, match='k_reps_sha256|representatives'):
            backend.v_start(model, binned, inputs, role='STOP', profile=profile, reps=bad, constants=CPU, check=None)
        assert seen == []                                                                                    # refused before any composer is made
    raw = run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN))
    raw_inputs = composed_inputs(factory)
    seen.clear()
    backend.v_start(backend.build_model(raw, raw_inputs.base.static, 'cpu'), raw, raw_inputs, role='STOP', profile=composed_profile(factory, tmp_path, name='raw.json'), reps=None, constants=CPU, check=None)
    assert len(seen) == 2 and all(np.array_equal(array, labels_reps) and kwargs['kill_law'] == 'poisson' for array, kwargs, _ in seen)
    zone_off = run_evaluate.load_trained('seed1', seed_run(factory, 1, **PLAIN, zone_mark=False))
    seen.clear()
    backend.v_start(backend.build_model(zone_off, raw_inputs.base.static, 'cpu'), zone_off, raw_inputs, role='STOP', profile=composed_profile(factory, tmp_path, name='zoneoff.json'), reps=None,
                    constants=CPU, check=None)
    assert all(kwargs['zone_mark'] is False for _, kwargs, _ in seen)


def test_the_execution_of_a_gpu_evaluation_records_the_device_and_the_determinism_settings():
    """M14 hand-off (m7): on a CUDA device the record names the device and the determinism settings that torch has in the process (read, never set); on the CPU the device only."""
    cpu = run_evaluate.execution_record(runners.RunnerConstants(device='cpu', threads=1))
    assert cpu == {'device': 'cpu'}
    torch.use_deterministic_algorithms(True)
    gpu = run_evaluate.execution_record(runners.RunnerConstants(device='cuda', threads=1))
    assert gpu['device'] == 'cuda' and gpu['deterministic_algorithms'] is True and gpu['cudnn_deterministic'] is torch.backends.cudnn.deterministic
    assert gpu['cudnn_benchmark'] is torch.backends.cudnn.benchmark and gpu['float32_matmul_precision'] == torch.get_float32_matmul_precision() and gpu['torch'] == torch.__version__
    torch.use_deterministic_algorithms(False)
    assert run_evaluate.execution_record(runners.RunnerConstants(device='cuda', threads=1))['deterministic_algorithms'] is False


# ---- the sample of matches on a world with three STOP matches -------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def world3(tmp_path_factory):
    """The toy EPT world with three STOP matches (the other has one): the sample of the composed pass picks among them."""
    built = build_ept_world(tmp_path_factory.mktemp('evaluate_world3'), roles=('FIT', 'FIT', 'FIT', 'FIT', 'STOP', 'STOP', 'STOP', 'CAL'), n_events=7, n_edges=5, consistent_first_kill=True)
    yield built
    del built
    gc.collect()                                                                                      # Windows keeps a folder with open memory maps


@pytest.fixture(scope='module')
def factory3(tmp_path_factory, world3):
    reps = value.gold_reps_sha256(value.fit_gold_bin_means(world3.s1, world3.labels, population='in_scope').means)
    return Factory(tmp_path_factory.mktemp('evaluate_runs3'), world3, reps)


def test_the_sample_of_matches_is_applied_to_the_rows_and_named_by_its_hash(factory3, tmp_path):
    """Three STOP matches, a sample of two: the rows of the pass are those of the two matches that ``vstart.sample_matches`` names for the seed (the third match has no row in the file), the losses are the
    Composer of those rows and no others, the record names the sample by its hash, the report counts two matches and the occurrence calibration has two matches on the sample and three on the split."""
    inputs = composed_inputs(factory3)
    stop = np.asarray(run_evaluate.Backend().split_ids(inputs, 'STOP'))
    assert len(stop) == 3
    block = {**COMPOSED, 'sample_matches': 2, 'sample_seed': 11}
    run = composed_run(factory3, tmp_path, composed={'sample_matches': 2, 'sample_seed': 11})
    chosen = run_vstart.sample_matches(stop, 2, 11)
    meta, payload = rows_of(run)
    table = stop_table(inputs)
    keep = np.isin(table.match, chosen)
    assert keep.any() and not keep.all() and set(np.unique(payload['match'])) == set(chosen) and np.array_equal(payload['frame_row'], table.frame_row[keep])
    trained = run_evaluate.load_trained('seed1', seed_run(factory3, 1, **PLAIN))
    _, loss, _ = reference_pass(trained, inputs, block, 'nominal', sub_table(table, keep))
    assert np.array_equal(payload['loss'], loss)                                                      # the Composer of the sampled rows alone
    sample = run.result['record']['composed']['sample']
    assert (sample['matches_requested'], sample['matches_selected'], sample['pool_matches']) == (2, 2, 3) and sample['ids_sha256'] == run_vstart.ids_sha256(chosen)
    part = run.result['report']['models']['seed1']['v_start']
    assert part['n_matches'] == 2 and part['n_rows'] == int(keep.sum())
    occurrence = part['occurrence']['60']
    assert occurrence['sample']['ept']['n_matches'] == 2 and occurrence['stop']['ept']['n_matches'] == 3
    other = composed_run(factory3, tmp_path, name='whole', composed={'sample_matches': 3, 'sample_seed': 11})
    assert set(np.unique(rows_of(other)[1]['match'])) == set(stop) and other.result['record']['composed']['sample']['ids_sha256'] == run_vstart.ids_sha256(stop)


def test_a_match_without_a_finite_v_start_is_not_in_the_pool_of_the_sample(factory3, tmp_path):
    """``value.v_start`` gives NaN to every row of a match that is not valued: such a match is out of the pool whole, a sample of the size of the pool that is left is that pool, and a larger sample is a
    refusal that says how many valued matches there are."""
    inputs = composed_inputs(factory3)
    table = stop_table(inputs)
    labels = np.array(inputs.base.labels.array('labels/v_start'))
    unvalued = int(np.unique(table.match)[1])
    labels[table.frame_row[table.match == unvalued], 1] = np.nan                                       # one horizon of one row is enough: the match is out whole
    trained = run_evaluate.load_trained('seed1', seed_run(factory3, 1, **PLAIN))
    inputs = bodies.real_eval_inputs(factory3.inputs_for({**S4B_FIELDS, **PLAIN}), dropped=tuple(trained.stats.dropped), labels_extra={'labels/v_start': labels})
    model = run_evaluate.Backend().build_model(trained, inputs.base.static, 'cpu')
    with pytest.raises(run_evaluate.EvaluationRefused, match='2 valued matches'):
        run_evaluate.Backend().v_start(model, trained, inputs, role='STOP', profile=composed_profile(factory3, tmp_path, name='three.json', composed={'sample_matches': 3}), reps=None, constants=CPU, check=None)
    parts = run_evaluate.Backend().v_start(model, trained, inputs, role='STOP', profile=composed_profile(factory3, tmp_path, name='two.json', composed={'sample_matches': 2}), reps=None, constants=CPU, check=None)
    assert set(np.unique(parts['match'])) == set(np.unique(table.match)) - {unvalued} and parts['sample']['pool_matches'] == 2
    assert np.isfinite(parts['loss']).all() and np.isfinite(parts['loss_batches']).all()


def test_a_supervised_composed_run_declares_the_rows_file_and_the_ledger_hashes_it(factory, world, tmp_path):
    """The stage of a composed run declares the rows file next to the record and the report (``stage_outputs``), and the ledger records its hash. The toy inputs hold no prices, so the pass is unavailable
    and the file is written without rows (the stage declares it): the report says which models it holds (none) and the record says why the pass is missing."""
    rig = EvalRig(tmp_path, factory, world, role='STOP', name='composed', numbers=(1,), exposure='nominal', composed=COMPOSED)
    assert rig.start() == 0, rig.stream.getvalue()
    outputs = {Path(path).as_posix().split('/run/')[-1]: digest for path, digest in rig.ledger().stage('evaluate').outputs.items()}
    assert set(outputs) == {run_evaluate.RECORD_FILE, run_evaluate.REPORT_FILE, run_vstart.ROWS_FILE} and outputs[run_vstart.ROWS_FILE] == run_io.sha_file(rig.run_dir / run_vstart.ROWS_FILE)
    report = run_io.read_json(rig.run_dir / run_evaluate.REPORT_FILE)
    record = run_io.read_json(rig.run_dir / run_evaluate.RECORD_FILE)
    assert report['v_start_rows']['models'] == [] and record['composed']['sample'] is None and record['composed']['draws'] == 8
    assert [u['item'] for u in record['unavailable'] if 'prices' in u['reason']] == ['v_start'] and record['profile']['composed']['sample_matches'] == 1
    meta, payloads = run_vstart.read_rows(rig.run_dir / run_vstart.ROWS_FILE, report['v_start_rows']['sha256'])
    assert payloads == {} and meta['models'] == {} and meta['draw_seed'] == 20261022
    default = EvalRig(tmp_path, factory, world, role='STOP', name='default', numbers=(1,))
    assert default.start() == 0, default.stream.getvalue()
    assert {Path(path).as_posix().split('/run/')[-1] for path in default.ledger().stage('evaluate').outputs} == {run_evaluate.RECORD_FILE, run_evaluate.REPORT_FILE}
    assert 'composed' not in run_io.read_json(default.run_dir / run_evaluate.RECORD_FILE) and 'composed' not in run_io.read_json(default.run_dir / run_evaluate.RECORD_FILE)['profile']
