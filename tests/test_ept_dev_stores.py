"""Only invented temporary .npy fixtures; never open a production Store."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from c_ept.data.store_core import Store
from c_ept.evaluation import stores
from c_ept.evaluation.stores import ReadOnlyVerifiedStore


def save_manifest(root, manifest):
    (root / 'STORE.json').write_text(json.dumps(manifest), encoding='utf-8')


@pytest.fixture
def world(tmp_path):
    root = tmp_path / 'synthetic-store'
    root.mkdir()
    arrays = {'dev/x': np.arange(12, dtype='<f4').reshape(4, 3),
              'other/never_read': np.array([100, 200], dtype='<i8')}
    manifest = {'version': 1, 'complete': True, 'stages': {'build': {'blocks': [0]}}, 'arrays': {}, 'files': {}}
    for name, values in arrays.items():
        path = root / (name + '.npy')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values, allow_pickle=False)
        manifest['arrays'][name] = {'shape': list(values.shape), 'dtype': values.dtype.str}
        manifest['files'][name + '.npy'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_manifest(root, manifest)
    return root, manifest, arrays


def close_maps(store):
    for mapped in store.maps.values():
        mapped._mmap.close()


def trace_hashes(monkeypatch):
    paths = []
    original = stores._stream_hash

    def traced(path, *args, **kwargs):
        paths.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(stores, '_stream_hash', traced)
    return paths


def test_constructor_is_metadata_only_and_preserves_canonical_read_api(world, monkeypatch):
    root, manifest, arrays = world
    original_open = Path.open
    opened = []

    def metadata_only(path, *args, **kwargs):
        assert path == root / 'STORE.json', 'constructor attempted an array read'
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('constructor mapped an array'))
    monkeypatch.setattr(Store, 'verify', lambda *_: pytest.fail('whole-store verification forbidden'))
    monkeypatch.setattr(Path, 'rglob', lambda *a, **k: pytest.fail('directory enumeration forbidden'))
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    assert isinstance(store, Store)
    assert store.manifest == manifest and store.maps == {}
    assert store.done('build') == {0}
    assert store.path('dev/x') == root / 'dev/x.npy'
    assert store.used_evidence()['arrays'] == {}
    store.verify_used()
    assert opened and all(p.name == 'STORE.json' for p in opened)


def test_only_requested_file_hashed_then_memoized_and_finally_rehashed(world, monkeypatch):
    root, _, arrays = world
    paths = trace_hashes(monkeypatch)
    other = root / 'other/never_read.npy'
    original_open, original_load = Path.open, np.load
    loads = []

    def guarded_open(path, *args, **kwargs):
        assert path != other, 'unrequested array opened'
        return original_open(path, *args, **kwargs)

    def load(path, *args, **kwargs):
        assert Path(path) != other
        assert kwargs == {'mmap_mode': 'r', 'allow_pickle': False}
        loads.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(np, 'load', load)
    monkeypatch.setattr(Store, 'verify', lambda *_: pytest.fail('Store.verify forbidden'))
    monkeypatch.setattr(Path, 'rglob', lambda *a, **k: pytest.fail('all-files traversal forbidden'))
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        mapped = store.array('dev/x')
        assert isinstance(mapped, np.memmap) and mapped.mode == 'r'
        np.testing.assert_array_equal(mapped, arrays['dev/x'])
        assert store.array('dev/x') is mapped
        assert loads == [root / 'dev/x.npy']
        assert paths.count(root / 'dev/x.npy') == 1
        store.verify_used()
        assert paths.count(root / 'dev/x.npy') == 2
        store.verify_used()
        assert paths.count(root / 'dev/x.npy') == 3
        assert other not in paths
    finally:
        close_maps(store)


def test_readonly_memmap_and_evidence_have_no_values(world):
    root, manifest, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        mapped = store.array('dev/x')
        with pytest.raises(ValueError):
            mapped[0, 0] = 999
        with pytest.raises(ValueError):
            mapped.setflags(write=True)
        store.verify_used()
        evidence = store.used_evidence()
        item = evidence['arrays']['dev/x']
        assert item['path'] == str(root / 'dev/x.npy')
        assert item['sha256'] == manifest['files']['dev/x.npy']
        assert item['shape'] == [4, 3] and item['dtype'] == '<f4'
        assert item['stat']['bytes'] == (root / 'dev/x.npy').stat().st_size
        assert evidence['manifest']['sha256'] == hashlib.sha256((root / 'STORE.json').read_bytes()).hexdigest()
        assert 'other/never_read' not in json.dumps(evidence, allow_nan=False)
        assert set(item) == {'path', 'sha256', 'shape', 'dtype', 'stat'}
        evidence['arrays'].clear()
        assert 'dev/x' in store.used_evidence()['arrays']  # defensive copy
    finally:
        close_maps(store)


@pytest.mark.parametrize('mutator,args', [
    ('save', ()), ('allocate', ('dev/x', (1,), 'f4')), ('_rows_view', ('dev/x', [])),
    ('put_rows', (0, {})), ('put_at', ([], {})), ('mark_done', ('build', 2, [])),
    ('set_stage', ('build',)), ('finalize', ()), ('verify', ()), ('stored_files', ()), ('hashes', ([],)),
])
def test_mutators_and_whole_store_scans_disabled(world, mutator, args):
    root, _, _ = world
    original = (root / 'STORE.json').read_bytes()
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    with pytest.raises(TypeError, match='read-only'):
        getattr(store, mutator)(*args)
    assert (root / 'STORE.json').read_bytes() == original


@pytest.mark.parametrize('fault', ['missing_manifest', 'incomplete', 'complete_int', 'missing_arrays', 'missing_files', 'duplicate', 'nan', 'temporary'])
def test_constructor_rejects_incomplete_or_invalid_manifest(world, fault):
    root, manifest, _ = world
    if fault == 'missing_manifest':
        (root / 'STORE.json').unlink()
    elif fault == 'duplicate':
        (root / 'STORE.json').write_text('{"complete":true,"complete":false}', encoding='utf-8')
    elif fault == 'nan':
        (root / 'STORE.json').write_text('{"complete":NaN}', encoding='utf-8')
    elif fault == 'temporary':
        (root / 'STORE.json.tmp').write_text('in-progress', encoding='utf-8')
    else:
        if fault == 'incomplete': manifest['complete'] = False
        elif fault == 'complete_int': manifest['complete'] = 1
        elif fault == 'missing_arrays': del manifest['arrays']
        else: del manifest['files']
        save_manifest(root, manifest)
    with pytest.raises((ValueError, OSError)):
        ReadOnlyVerifiedStore(root, check=lambda: None)


@pytest.mark.parametrize('name', ['../outside', 'dev/../../outside', '/absolute', 'C:/outside',
                                'C:outside', r'dev\x', 'dev/x:stream', 'dev/x.tmp',
                                'dev/x.tmp.interrupted', 'dev//x', '', None, 'dev. /x'])
def test_array_path_escape_and_temporary_names_refused(world, name):
    root, _, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    with pytest.raises((ValueError, OSError)):
        store.array(name)
    assert not store.maps


def test_resolved_path_escape_refused_without_read(world, monkeypatch):
    root, _, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    original = Path.resolve
    target = root / 'dev/x.npy'

    def resolve(path, *args, **kwargs):
        if path == target:
            return original(root.parent / 'outside.npy')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve)
    with pytest.raises(ValueError, match='escapes'):
        store.array('dev/x')


def test_linked_parent_refused_without_read(world, monkeypatch):
    root, _, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p == root / 'dev' or original(p))
    with pytest.raises(ValueError, match='linked'):
        store.array('dev/x')


@pytest.mark.parametrize('fault', ['missing_file', 'unrecorded', 'missing_hash', 'bad_hash', 'tamper', 'dtype', 'shape', 'bool_shape', 'object', 'temporary'])
def test_requested_array_validation_fails_closed(world, fault):
    root, manifest, _ = world
    path = root / 'dev/x.npy'
    if fault == 'missing_file': path.unlink()
    elif fault == 'unrecorded': del manifest['arrays']['dev/x']
    elif fault == 'missing_hash': del manifest['files']['dev/x.npy']
    elif fault == 'bad_hash': manifest['files']['dev/x.npy'] = 'not-sha256'
    elif fault == 'tamper':
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif fault == 'dtype': manifest['arrays']['dev/x']['dtype'] = '<i4'
    elif fault == 'shape': manifest['arrays']['dev/x']['shape'] = [2, 6]
    elif fault == 'bool_shape': manifest['arrays']['dev/x']['shape'] = [True, 3]
    elif fault == 'object': manifest['arrays']['dev/x']['dtype'] = '|O'
    else: (root / 'dev/x.npy.tmp').write_bytes(b'interrupted')
    save_manifest(root, manifest)
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    with pytest.raises((ValueError, OSError)):
        store.array('dev/x')
    assert store.maps == {} and store.used_evidence()['arrays'] == {}


@pytest.mark.parametrize('where', ['before_array', 'cached_array', 'final_verify'])
def test_metadata_hash_change_refused_even_with_original_size_and_mtime(world, where):
    root, manifest, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        if where != 'before_array': store.array('dev/x')
        metadata = root / 'STORE.json'
        before = metadata.stat()
        # Same-size change and restore mtime: verification must still stream/hash.
        metadata.write_bytes(metadata.read_bytes().replace(b'"version": 1', b'"version": 2'))
        os.utime(metadata, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(ValueError, match='STORE.json changed'):
            store.verify_used() if where == 'final_verify' else store.array('dev/x')
    finally:
        close_maps(store)


def test_after_mapping_tamper_detected_by_final_rehash_with_unchanged_stat(world):
    root, _, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        mapped = store.array('dev/x')
        path = root / 'dev/x.npy'
        before = path.stat()
        with path.open('r+b') as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
            stream.flush()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert mapped.flags.writeable is False
        with pytest.raises(ValueError, match='changed'):
            store.verify_used()
    finally:
        close_maps(store)


def test_cached_array_stat_change_refused_without_rehash(world):
    root, _, _ = world
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        store.array('dev/x')
        path = root / 'dev/x.npy'
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        with pytest.raises(ValueError, match='file changed'):
            store.array('dev/x')
    finally:
        close_maps(store)


def test_unused_corrupt_or_missing_arrays_never_read_or_verified(world):
    root, _, _ = world
    (root / 'other/never_read.npy').unlink()
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        store.array('dev/x')
        store.verify_used()
        assert list(store.used_evidence()['arrays']) == ['dev/x']
    finally:
        close_maps(store)


def test_hash_reads_one_mib_chunks_with_guard_during_stream(world, monkeypatch):
    root, manifest, _ = world
    path = root / 'dev/x.npy'
    values = np.arange(600_000, dtype='<i4')
    np.save(path, values, allow_pickle=False)
    manifest['arrays']['dev/x'] = {'shape': list(values.shape), 'dtype': values.dtype.str}
    manifest['files']['dev/x.npy'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_manifest(root, manifest)
    original = Path.open
    events = []

    class Tracked:
        def __init__(self, file): self.file = file
        def __enter__(self): return self
        def __exit__(self, *args): self.file.close()
        def fileno(self): return self.file.fileno()
        def read(self, size):
            assert size == 1024 * 1024
            events.append(('read', size))
            return self.file.read(size)

    def opened(p, *args, **kwargs):
        file = original(p, *args, **kwargs)
        return Tracked(file) if p == path and args == ('rb',) else file

    def check(): events.append(('check', None))

    store = ReadOnlyVerifiedStore(root, check=check)
    monkeypatch.setattr(Path, 'open', opened)
    try:
        store.array('dev/x')
        reads = [i for i, event in enumerate(events) if event[0] == 'read']
        assert len(reads) == 4  # three data chunks and EOF, all bounded reads
        assert all(events[i - 1][0] == 'check' for i in reads)
    finally:
        close_maps(store)


def test_callback_abort_never_caches_unverified_array(world, monkeypatch):
    root, _, _ = world
    state = {'array_hash': False}
    original = stores._stream_hash

    def hash_file(path, *args, **kwargs):
        state['array_hash'] = path.suffix == '.npy'
        return original(path, *args, **kwargs)

    def check():
        if state['array_hash']:
            raise TimeoutError('synthetic budget')

    store = ReadOnlyVerifiedStore(root, check=check)
    monkeypatch.setattr(stores, '_stream_hash', hash_file)
    with pytest.raises(TimeoutError, match='budget'):
        store.array('dev/x')
    assert not store.maps and not store.used_evidence()['arrays']
    with pytest.raises(RuntimeError, match='guard'):
        ReadOnlyVerifiedStore(root, check=lambda: False)


def test_file_changed_between_hash_and_mapping_refused(world, monkeypatch):
    root, _, _ = world
    path = root / 'dev/x.npy'
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    original_load = np.load

    def changed(*args, **kwargs):
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, 'load', changed)
    with pytest.raises(ValueError, match='between hash and mapping'):
        store.array('dev/x')
    assert not store.maps


def test_stat_fstat_ctime_api_difference_does_not_weaken_identity_check(world, monkeypatch):
    root, _, _ = world
    original = os.fstat

    def different_ctime(fd):
        info = original(fd)
        fields = {k: getattr(info, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
        fields['st_ctime_ns'] += 100
        return SimpleNamespace(**fields)

    monkeypatch.setattr(os, 'fstat', different_ctime)
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    try:
        store.array('dev/x')
        store.verify_used()
        assert store.used_evidence()['arrays']['dev/x']['stat']['inode'] == (root / 'dev/x.npy').stat().st_ino
    finally:
        close_maps(store)


def test_change_during_stream_hash_is_rejected(world, monkeypatch):
    root, _, _ = world
    path = root / 'dev/x.npy'
    store = ReadOnlyVerifiedStore(root, check=lambda: None)
    original_hash = stores._stream_hash

    def changing_hash(p, check, **kwargs):
        if p != path:
            return original_hash(p, check, **kwargs)
        calls = 0

        def change_on_eof():
            nonlocal calls
            calls += 1
            check()
            if calls == 3:
                before = path.stat()
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))

        return original_hash(p, change_on_eof, **kwargs)

    monkeypatch.setattr(stores, '_stream_hash', changing_hash)
    with pytest.raises(ValueError, match='changed during hashing'):
        store.array('dev/x')
    assert not store.maps and not store.used_evidence()['arrays']
