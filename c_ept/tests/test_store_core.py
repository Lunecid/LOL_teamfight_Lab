"""Tests of the vendored r2 store format layer (c_ept.store_core). Every store lives under pytest's tmp_path."""
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from c_ept.data import store_core


def _manifest(root):
    return json.loads((root / 'STORE.json').read_text(encoding='utf-8'))


def _toy():
    """7 matches of 2, 1, 3, 2, 2, 1, 3 rows: consecutive match ids and a float table with one distinct value per cell."""
    ids = np.repeat(np.arange(7), [2, 1, 3, 2, 2, 1, 3])
    return ids, np.arange(len(ids) * 3, dtype='f8').reshape(-1, 3)


def _fill(root, ids, data, block_matches, reverse=False):
    """What a runner does with a store: allocate once, then write every block that is not marked done, and mark it."""
    s = store_core.Store(root)
    s.allocate('t/x', data.shape, data.dtype)
    plan = list(enumerate(store_core.block_plan(ids, block_matches)))
    for b, (lo, hi) in (reversed(plan) if reverse else plan):
        if b not in s.done('fill'):
            s.put_rows(lo, {'t/x': data[lo:hi]})
            s.mark_done('fill', b, ['t/x'])
    return s


# ---- the five tests of the task brief (statements split over lines, otherwise as written) ----

def test_allocate_reopen_checks_shape(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('a/x', (4, 2), 'f4')
    with pytest.raises(ValueError):
        store_core.Store(tmp_path).allocate('a/x', (5, 2), 'f4')


def test_put_rows_refuses_cast(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (4,), 'i4')
    with pytest.raises(AssertionError):
        s.put_rows(0, {'x': np.zeros(2, 'i8')})


def test_verify_detects_change(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (4,), 'i4')
    s.put_rows(0, {'x': np.arange(4, dtype='i4')})
    s.finalize()
    a = np.load(s.path('x'), mmap_mode='r+')
    a[0] = 9
    a.flush()
    del a
    with pytest.raises(ValueError, match='x.npy'):
        store_core.Store(tmp_path).verify()


def test_block_plan_whole_matches():
    assert store_core.block_plan(np.array([0, 0, 1, 2, 2, 2, 3]), 2) == [(0, 3), (3, 7)]


def test_require_free_bytes(tmp_path):
    with pytest.raises(OSError, match='GiB'):
        store_core.require_free_bytes(tmp_path, 1 << 62)


# ---- finalize and verify ----

def test_finalize_then_verify_round_trip(tmp_path):
    root = tmp_path / 'run' / 's1'                      # the root folder does not exist yet: the store creates it
    s = store_core.Store(root)
    s.allocate('x', (4,), 'i4')
    s.allocate('g/y', (3, 2), 'f8')
    s.put_rows(0, {'x': np.arange(4, dtype='i4'), 'g/y': np.arange(6, dtype='f8').reshape(3, 2)})
    store_core.write_atomic(root / 'g' / 'note.json', '{"k": 1}')       # a file that is not an array is covered as well
    s.finalize()
    s.verify()                                          # the object that finalized
    store_core.Store(root).verify()                     # a fresh object that reads STORE.json back from disk
    m = _manifest(root)
    assert m['complete'] is True
    assert m['files'] == {k: store_core.sha(root / k) for k in ('g/note.json', 'g/y.npy', 'x.npy')}     # STORE.json is not in it
    assert m['arrays'] == {'x': {'shape': [4], 'dtype': '<i4'}, 'g/y': {'shape': [3, 2], 'dtype': '<f8'}}
    assert store_core.Store(root).array('g/y').tolist() == [[0, 1], [2, 3], [4, 5]]


def test_verify_fails_unless_finalized_and_when_a_file_is_added_or_removed(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (4,), 'i4')
    with pytest.raises(ValueError, match='STORE.json'):      # nothing finalized yet: no recorded hashes to compare with
        store_core.Store(tmp_path).verify()
    with pytest.raises(ValueError, match='STORE.json'):      # even when there is no file to disagree about, an unfinalized store fails
        store_core.Store(tmp_path / 'empty').verify()
    s.finalize()
    store_core.Store(tmp_path).verify()
    (tmp_path / 'extra.bin').write_bytes(b'1')
    with pytest.raises(ValueError, match='extra.bin'):
        store_core.Store(tmp_path).verify()
    (tmp_path / 'extra.bin').unlink()
    store_core.Store(tmp_path).verify()                      # back to the recorded state
    s.path('x').unlink()
    with pytest.raises(ValueError, match='x.npy'):
        store_core.Store(tmp_path).verify()


def test_stored_files_skips_the_manifest_and_temporary_files(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('a/x', (2,), 'i4')                            # also writes STORE.json
    (tmp_path / 'a' / 'half.npz.tmp').write_bytes(b'cut off')       # what an interrupted write_atomic leaves behind
    (tmp_path / 'a' / 'keep.npz').write_bytes(b'ok')
    assert [p.relative_to(tmp_path).as_posix() for p in s.stored_files()] == ['a/keep.npz', 'a/x.npy']


def test_hashes_accepts_a_generator(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('a/x', (2,), 'i4')
    s.allocate('y', (3,), 'f8')
    files = s.stored_files()
    want = s.hashes(files)
    got = s.hashes(p for p in files)                    # a generator is read once: it must not come back as an empty dict
    assert sorted(want) == ['a/x.npy', 'y.npy'] and all(len(v) == 64 for v in want.values())
    assert got == want


# ---- resumability: block marks and stage information ----

def test_done_and_mark_done_persist_across_reopening(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (6,), 'i4')
    assert s.done('extract') == set()
    s.put_rows(0, {'x': np.arange(2, dtype='i4')})
    s.mark_done('extract', 0, ['x'])
    s.put_rows(4, {'x': np.arange(4, 6, dtype='i4')})
    s.mark_done('extract', 2, ['x'])
    s.mark_done('extract', 2, ['x'])                    # a block marked twice is recorded once
    assert s.done('extract') == {0, 2}
    t = store_core.Store(tmp_path)                      # a restarted run reads the marks back from STORE.json
    assert t.done('extract') == {0, 2}
    assert t.done('assemble') == set()                  # stages are kept apart
    t.mark_done('extract', 1, [])                       # marks added after the restart are kept as well
    assert store_core.Store(tmp_path).done('extract') == {0, 1, 2}
    assert sorted(_manifest(tmp_path)['stages']['extract']['blocks']) == [0, 1, 2]


def test_mark_done_syncs_files_before_it_records_and_accepts_paths(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (2,), 'i4')
    block = tmp_path / 'blocks' / 'b0000.npz'
    with pytest.raises(OSError):                        # the block file is not there yet: nothing may be recorded
        s.mark_done('extract', 0, ['x', block])
    assert s.done('extract') == set()
    assert store_core.Store(tmp_path).done('extract') == set()
    block.parent.mkdir()
    store_core.write_atomic(block, b'payload')
    s.mark_done('extract', 0, ['x', block])             # array names and block files may be mixed
    assert store_core.Store(tmp_path).done('extract') == {0}


def test_set_stage_keeps_the_info_and_the_block_list(tmp_path):
    root = tmp_path / 'new' / 'root'                    # saving the manifest creates a root folder that is not there yet
    s = store_core.Store(root)
    s.set_stage('assemble', rows=12, status='running')
    s.mark_done('assemble', 0, [])
    s.set_stage('assemble', status='finished')          # updates what is given, keeps the rest and the marks
    assert store_core.Store(root).done('assemble') == {0}
    assert _manifest(root)['stages']['assemble'] == {'blocks': [0], 'rows': 12, 'status': 'finished'}


def test_file_hashes_do_not_depend_on_block_size_or_block_order(tmp_path):
    ids, data = _toy()
    stores = [_fill(tmp_path / 'a', ids, data, 1), _fill(tmp_path / 'b', ids, data, 3, reverse=True), _fill(tmp_path / 'c', ids, data, 99)]
    for s in stores:
        s.finalize()
    assert stores[0].manifest['files'] == stores[1].manifest['files'] == stores[2].manifest['files']
    assert list(stores[0].manifest['files']) == ['t/x.npy']
    assert [len(s.done('fill')) for s in stores] == [7, 3, 1]
    assert np.array_equal(stores[1].array('t/x'), data)


def test_resume_after_an_interruption_gives_the_same_files(tmp_path):
    ids, data = _toy()
    whole = _fill(tmp_path / 'whole', ids, data, 2)
    whole.finalize()
    plan = store_core.block_plan(ids, 2)                # four blocks
    s = store_core.Store(tmp_path / 'cut')
    s.allocate('t/x', data.shape, data.dtype)
    for b in (0, 1):                                    # two blocks written and marked ...
        lo, hi = plan[b]
        s.put_rows(lo, {'t/x': data[lo:hi]})
        s.mark_done('fill', b, ['t/x'])
    lo = plan[2][0]
    s.put_rows(lo, {'t/x': -data[lo:lo + 2]})           # ... the third only partly written and not marked: the process dies here
    del s
    resumed = _fill(tmp_path / 'cut', ids, data, 2)     # a new object on the same folder skips the marked blocks, redoes the rest
    assert resumed.done('fill') == {0, 1, 2, 3}
    resumed.finalize()
    assert resumed.manifest['files'] == whole.manifest['files']
    resumed.verify()


# ---- arrays ----

def test_allocate_reopen_keeps_the_data_and_checks_the_dtype(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (3,), 'i4')
    s.put_rows(0, {'x': np.array([7, 8, 9], dtype='i4')})
    again = store_core.Store(tmp_path).allocate('x', (3,), 'i4')        # the same request is accepted and clears nothing
    assert again.tolist() == [7, 8, 9]
    again[0] = 5                                        # and it is a writable view
    again.flush()
    del again
    assert store_core.Store(tmp_path).array('x').tolist() == [5, 8, 9]
    with pytest.raises(ValueError, match='x: stored'):
        store_core.Store(tmp_path).allocate('x', (3,), 'i8')


def test_array_is_a_read_only_view(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (2,), 'i4')
    with pytest.raises(ValueError):
        s.array('x')[0] = 1


def test_put_at_writes_scattered_rows_and_refuses_casts(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (5, 2), 'f4')
    s.put_at(np.array([4, 1]), {'x': np.array([[1, 2], [3, 4]], dtype='f4')})
    expected = [[0, 0], [3, 4], [0, 0], [0, 0], [1, 2]]
    assert s.array('x').tolist() == expected
    with pytest.raises(AssertionError):                 # the same no-cast rule as put_rows
        s.put_at(np.array([0]), {'x': np.zeros((1, 2), 'f8')})
    with pytest.raises(AssertionError):                 # and the row shape must agree as well
        s.put_rows(0, {'x': np.zeros((1, 3), 'f4')})
    assert s.array('x').tolist() == expected            # a refused write changed nothing


def test_no_cast_error_names_the_array_and_both_dtypes(tmp_path):
    s = store_core.Store(tmp_path)
    s.allocate('x', (4,), 'i4')
    with pytest.raises(AssertionError) as err:
        s.put_rows(0, {'x': np.zeros(2, 'i8')})
    assert err.value.args == (('x', np.dtype('i4'), np.dtype('i8'), (4,), (2,)),)      # what `assert cond, msg` would have carried


def test_the_no_cast_check_survives_python_dash_O(tmp_path):
    # `assert` statements are removed under `python -O`; the check must still refuse the cast in such a process
    code = textwrap.dedent('''
        import sys
        import numpy as np
        from c_ept import store_core
        if __debug__:
            sys.exit(5)                  # -O is not in effect: this run would prove nothing
        s = store_core.Store(sys.argv[1])
        s.allocate('x', (4,), 'i4')
        try:
            s.put_rows(0, {'x': np.zeros(2, 'i8')})
        except AssertionError:
            sys.exit(0)
        sys.exit(3)                      # the int64 rows were written into the int32 array
    ''')
    hub = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, '-O', '-B', '-c', code, str(tmp_path)], cwd=hub, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr[-300:])


# ---- files, hashes, plans, disk space ----

def test_write_atomic_replaces_the_target_and_leaves_no_temporary_file(tmp_path):
    target = tmp_path / 'f.bin'
    text = 'new caf' + chr(0xE9)                          # e-acute: two bytes in UTF-8
    store_core.write_atomic(target, b'old')
    store_core.write_atomic(target, text)               # text is written as UTF-8 and replaces the old content
    assert target.read_bytes() == b'new caf\xc3\xa9'
    assert [p.name for p in tmp_path.iterdir()] == ['f.bin']


def test_write_atomic_keeps_the_old_file_when_the_write_fails(tmp_path, monkeypatch):
    target = tmp_path / 'f.bin'
    store_core.write_atomic(target, b'old')

    def crash(fd):
        raise OSError('disk gone')

    monkeypatch.setattr(store_core.os, 'fsync', crash)
    with pytest.raises(OSError, match='disk gone'):
        store_core.write_atomic(target, b'new')
    monkeypatch.undo()
    assert target.read_bytes() == b'old'                # the target was never touched: a reader sees the old file
    assert sorted(p.name for p in tmp_path.iterdir()) == ['f.bin', 'f.bin.tmp']      # the cut-off write is left in a *.tmp file


def test_sha_is_the_sha256_hex_digest(tmp_path):
    p = tmp_path / 'f.bin'
    p.write_bytes(b'abc')
    assert store_core.sha(p) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert store_core.sha(str(p)) == hashlib.sha256(b'abc').hexdigest()      # a str path works as well


def test_block_plan_tiles_the_rows_and_never_splits_a_match():
    rng = np.random.default_rng(20261020)
    for _ in range(60):
        sizes = rng.integers(1, 5, size=int(rng.integers(1, 25)))
        ids = np.repeat(rng.permutation(1000)[:len(sizes)], sizes)       # arbitrary distinct ids, each match one consecutive run
        k = int(rng.integers(1, 9))
        plan = store_core.block_plan(ids, k)
        assert plan[0][0] == 0 and plan[-1][1] == len(ids)
        assert all(hi == lo for (_, hi), (lo, _) in zip(plan, plan[1:]))        # no gap and no overlap
        counts = [len(set(ids[lo:hi].tolist())) for lo, hi in plan]
        assert counts[:-1] == [k] * (len(plan) - 1) and 1 <= counts[-1] <= k    # k matches per block, the last block takes the rest
        assert len(plan) == -(-len(sizes) // k)
        assert not any(set(ids[a:b].tolist()) & set(ids[c:d].tolist()) for (a, b), (c, d) in zip(plan, plan[1:]))    # no match twice


def test_block_plan_handles_no_rows_and_string_keys():
    assert store_core.block_plan(np.array([], dtype='i8'), 3) == []
    assert store_core.block_plan(['m1', 'm1', 'm2', 'm3', 'm3'], 1) == [(0, 2), (2, 3), (3, 5)]
    assert store_core.block_plan(['m1', 'm1', 'm2', 'm3', 'm3'], 7) == [(0, 5)]


def test_block_plan_rejects_bad_block_sizes():
    ids = np.array([0, 0, 1, 2, 2, 2, 3])
    for bad in (0, -1, -2):                             # below 1: used to give a slice error (0) or a plan that drops rows (negative)
        with pytest.raises(ValueError, match='at least 1'):
            store_core.block_plan(ids, bad)
    for bad in (True, False, np.True_, np.False_, 2.5, 2.0, '2', None):         # not an integer; a bool is not a size either
        with pytest.raises(TypeError):
            store_core.block_plan(ids, bad)
    with pytest.raises(ValueError, match='at least 1'):         # the size is checked before the rows are looked at
        store_core.block_plan(np.array([], dtype='i8'), 0)
    assert store_core.block_plan(ids, 1) == [(0, 2), (2, 3), (3, 6), (6, 7)]
    assert store_core.block_plan(ids, np.int64(2)) == [(0, 3), (3, 7)]          # numpy integers are sizes


def test_require_free_bytes_checks_the_nearest_existing_folder_and_scales_with_the_margin(tmp_path):
    ahead = tmp_path / 'not' / 'made' / 'yet'
    store_core.require_free_bytes(ahead, 1024)          # room enough: returns quietly although the path does not exist yet
    with pytest.raises(OSError, match=r'need .* GiB, free .* GiB at ') as err:
        store_core.require_free_bytes(ahead, 1 << 62)
    assert str(err.value).endswith(str(tmp_path))       # the nearest existing parent was asked
    with pytest.raises(OSError, match='GiB'):
        store_core.require_free_bytes(tmp_path, 1, margin=float(1 << 62))       # the margin multiplies the need
    assert not (tmp_path / 'not').exists()              # asking creates nothing


# ---- scope ----

def test_no_r2_domain_code_was_vendored():
    methods = ('write_meta', 'put_items', 'meta', 'y', 'adapter', 'block_coo', 'item_coo', 'observed', 'records', 'read_x', 'view',
               'gather')
    names = ('COO_KEYS', 'RECORD_KEYS', 'OBSERVED_KEYS', 'X_KEYS', 'StoreView', 'swap')
    assert [n for n in methods if hasattr(store_core.Store, n)] == []
    assert [n for n in names if hasattr(store_core, n)] == []
