"""Read-only, lazy integrity checks for the canonical Store API.

Construction reads only STORE.json. array(name) hashes only the requested .npy,
then maps it read-only; repeated requests reuse that map after metadata/stat
checks. verify_used() streams only previously used arrays again, including when
their stat is unchanged. No directory enumeration or whole-Store.verify occurs.

These are before/after integrity checks, not an OS lock against concurrent writes.
Call verify_used() before accepting results; an already returned memmap cannot
prevent another process from modifying its underlying file between checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

from c_ept.data.store_core import Store


_CHUNK = 1024 * 1024
_MAX_METADATA = 16 * _CHUNK


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _checkpoint(check):
    if check() is False:
        raise RuntimeError('store verification guard refused continuation')


def _signature(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def _regular(path):
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode), f'not a regular store file: {path}')
    return _signature(info)


def _stream_hash(path, check, *, metadata=False):
    """Bounded reads, guard per chunk, and identity checks on path and handle."""
    _checkpoint(check)
    before = _regular(path)
    if metadata:
        _require(before['bytes'] <= _MAX_METADATA, 'STORE.json exceeds metadata size bound')
    digest, blocks, size = hashlib.sha256(), [], 0
    with path.open('rb') as stream:
        handle_before = _signature(os.fstat(stream.fileno()))
        # Windows Python can report creation time for stat().st_ctime and
        # modification time for fstat().st_ctime on the very same file. Compare
        # common fields across APIs, but retain full before/after checks within
        # each API (including ctime) so genuine changes still fail closed.
        common = ('device', 'inode', 'bytes', 'mtime_ns')
        _require(all(handle_before[k] == before[k] for k in common), f'file changed before hashing: {path}')
        while True:
            _checkpoint(check)
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            _require(size <= before['bytes'], f'file grew during hashing: {path}')
            digest.update(chunk)
            if metadata:
                blocks.append(chunk)
        _require(_signature(os.fstat(stream.fileno())) == handle_before, f'file changed during hashing: {path}')
    _require(size == before['bytes'] and _regular(path) == before, f'file changed while hashing: {path}')
    return digest.hexdigest(), before, b''.join(blocks) if metadata else None


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, f'duplicate STORE.json key: {key}')
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError('nonfinite STORE.json value: ' + value)


class ReadOnlyVerifiedStore(Store):
    """A canonical Store subclass with used-file-only evidence and no writers.

    check is a required no-argument guard callback. Exceptions and literal False
    abort verification. root/file/manifest/maps/path/array/done retain the Store
    read API. Mutators, whole-store enumeration/hashing and verify are disabled.
    """

    def __init__(self, root, *, check):
        if not callable(check):
            raise TypeError('check must be callable')
        self._check = check
        _checkpoint(check)
        supplied = Path(root).absolute()
        _require(not supplied.is_symlink() and not self._junction(supplied), 'store root is a link')
        self.root = supplied.resolve(strict=True)
        _require(self.root.is_dir(), 'store root must be an existing directory')
        self._root_identity = self._directory_identity(self.root)
        self.file = self._safe('STORE.json')
        digest, signature, raw = _stream_hash(self.file, check, metadata=True)
        manifest = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique, parse_constant=_nonfinite)
        _require(type(manifest) is dict and manifest.get('complete') is True, 'complete STORE.json required')
        _require(type(manifest.get('arrays')) is dict and type(manifest.get('files')) is dict,
                 'STORE.json arrays/files mappings required')
        _require(type(manifest.get('stages', {})) is dict, 'STORE.json stages must be a mapping')
        # Initialize the canonical read attributes without Store.__init__'s
        # unbound second manifest read or its missing-manifest writer fallback.
        self.manifest = copy.deepcopy(manifest)
        self._bound = manifest
        self.maps = {}
        self._used = {}
        self._metadata = {'path': str(self.file), 'sha256': digest, 'stat': signature}
        self._safe('STORE.json')  # detect temp/link/root replacement during read

    @staticmethod
    def _junction(path):
        return getattr(path, 'is_junction', lambda: False)()

    @staticmethod
    def _directory_identity(path):
        info = path.stat()
        _require(stat.S_ISDIR(info.st_mode), f'store root is no longer a directory: {path}')
        return info.st_dev, info.st_ino

    def _safe(self, relative):
        _require(type(relative) is str and relative and '\\' not in relative and ':' not in relative
                 and '\x00' not in relative, 'unsafe store relative path')
        parts = relative.split('/')
        _require(all(p not in ('', '.', '..') and not p.endswith((' ', '.'))
                     and not p.lower().endswith('.tmp') and '.tmp.' not in p.lower()
                     for p in parts), 'unsafe or temporary store path')
        _require(not Path(relative).is_absolute(), 'absolute store path forbidden')
        _require(not self.root.is_symlink() and not self._junction(self.root)
                 and self.root.resolve(strict=True) == self.root
                 and self._directory_identity(self.root) == self._root_identity, 'store root changed')
        path = self.root
        for part in parts:
            path = path / part
            _require(not path.is_symlink() and not self._junction(path), f'linked store artifact: {path}')
        _require(path.resolve().is_relative_to(self.root), f'store path escapes root: {path}')
        temporary = path.with_name(path.name + '.tmp')
        _require(not temporary.exists() and not temporary.is_symlink(), f'temporary store artifact present: {temporary}')
        return path

    def _metadata_check(self):
        path = self._safe('STORE.json')
        digest, signature, _ = _stream_hash(path, self._check, metadata=True)
        _require(digest == self._metadata['sha256'] and signature == self._metadata['stat'],
                 'STORE.json changed after construction')
        self._safe('STORE.json')

    def path(self, name):
        _require(type(name) is str and name, 'array name must be a nonempty string')
        _require(all(part not in ('', '.', '..') for part in name.split('/')), 'unsafe array name')
        return self._safe(name + '.npy')

    def _spec(self, name):
        path = self.path(name)
        _require(name in self._bound['arrays'], f'unrecorded array: {name}')
        record = self._bound['arrays'][name]
        _require(type(record) is dict and type(record.get('shape')) is list
                 and all(type(n) is int and n >= 0 for n in record['shape']), f'invalid array shape metadata: {name}')
        _require(type(record.get('dtype')) is str, f'invalid array dtype metadata: {name}')
        dtype = np.dtype(record['dtype'])
        _require(not dtype.hasobject, f'object array forbidden: {name}')
        expected = self._bound['files'].get(name + '.npy')
        _require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None,
                 f'missing or invalid array SHA256: {name}')
        return path, tuple(record['shape']), dtype, expected

    def array(self, name):
        self._metadata_check()
        path, shape, dtype, expected = self._spec(name)
        if name in self._used:
            _require(_regular(path) == self._used[name]['stat'], f'used array file changed: {name}')
            mapped = self.maps[name]
            _require(isinstance(mapped, np.memmap) and not mapped.flags.writeable
                     and mapped.shape == shape and mapped.dtype == dtype, f'cached array changed: {name}')
            return mapped
        digest, signature, _ = _stream_hash(path, self._check)
        _require(digest == expected, f'array SHA256 mismatch: {name}')
        self._safe(name + '.npy')
        mapped = None
        try:
            mapped = np.load(path, mmap_mode='r', allow_pickle=False)
            _require(isinstance(mapped, np.memmap) and not mapped.flags.writeable,
                     f'read-only memmap required: {name}')
            _require(mapped.shape == shape and mapped.dtype == dtype, f'array shape/dtype mismatch: {name}')
            _require(_regular(path) == signature, f'array changed between hash and mapping: {name}')
            self._safe(name + '.npy')
            self._metadata_check()
        except BaseException:
            if isinstance(mapped, np.memmap):
                mapped._mmap.close()
            raise
        self.maps[name] = mapped
        self._used[name] = {'path': str(path), 'sha256': expected, 'shape': list(shape),
                            'dtype': dtype.str, 'stat': signature}
        return mapped

    def verify_used(self):
        """Rehash every used file even if its stat is unchanged; return no payload."""
        self._metadata_check()
        for name, record in self._used.items():
            path = self.path(name)
            digest, signature, _ = _stream_hash(path, self._check)
            _require(digest == record['sha256'] and signature == record['stat'],
                     f'used array changed at final verification: {name}')
            self._safe(name + '.npy')
        self._metadata_check()

    def used_evidence(self):
        """Snapshot of verified identities, not an implicit final verification.

        Includes only used array paths, hashes, shape/dtype and stat metadata.
        The caller must run verify_used() before publishing accepted results.
        """
        return copy.deepcopy({'root': str(self.root), 'manifest': self._metadata,
                              'arrays': dict(sorted(self._used.items()))})

    def _read_only(self, *args, **kwargs):
        raise TypeError('read-only evaluation store; use array() and verify_used() only')

    save = allocate = _rows_view = put_rows = put_at = mark_done = set_stage = finalize = _read_only
    # Explicitly prevent accidental calls to inherited whole-store traversal.
    stored_files = hashes = verify = _read_only
