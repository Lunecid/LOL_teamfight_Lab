"""Read immutable numeric NPY blocks without materializing each condition."""
from pathlib import Path

import numpy as np


def _indices(values, size, name):
    if values is None:
        return None
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer index")
    if np.any(raw < 0) or np.any(raw >= size):
        raise ValueError(f"{name} index outside block")
    if np.unique(raw).size != raw.size:
        raise ValueError(f"{name} contains duplicate indices")
    result = np.array(raw, dtype=np.intp, copy=True)
    result.flags.writeable = False
    return result


def _contiguous_slice(indices):
    if not len(indices):
        return slice(0, 0)
    if len(indices) == 1 or np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return None


class ProjectedNpy:
    """A read-only row/feature view with an explicit per-batch byte ceiling.

    Source manifests must verify the immutable block hash once upstream.
    Metadata is rechecked before each batch. Views must not outlive ``close``.
    Values, dtype and the requested index order are retained exactly.
    """

    def __init__(self, path, *, rows=None, columns=None, max_batch_bytes=32 << 20):
        self.path = Path(path).resolve(strict=True)
        if self.path.suffix.lower() != ".npy":
            raise ValueError("projection requires an uncompressed NPY block")
        if isinstance(max_batch_bytes, bool) or not isinstance(max_batch_bytes, int) or max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be a positive integer")
        self.array = None
        initial = self.path.stat()
        self.signature = (initial.st_size, initial.st_mtime_ns)
        try:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if not isinstance(self.array, np.memmap) or self.array.ndim not in (2, 3) or self.array.dtype.kind not in "iufc":
                raise ValueError("block must be a numeric 2D or 3D array")
            self.rows = _indices(rows, self.array.shape[0], "rows")
            self.columns = _indices(columns, self.array.shape[-1], "columns")
            self.n_rows = self.array.shape[0] if self.rows is None else len(self.rows)
            self.n_columns = self.array.shape[-1] if self.columns is None else len(self.columns)
            self.column_slice = slice(None) if self.columns is None else _contiguous_slice(self.columns)
            self.row_bytes = int(np.prod(self.array.shape[1:-1], dtype=np.int64)) * self.n_columns * self.array.dtype.itemsize
            self.max_batch_bytes = max_batch_bytes
            self._check_source()
        except BaseException:
            self.close()
            raise

    def _check_source(self):
        if self.array is None:
            raise RuntimeError("projection is closed")
        current = self.path.stat()
        if (current.st_size, current.st_mtime_ns) != self.signature:
            raise RuntimeError("immutable block metadata changed")

    def get_batch(self, begin, end):
        self._check_source()
        return self._project(begin, end)

    def _project(self, begin, end):
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (begin, end)):
            raise ValueError("batch bounds must be integers")
        if not 0 <= begin <= end <= self.n_rows:
            raise ValueError("batch bounds outside selected rows")
        if (end - begin) * self.row_bytes > self.max_batch_bytes:
            raise MemoryError("projected batch exceeds explicit byte ceiling")
        selected_rows = None if self.rows is None else self.rows[begin:end]
        row_slice = slice(begin, end) if selected_rows is None else _contiguous_slice(selected_rows)
        if row_slice is not None and self.column_slice is not None:
            result = self.array[row_slice, ..., self.column_slice]
        else:
            row_ids = np.arange(begin, end, dtype=np.intp) if selected_rows is None else selected_rows
            column_ids = np.arange(self.array.shape[-1], dtype=np.intp) if self.columns is None else self.columns
            # Broadcast index vectors directly to the final projected shape;
            # never gather all features before selecting the requested ones.
            indices = [row_ids] + [np.arange(size, dtype=np.intp) for size in self.array.shape[1:-1]] + [column_ids]
            result = self.array[np.ix_(*indices)]
        result.flags.writeable = False
        return result

    def iter_batches(self, batch_rows):
        """Stream an immutable cohort with boundary metadata validation.

        Exhaust the iterator or explicitly close it. A changed source fails
        at the boundary and invalidates the entire consuming run. Upstream
        source SHA checks before and after fitting remain required.
        """
        self._check_source()
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, (int, np.integer)) or batch_rows <= 0:
            raise ValueError("batch_rows must be a positive integer")
        try:
            for begin in range(0, self.n_rows, batch_rows):
                yield self._project(begin, min(self.n_rows, begin + batch_rows))
        finally:
            self._check_source()

    def close(self):
        if self.array is not None:
            if isinstance(self.array, np.memmap):
                self.array._mmap.close()
            elif hasattr(self.array, "close"):
                self.array.close()
            self.array = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
