import os

import numpy as np
import pytest

from studies.training_io_20261002.projection import ProjectedNpy


@pytest.mark.parametrize("shape", [(9, 7), (9, 10, 7)])
@pytest.mark.parametrize("fortran", [False, True])
def test_projection_retains_rows_features_dtype_and_source(tmp_path, shape, fortran):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if fortran:
        source = np.asfortranarray(source)
    path = tmp_path / "block.npy"
    np.save(path, source)
    before = path.read_bytes()
    rows, columns = [7, 1, 4], [5, 0, 3]
    with ProjectedNpy(path, rows=rows, columns=columns) as block:
        got = block.get_batch(0, 2)
        np.testing.assert_array_equal(got, source[rows[:2]][..., columns])
        assert got.dtype == source.dtype and not got.flags.writeable
        assert got.nbytes == 2 * block.row_bytes
        with pytest.raises(ValueError):
            got[...] = 0
    assert path.read_bytes() == before


def test_contiguous_selection_is_zero_copy_and_masks_are_owned(tmp_path):
    path = tmp_path / "block.npy"
    np.save(path, np.arange(80, dtype=np.int64).reshape(10, 8))
    rows, columns = np.arange(1, 7), np.arange(2, 5)
    with ProjectedNpy(path, rows=rows, columns=columns) as block:
        rows[:] = 9
        columns[:] = 7
        got = block.get_batch(1, 3)
        assert np.shares_memory(got, block.array)
        np.testing.assert_array_equal(got, np.arange(80).reshape(10, 8)[2:4, 2:5])


def test_byte_guard_bounds_empty_and_closed(tmp_path):
    path = tmp_path / "block.npy"
    np.save(path, np.zeros((10, 10, 8), dtype=np.float32))
    block = ProjectedNpy(path, columns=[1, 3], max_batch_bytes=80)
    assert block.get_batch(0, 1).shape == (1, 10, 2)
    assert block.get_batch(0, 0).shape == (0, 10, 2)
    with pytest.raises(MemoryError):
        block.get_batch(0, 2)
    for bounds in [(-1, 1), (2, 1), (0, 11), (False, 1), (0, 1.5)]:
        with pytest.raises(ValueError):
            block.get_batch(*bounds)
    block.close()
    with pytest.raises(RuntimeError, match="closed"):
        block.get_batch(0, 1)


@pytest.mark.parametrize("bad", [[True, False], [1.5], [-1], [10], [1, 1], [[1]]])
def test_invalid_indices_rejected_and_handle_closed(tmp_path, bad):
    path = tmp_path / "block.npy"
    np.save(path, np.zeros((10, 8)))
    with pytest.raises(ValueError):
        ProjectedNpy(path, rows=bad)
    # Windows replacement must succeed after a rejected constructor.
    os.replace(path, tmp_path / "replaced.npy")


def test_changed_source_rejected(tmp_path):
    path = tmp_path / "block.npy"
    np.save(path, np.zeros((3, 4)))
    with ProjectedNpy(path) as block:
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10_000_000))
        with pytest.raises(RuntimeError, match="metadata changed"):
            block.get_batch(0, 1)


def test_stream_checks_metadata_on_exhaustion_and_explicit_close(tmp_path):
    path = tmp_path / "block.npy"
    np.save(path, np.arange(40).reshape(10, 4))
    with ProjectedNpy(path) as block:
        pieces = list(block.iter_batches(3))
        np.testing.assert_array_equal(np.concatenate(pieces), np.arange(40).reshape(10, 4))
        stream = block.iter_batches(3)
        next(stream)
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10_000_000))
        with pytest.raises(RuntimeError, match="metadata changed"):
            stream.close()


def test_disguised_npz_is_rejected_and_closed(tmp_path):
    path = tmp_path / "block.npy"
    with path.open("wb") as handle:
        np.savez(handle, values=np.zeros((3, 4)))
    with pytest.raises(ValueError):
        ProjectedNpy(path)
    os.replace(path, tmp_path / "replaced")


@pytest.mark.parametrize("kind", ["object", "vector", "npz"])
def test_unsupported_blocks_rejected(tmp_path, kind):
    path = tmp_path / ("block.npz" if kind == "npz" else "block.npy")
    if kind == "npz":
        np.savez(path, block=np.zeros((3, 4)))
    else:
        np.save(path, np.zeros(3) if kind == "vector" else np.array([[object()]]))
    with pytest.raises(ValueError):
        ProjectedNpy(path)
    os.replace(path, tmp_path / "replaced")
