"""Measure complete synthetic cohort projection; never train or read raw data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.out_dir.resolve()
    output.relative_to(root)
    if output == root:
        raise ValueError("Output must be a new workspace child")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "3"
    import psutil
    psutil.Process().cpu_affinity([0, 2, 4])
    import numpy as np
    from .projection import ProjectedNpy

    path = output / "synthetic.npy"
    shape, batch_rows = (32768, 512), 512
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for begin in range(0, shape[0], 1024):
        stop = min(shape[0], begin + 1024)
        values[begin:stop] = np.arange(begin, stop, dtype=np.float32)[:, None] + np.arange(shape[1], dtype=np.float32)[None, :]
    values.flush()
    values._mmap.close()
    del values
    rows = np.arange(0, shape[0], 2, dtype=np.intp)
    columns = np.arange(0, 128, 4, dtype=np.intp)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dense_seconds, bounded_seconds, checksums = [], [], []
    # Each path consumes all selected rows and fields in the same batch order.
    # The dense baseline includes its full source and full selected-row copy.
    for repeat in range(6):
        started = time.perf_counter()
        loaded = np.load(path, allow_pickle=False)
        full_rows = loaded[rows]
        materialized = full_rows[:, columns]
        del full_rows, loaded
        dense_checksum = sum(float(materialized[begin:begin + batch_rows].sum(dtype=np.float64)) for begin in range(0, len(rows), batch_rows))
        dense_elapsed = time.perf_counter() - started
        del materialized
        started = time.perf_counter()
        with ProjectedNpy(path, rows=rows, columns=columns, max_batch_bytes=batch_rows * len(columns) * 4) as projected:
            bounded_checksum = sum(float(batch.sum(dtype=np.float64)) for batch in projected.iter_batches(batch_rows))
        bounded_elapsed = time.perf_counter() - started
        if dense_checksum != bounded_checksum:
            raise RuntimeError("Synthetic cohort checksum differs")
        checksums.append(dense_checksum)
        if repeat:  # warmup is excluded from reported medians
            dense_seconds.append(dense_elapsed)
            bounded_seconds.append(bounded_elapsed)
    # Verify every value independently outside timing, including column order.
    oracle = np.load(path, mmap_mode="r", allow_pickle=False)
    with ProjectedNpy(path, rows=rows, columns=columns) as projected:
        for begin in range(0, len(rows), batch_rows):
            end = min(len(rows), begin + batch_rows)
            np.testing.assert_array_equal(projected.get_batch(begin, end), oracle[rows[begin:end]][:, columns])
    oracle._mmap.close()
    # A separate steady-state workload isolates batch gathering after source
    # setup, as in repeated epochs. Both paths retain their source throughout.
    dense_source = np.load(path, allow_pickle=False)
    steady_dense_seconds, steady_bounded_seconds = [], []
    with ProjectedNpy(path, rows=rows, columns=columns) as projected:
        for repeat in range(6):
            started = time.perf_counter()
            dense_checksum = sum(float(dense_source[rows[begin:begin + batch_rows]][:, columns].sum(dtype=np.float64)) for begin in range(0, len(rows), batch_rows))
            dense_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            bounded_checksum = sum(float(batch.sum(dtype=np.float64)) for batch in projected.iter_batches(batch_rows))
            bounded_elapsed = time.perf_counter() - started
            if dense_checksum != bounded_checksum or dense_checksum != checksums[0]:
                raise RuntimeError("Steady-state batch checksum differs")
            if repeat:
                steady_dense_seconds.append(dense_elapsed)
                steady_bounded_seconds.append(bounded_elapsed)
    del dense_source
    source_unchanged = hashlib.sha256(path.read_bytes()).hexdigest() == source_sha
    if not source_unchanged:
        raise RuntimeError("Synthetic source changed")
    baseline_bytes = shape[0] * shape[1] * 4 + len(rows) * shape[1] * 4 + len(rows) * len(columns) * 4
    projected_bytes = batch_rows * len(columns) * 4
    result = {"synthetic_only": True, "training_performed": False,
              "shape": list(shape), "selected_rows": len(rows), "selected_columns": len(columns),
              "batch_rows": batch_rows, "cpu_affinity": [0, 2, 4], "repeats_after_warmup": 5,
              "source_check_policy": "metadata at iterator boundaries; all-value/source-byte verification outside timing",
              "dense_seconds": dense_seconds, "bounded_seconds": bounded_seconds,
              "dense_median_seconds": statistics.median(dense_seconds),
              "bounded_median_seconds": statistics.median(bounded_seconds),
              "local_speed_ratio": statistics.median(dense_seconds) / statistics.median(bounded_seconds),
              "steady_dense_seconds": steady_dense_seconds,
              "steady_bounded_seconds": steady_bounded_seconds,
              "steady_dense_median_seconds": statistics.median(steady_dense_seconds),
              "steady_bounded_median_seconds": statistics.median(steady_bounded_seconds),
              "steady_batch_speed_ratio": statistics.median(steady_dense_seconds) / statistics.median(steady_bounded_seconds),
              "steady_dense_batch_intermediate_bytes": batch_rows * shape[1] * 4,
              "dense_peak_payload_bytes_upper_bound": baseline_bytes,
              "projected_batch_payload_bytes": projected_bytes,
              "row_index_bytes": rows.nbytes, "column_index_bytes": columns.nbytes,
              "source_npy_bytes": path.stat().st_size, "source_sha256": source_sha,
              "all_values_equal": True, "source_unchanged": source_unchanged,
              "checksum": checksums[0], "rss_measured": False,
              "limits": ["Synthetic repeated cached IO; no cold-cache control",
                         "Payload formula excludes mmap resident pages, interpreter, page cache and index metadata",
                         "Ratio includes dense materialization cost; not a batch-kernel-only comparison",
                         "Steady-state ratio isolates all-column gather then mask vs direct projected gather; source setup excluded",
                         "Not a model training benchmark or whole-study ETA"]}
    (output / "BENCHMARK.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
