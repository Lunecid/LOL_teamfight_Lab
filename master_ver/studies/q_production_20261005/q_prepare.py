"""Prepare the inputs of one q variant: row table, pre-frame XY and the two fitted scope contexts.

Layout under the run root (ARCHITECTURE_KO.md section 6.3):
  Q_ROWS_h{60,90,120}.npz            written by rows.py from the label lock
  raw/store/                         the raw input store (features.py CLI)
  inputs/<variant>/Q_ROWS.npz        filtered variants only (rows.derive_variant), with VARIANT.json
  inputs/<variant>/XY.npy            pre-frame XY [n, 10, 2] float64 aligned to the variant's rows
  inputs/<variant>/{FIT,REFIT_TRAIN}/ ScopeContext.save output (bundle/, CONTEXT.npz, CONTEXT.json)
  inputs/<variant>/PREPARED.json     hashes of everything above, written last
A partial scope directory (no CONTEXT.json) is moved aside, never deleted, and refitted.
Runs at below-normal priority; RAM peaks at a few GiB (the scope's spatial blocks).

CLI: python -B studies/q_production_20261005/q_prepare.py --run D:/LOL_Training_20261004/q_run_1
     --variant main_h90 --release reports/q_production_20261005/SCIENTIFIC_Q_<digest16>.json
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from studies.label_production_20261005 import assemble  # noqa: E402
from studies.q_production_20261005 import features as qf  # noqa: E402
from studies.q_production_20261005 import qbatch, release, rows as qrows, spatial  # noqa: E402

VERSION = "q_prepare_v1_20261005"
SCOPES = ("FIT", "REFIT_TRAIN")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def rows_path(run: Path, variant: str) -> Path:
    if variant not in qrows.VARIANT_HORIZON:
        raise ValueError(f"unknown variant {variant!r}")
    if variant.startswith("exclude_"):
        return run / "inputs" / variant / "Q_ROWS.npz"
    return run / f"Q_ROWS_h{qrows.VARIANT_HORIZON[variant]}.npz"


def query_xy(query_pre: np.ndarray) -> np.ndarray:
    record = spatial.read_query_xy(np.asarray(query_pre, dtype=np.int64))
    if not record.support.all() or not np.isfinite(record.xy).all():
        raise ValueError("a q row has no finite pre-frame XY; the population would change (CCD 106)")
    return np.asarray(record.xy, dtype=np.float64)


def _write_npy(path: Path, array: np.ndarray) -> str:
    pending = path.with_name(path.name + ".pending.npy")
    np.save(pending, array, allow_pickle=False)
    os.replace(pending, path)
    return qrows.sha256_file(path)


def ensure_rows(run: Path, variant: str) -> Path:
    """The variant's row table; filtered variants are derived once from the public h90 table."""

    target = rows_path(run, variant)
    if target.is_file() or not variant.startswith("exclude_"):
        if not target.is_file():
            raise FileNotFoundError(f"{target} is missing; write it with rows.py from the label lock")
        return target
    main = load_npz(run / "Q_ROWS_h90.npz")
    zero = None
    if variant == "exclude_zero_coordinate_h90":
        zero = spatial.zero_coordinate_players(query_xy(main["query_pre"])).any(axis=1)
    rows, counts = qrows.derive_variant(main, variant, zero)
    target.parent.mkdir(parents=True, exist_ok=True)
    identical = counts["excluded"] == 0 and rows.keys() == main.keys() and all(
        rows[name].dtype == main[name].dtype and np.array_equal(rows[name], main[name], equal_nan=rows[name].dtype.kind == "f") for name in main)
    record = {"version": VERSION, "utc": utc(), "variant": variant, "source_rows": "Q_ROWS_h90.npz",
              "source_rows_sha256": qrows.sha256_file(run / "Q_ROWS_h90.npz"), "counts": counts,
              "identical_to_main": bool(identical)}
    record["rows_sha256"] = assemble.write_npz(target, rows)
    (target.parent / "VARIANT.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return target


def prepare(run: Path, variant: str, release_path: Path, scopes: tuple[str, ...] = SCOPES) -> dict[str, Any]:
    run = Path(run)
    payload = release.verify(release_path)
    release.check_loaded(payload)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 - priority is a courtesy, not a requirement
        pass
    base = run / "inputs" / variant
    base.mkdir(parents=True, exist_ok=True)
    source = ensure_rows(run, variant)
    if variant.startswith("exclude_") and json.loads((base / "VARIANT.json").read_text(encoding="utf-8")).get("identical_to_main"):
        return _alias_main(run, variant, source, payload)
    rows = load_npz(source)
    rows_sha = qrows.sha256_file(source)
    xy_path = base / "XY.npy"
    if not xy_path.is_file():
        _write_npy(xy_path, query_xy(rows["query_pre"]))
    xy = np.load(xy_path, allow_pickle=False)
    store_dir = run / "raw" / "store"
    store = qf.RawStore(store_dir)
    provenance = {"variant": variant, "rows_file": source.relative_to(run).as_posix(), "rows_sha256": rows_sha,
                  "xy_file": xy_path.relative_to(run).as_posix(), "xy_sha256": qrows.sha256_file(xy_path),
                  "store_build_sha256": qrows.sha256_file(store_dir / "BUILD.json"), "release_digest": payload["digest"], "prepare_version": VERSION}
    contexts = {}
    for scope in scopes:
        directory = base / scope
        if (directory / "CONTEXT.json").is_file():
            context = qbatch.ScopeContext(store, rows, xy, scope, saved=directory)
            for key in ("rows_sha256", "xy_sha256", "store_build_sha256"):
                if context.record.get(key) != provenance[key]:
                    raise ValueError(f"{directory} was prepared from another {key}")
            contexts[scope] = context.record
            continue
        if directory.exists():
            directory.rename(directory.with_name(f"{scope}.partial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
        started = time.time()
        context = qbatch.ScopeContext(store, rows, xy, scope)
        contexts[scope] = context.save(directory, {**provenance, "fit_seconds": round(time.time() - started, 1)})
    record = {"version": VERSION, "utc": utc(), **provenance, "rows": int(len(rows["role"])),
              "contexts": {scope: {"dir": f"inputs/{variant}/{scope}", "context_json_sha256": qrows.sha256_file(base / scope / "CONTEXT.json"),
                                   "sigma": value["sigma"], "scope_rows": value["scope_rows"]} for scope, value in contexts.items()}}
    pending = base / "PREPARED.json.pending"
    pending.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, base / "PREPARED.json")
    return record


def _alias_main(run: Path, variant: str, source: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """A filtered variant whose rows equal the main table reuses the main inputs (DECISIONS D-Q16)."""

    main = json.loads((run / "inputs" / "main_h90" / "PREPARED.json").read_text(encoding="utf-8"))
    if main.get("release_digest") != payload["digest"]:
        raise RuntimeError("prepare main_h90 under this release first")
    record = {**main, "version": VERSION, "utc": utc(), "variant": variant, "identical_to": "main_h90",
              "variant_rows_file": source.relative_to(run).as_posix(), "variant_rows_sha256": qrows.sha256_file(source),
              "note": "no row excluded; the variant uses the main row table, XY and scope contexts and reuses the main results (D-Q16)"}
    base = run / "inputs" / variant
    pending = base / "PREPARED.json.pending"
    pending.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, base / "PREPARED.json")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--variant", default="main_h90")
    parser.add_argument("--release", required=True)
    parser.add_argument("--scope", choices=(*SCOPES, "both"), default="both")
    args = parser.parse_args(argv)
    scopes = SCOPES if args.scope == "both" else (args.scope,)
    record = prepare(Path(args.run), args.variant, Path(args.release), scopes)
    print(json.dumps({key: record[key] for key in ("variant", "rows", "contexts")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
