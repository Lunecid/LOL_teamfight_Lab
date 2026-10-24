"""File-by-file comparison of two per-match caches (meta.json, events.json, npz).  Pure numpy/json; no old-repo import.

meta.json: byte equality plus the dotted paths of every differing key/value (parsed JSON).
events.json: exact byte equality (parsed equality and event counts reported when bytes differ).
npz: array-name sets; per array dtype, shape, exact value equality (NaN-aware), byte equality, and max |a-b| when the
values differ.  Container byte equality (whole .npz file) is reported as extra information only: numpy writes fixed zip
member timestamps, so equal arrays usually give equal files, but a different zlib could change the compressed bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAX_PATHS = 50


def json_diff_paths(a, b, prefix: str = "", out: list | None = None) -> list[str]:
    out = [] if out is None else out
    if len(out) >= MAX_PATHS:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a.keys()) + [k for k in b.keys() if k not in a]:
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                json_diff_paths(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{prefix}[len {len(a)}!={len(b)}]")
        for i, (x, y) in enumerate(zip(a, b)):
            json_diff_paths(x, y, f"{prefix}[{i}]", out)
    elif type(a) is not type(b) or a != b:
        out.append(prefix or "<root>")
    return out


def _arr_compare(x: np.ndarray, y: np.ndarray) -> dict:
    r = dict(dtype=[str(x.dtype), str(y.dtype)], shape=[list(x.shape), list(y.shape)],
             dtype_equal=x.dtype == y.dtype, shape_equal=x.shape == y.shape)
    if r["shape_equal"]:
        try:
            nan_ok = np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating)
            r["values_equal"] = bool(np.array_equal(x, y, equal_nan=nan_ok))
        except TypeError:
            r["values_equal"] = bool(np.array_equal(x, y))
        r["bytes_equal"] = bool(r["dtype_equal"] and x.tobytes() == y.tobytes())
        if not r["values_equal"] and x.size:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            r["max_abs_diff"] = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
            r["n_diff"] = int(np.count_nonzero(~((x == y) | (np.isnan(x.astype(np.float64)) & np.isnan(y.astype(np.float64))))))
    else:
        r["values_equal"] = r["bytes_equal"] = False
    r["equal"] = bool(r["dtype_equal"] and r["shape_equal"] and r["values_equal"] and r["bytes_equal"])
    return r


def compare_npz(path_a, path_b) -> dict:
    with np.load(path_a, allow_pickle=False) as za, np.load(path_b, allow_pickle=False) as zb:
        names_a, names_b = list(za.files), list(zb.files)
        arrays = {k: _arr_compare(za[k], zb[k]) for k in names_a if k in names_b}
    res = dict(names=[names_a, names_b], names_equal=sorted(names_a) == sorted(names_b),
               order_equal=names_a == names_b, arrays=arrays)
    res["equal"] = bool(res["names_equal"] and all(v["equal"] for v in arrays.values()))
    return res


def compare_events(path_a, path_b) -> dict:
    ba, bb = Path(path_a).read_bytes(), Path(path_b).read_bytes()
    res = dict(bytes_equal=ba == bb, bytes=[len(ba), len(bb)])
    if not res["bytes_equal"]:
        ea, eb = json.loads(ba), json.loads(bb)
        res.update(parsed_equal=ea == eb, n_events=[len(ea), len(eb)], diff_paths=json_diff_paths(ea, eb)[:10])
    res["equal"] = res["bytes_equal"]
    return res


def compare_meta(path_a, path_b) -> dict:
    ba, bb = Path(path_a).read_bytes(), Path(path_b).read_bytes()
    ma, mb = json.loads(ba), json.loads(bb)
    diffs = json_diff_paths(ma, mb)
    top = sorted({p.split(".")[0].split("[")[0] for p in diffs})
    return dict(bytes_equal=ba == bb, keys_equal=list(ma.keys()) == list(mb.keys()), differing_paths=diffs,
                differing_top_level_keys=top, equal=not diffs and ba == bb)


def compare_match(match_id: str, dir_a, dir_b) -> dict:
    """dir_a = rebuilt cache, dir_b = reference cache."""
    a, b = Path(dir_a), Path(dir_b)
    out = dict(match_id=match_id)
    names = dict(meta=f"{match_id}.meta.json", events=f"{match_id}.events.json", npz=f"{match_id}.npz")
    missing = [f"{side}:{k}" for k, n in names.items() for side, d in (("rebuilt", a), ("reference", b)) if not (d / n).exists()]
    if missing:
        out.update(missing=missing, equal=False)
        return out
    out["meta"] = compare_meta(a / names["meta"], b / names["meta"])
    out["events"] = compare_events(a / names["events"], b / names["events"])
    out["npz"] = compare_npz(a / names["npz"], b / names["npz"])
    out["npz"]["file_bytes_equal"] = (a / names["npz"]).read_bytes() == (b / names["npz"]).read_bytes()
    out["equal"] = bool(out["meta"]["equal"] and out["events"]["equal"] and out["npz"]["equal"])
    return out
