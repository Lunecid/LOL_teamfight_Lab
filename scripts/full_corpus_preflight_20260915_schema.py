"""Record runtime schemas (key names, dtypes, shapes) for one sample of each input kind.

Only structure is recorded; no outcome/winner values are printed or stored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
RAW = {"kr_2026_current": Path("D:/LOL_Project/data/raw/2026_current/kr"),
       "na1_2026_current": Path("D:/LOL_Project/data/raw/2026_current/na1")}


def shape_of(x, depth=0):
    if isinstance(x, dict):
        if depth >= 2:
            return {"type": "dict", "n_keys": len(x)}
        return {"type": "dict", "keys": {k: shape_of(v, depth + 1) for k, v in list(x.items())[:60]}}
    if isinstance(x, list):
        return {"type": "list", "len": len(x), "item": shape_of(x[0], depth + 1) if x and depth < 2 else None}
    return {"type": type(x).__name__}


def main():
    ws = Path(os.environ["TEAMFIGHT_WORKSPACE"])
    out = ws / "outputs/full_corpus_preflight_20260915/runtime_schemas.json"
    stem = "KR_7715477686"
    res = {"sample_cache_stem": stem}
    with np.load(CACHE / f"{stem}.npz", allow_pickle=False) as z:
        res["npz"] = {k: {"dtype": str(z[k].dtype), "shape": list(z[k].shape)} for k in z.files}
    meta = json.loads((CACHE / f"{stem}.meta.json").read_text(encoding="utf-8"))
    res["meta_json_keys"] = sorted(meta)
    ev = json.loads((CACHE / f"{stem}.events.json").read_text(encoding="utf-8"))
    res["events_json"] = shape_of(ev)
    for name, root in RAW.items():
        det = next(os.scandir(root / "detail")).name
        d = json.loads((root / "detail" / det).read_text(encoding="utf-8"))
        t = json.loads((root / "timeline" / det).read_text(encoding="utf-8"))
        res[name] = {
            "detail_top_keys": sorted(d),
            "detail_metadata_keys": sorted(d.get("metadata", {})),
            "detail_info_keys": sorted(d.get("info", {})),
            "timeline_top_keys": sorted(t),
            "timeline_info_keys": sorted(t.get("info", {})),
            "timeline_frame_keys": sorted(t.get("info", {}).get("frames", [{}])[0]),
            "timeline_event_type_names_first_frames": sorted({e.get("type") for fr in t.get("info", {}).get("frames", [])[:40] for e in fr.get("events", [])}),
        }
    out.write_bytes(json.dumps(res, indent=2, sort_keys=True).encode("utf-8"))


if __name__ == "__main__":
    main()
