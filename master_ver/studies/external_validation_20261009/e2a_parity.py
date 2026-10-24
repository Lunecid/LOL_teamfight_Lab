"""E2a parity gate: compare the parity_15_16 run of the adapted pipeline with the accepted 2025 artifacts, per match.

README
  For each of the 500 parity matches (E2a_PARITY_SUBSET.json) and each artifact, the match is located by canonical match
  key in both runs and every stored value is compared with exact equality: arrays bitwise (dtype, shape and bytes),
  JSON records by canonical text.  Global positions necessarily differ in a 500-match subset (shard index, local index,
  first_* offsets, match_index, query ids, item offsets); those fields are listed as 'position' and are compared
  through the local position they encode instead (e.g. query_id - first_query, offsets relative to the query's first
  element).  The role fields are the only fields allowed to differ in value, and only to the documented EXTERNAL codes
  ('role_only'); the ledger key 'dataset' is an E2a addition ('provenance').
  Artifacts: cache source audit ledger, observable bank (every BANK.npz match field incl. components, events, xy,
  node_frame, query arrays; LAYOUT; MATCH_LEDGER), V preparation (MATCH_TABLE, V_ROWS, CONTEXT_MEMBERSHIP), item-node
  bank (MATCH_TABLE, QUERY_TABLE, NODE_OFFSETS, NODE_SUPPORT, NODE_REASON_MASK, NOMINAL_TABLE, ITEM_IDS/COUNTS,
  EVENT_SOURCE_INDEX_*), projection (all feature blocks, QUERY_TABLE, MATCH_QUERIES, table copies, FIELD_MANIFEST),
  shared history (every record's query_ms, local query position, nonitem and item payloads).

Output: reports/external_validation_20261009/E2a_PARITY_REPORT.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

MAX_EXAMPLES = 5


class FieldStats:
    """Per-field counters with the first differing examples."""

    def __init__(self):
        self.fields = defaultdict(lambda: {"compared": 0, "equal": 0, "examples": []})

    def add(self, field: str, ok: bool, example=None) -> None:
        row = self.fields[field]
        row["compared"] += 1
        row["equal"] += int(bool(ok))
        if not ok and len(row["examples"]) < MAX_EXAMPLES:
            row["examples"].append(example)

    def result(self) -> dict:
        out = {}
        for name in sorted(self.fields):
            row = self.fields[name]
            out[name] = {"compared": row["compared"], "equal": row["equal"],
                         "exact": row["compared"] > 0 and row["equal"] == row["compared"],
                         **({"examples": row["examples"]} if row["examples"] else {})}
        return out

    def exact(self) -> bool:
        return all(row["compared"] > 0 and row["equal"] == row["compared"] for row in self.fields.values())


def _array_equal(a, b) -> tuple[bool, dict | None]:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, {"dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)]}
    if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes():
        return True, None
    detail = {"dtype": str(a.dtype), "shape": list(a.shape)}
    try:
        if a.dtype.names is None:
            diff = np.flatnonzero(np.ascontiguousarray(a).reshape(-1).view(np.uint8) != np.ascontiguousarray(b).reshape(-1).view(np.uint8))
            detail["first_byte_offset"] = int(diff[0]) if diff.size else None
            flat_a, flat_b = a.reshape(-1), b.reshape(-1)
            idx = np.flatnonzero(~((flat_a == flat_b) | ((flat_a != flat_a) & (flat_b != flat_b)))) if a.dtype.kind in "fc" else np.flatnonzero(flat_a != flat_b)
            if idx.size:
                detail["first_index"] = int(idx[0])
                detail["values"] = [repr(flat_a[idx[0]].item()), repr(flat_b[idx[0]].item())]
                detail["n_values_differ"] = int(idx.size)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        detail["detail_error"] = repr(exc)
    return False, detail


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)


def _subset() -> tuple[list[dict], dict]:
    subset = C.read_json(C.PARITY_SUBSET)
    rows = sorted(subset["matches"], key=lambda r: r["original_index"])
    return rows, {r["canonical_match_key"]: r for r in rows}


# ------------------------------------------------------------------------------------------------------- audit
def compare_audit(keys: set) -> dict:
    stats = FieldStats()
    mine = {}
    with (C.stage_dir(C.DATASET_PARITY, "audit") / "row_ledger.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mine[row["match_key_sha256"]] = row
    ref = {}
    marker = '"match_key_sha256": "'
    with C.ACCEPTED["audit_ledger"].open(encoding="utf-8") as handle:
        for line in handle:
            at = line.find(marker)
            if at >= 0 and line[at + len(marker):at + len(marker) + 64] not in keys:
                continue
            row = json.loads(line)
            if row["match_key_sha256"] in keys:
                ref[row["match_key_sha256"]] = row
    for key in sorted(keys):
        a, b = mine.get(key), ref.get(key)
        stats.add("present_in_both", a is not None and b is not None, {"key": key})
        if a is None or b is None:
            continue
        stats.add("key_set", set(a) == set(b), {"key": key, "only_mine": sorted(set(a) - set(b)), "only_ref": sorted(set(b) - set(a))})
        for field in sorted(set(a) & set(b)):
            if field == "row_no":
                continue  # position in the respective audit order
            if field == "role":
                stats.add("role[role_only: EXTERNAL vs TEST]", a[field] == C.EXTERNAL_ROLE and b[field] == "TEST", {"key": key, "values": [a[field], b[field]]})
                continue
            stats.add(field, _canon(a[field]) == _canon(b[field]), {"key": key})
    return {"fields": stats.result(), "exact": stats.exact(), "n_mine": len(mine), "n_reference_found": len(ref),
            "position_fields": ["row_no"], "role_only_fields": {"role": [C.EXTERNAL_ROLE, "TEST"]}}


# ------------------------------------------------------------------------------------------------------- bank
def _ledger_map(bank_root: Path, prefix: str | None = None) -> dict:
    out = {}
    for entry in sorted(bank_root.iterdir()):
        if not entry.is_dir() or not (entry / "MATCH_LEDGER.jsonl").is_file():
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        for local, line in enumerate((entry / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            out[row["canonical_match_key"]] = (entry.name, local, row)
    return out


def _load_shard(directory: Path) -> tuple[dict, dict]:
    import numpy as np

    layout = json.loads((directory / "LAYOUT.json").read_text(encoding="utf-8"))
    with np.load(directory / "BANK.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, layout


def compare_bank(keys: set) -> dict:
    from studies.observable_state_20261002.bank import match_view

    stats = FieldStats()
    mine_root, ref_root = C.stage_dir(C.DATASET_PARITY, "bank"), C.ACCEPTED["bank"]
    mine_map = _ledger_map(mine_root)
    ref_map = _ledger_map(ref_root, prefix="test_")
    ref_by_shard = defaultdict(list)
    for key in keys:
        if key in ref_map:
            ref_by_shard[ref_map[key][0]].append(key)
    mine_cache = {}
    layouts = set()
    for shard_name in sorted(ref_by_shard):
        ref_arrays, ref_layout = _load_shard(ref_root / shard_name)
        layouts.add(_canon(ref_layout))
        for key in sorted(ref_by_shard[shard_name]):
            if key not in mine_map:
                stats.add("present_in_both", False, {"key": key})
                continue
            stats.add("present_in_both", True)
            m_shard, m_local, m_ledger = mine_map[key]
            if m_shard not in mine_cache:
                mine_cache.clear()
                mine_cache[m_shard] = _load_shard(mine_root / m_shard)
            m_arrays, m_layout = mine_cache[m_shard]
            layouts.add(_canon(m_layout))
            stats.add("LAYOUT.json", _canon(m_layout) == _canon(ref_layout), {"key": key})
            a = match_view(m_arrays, m_layout, m_local)
            b = match_view(ref_arrays, ref_layout, ref_map[key][1])
            for field in sorted(set(a) | set(b)):
                if field not in a or field not in b:
                    stats.add(f"array:{field}", False, {"key": key, "missing_in": "mine" if field not in a else "reference"})
                    continue
                ok, detail = _array_equal(a[field], b[field])
                stats.add(f"array:{field}", ok, {"key": key, **(detail or {})})
            r_ledger = ref_map[key][2]
            for field in sorted(set(m_ledger) | set(r_ledger)):
                if field == "dataset":
                    stats.add("ledger:dataset[provenance: added by E2a]", field not in r_ledger and m_ledger.get(field) == C.DATASET_PARITY, {"key": key})
                    continue
                if field == "role":
                    stats.add("ledger:role[role_only: EXTERNAL vs TEST]", m_ledger.get("role") == C.EXTERNAL_ROLE and r_ledger.get("role") == "TEST", {"key": key})
                    continue
                stats.add(f"ledger:{field}", _canon(m_ledger.get(field)) == _canon(r_ledger.get(field)), {"key": key, "field": field})
    field_manifest_equal = (mine_root / "FIELD_MANIFEST.json").read_bytes() == (ref_root / "FIELD_MANIFEST.json").read_bytes()
    stats.add("FIELD_MANIFEST.json[bytes]", field_manifest_equal, {"files": [str(mine_root / "FIELD_MANIFEST.json"), str(ref_root / "FIELD_MANIFEST.json")]})
    manifest = C.read_json(mine_root / "RUN_MANIFEST.json")
    return {"fields": stats.result(), "exact": stats.exact(), "n_mine": len(mine_map), "accepted_shards_touched": len(ref_by_shard),
            "static_repair_counts_mine": manifest["static_repair_counts"],
            "note_static_repair": "n_static_values_changed / n_bans_changed are the repair's changes to the current-kernel "
                                  "arrays (0 = the producer already preserves negative cached IDs, as the accepted repair did)",
            "position_fields": ["offset_* (shard-local cumulative offsets; compared through the per-match slices)"],
            "role_only_fields": {"ledger.role": [C.EXTERNAL_ROLE, "TEST"]}, "provenance_fields": ["ledger.dataset"]}


# ------------------------------------------------------------------------------------------------------- tables
def _key_index(table, field: str) -> dict:
    return {bytes(value).decode("ascii"): index for index, value in enumerate(table[field])}


def _record_compare(stats: FieldStats, prefix: str, a, b, equal_fields, key, role_only=None) -> None:
    for name in equal_fields:
        ok, detail = _array_equal(a[name], b[name])
        stats.add(f"{prefix}.{name}", ok, {"key": key, **(detail or {})})
    for name, (mine_value, ref_value) in (role_only or {}).items():
        x, y = a[name], b[name]
        x = x.item() if hasattr(x, "item") else x
        y = y.item() if hasattr(y, "item") else y
        stats.add(f"{prefix}.{name}[role_only: {mine_value!r} vs {ref_value!r}]", x == mine_value and y == ref_value,
                  {"key": key, "values": [repr(x), repr(y)]})


def compare_preparation(keys: set, root_mine: Path, root_ref: Path, label: str) -> dict:
    import numpy as np

    stats = FieldStats()
    mt, rt = (np.load(r / "MATCH_TABLE.npy", mmap_mode="r") for r in (root_mine, root_ref))
    mv, rv = (np.load(r / "V_ROWS.npy", mmap_mode="r") for r in (root_mine, root_ref))
    mc, rc = (np.load(r / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r") for r in (root_mine, root_ref))
    stats.add("dtype:MATCH_TABLE", mt.dtype == rt.dtype, {"dtypes": [str(mt.dtype), str(rt.dtype)]})
    stats.add("dtype:V_ROWS", mv.dtype == rv.dtype, {"dtypes": [str(mv.dtype), str(rv.dtype)]})
    mi, ri = _key_index(mt, "key"), _key_index(rt, "key")
    bank_map = _ledger_map(C.stage_dir(C.DATASET_PARITY, "bank"))
    shards_mine = sorted({v[0] for v in bank_map.values()})
    for key in sorted(keys):
        if key not in mi or key not in ri:
            stats.add("present_in_both", False, {"key": key})
            continue
        stats.add("present_in_both", True)
        a, b = mt[mi[key]], rt[ri[key]]
        _record_compare(stats, "MATCH_TABLE", a, b, ("key", "original_index", "fold", "stop", "target_Blue_win", "n_nominal_rows"), key,
                        role_only={"patch_role": (C.EXTERNAL_PATCH_ROLE_CODE, 2), "role": (C.EXTERNAL_ROLE_CODE, 5)})
        shard_name, local, _ = bank_map[key]
        stats.add("MATCH_TABLE.shard/local_match[position: points to the match in the E2a bank]",
                  shards_mine[int(a["shard"])] == shard_name and int(a["local_match"]) == local, {"key": key})
        fa, na = int(a["first_nominal_row"]), int(a["n_nominal_rows"])
        fb, nb = int(b["first_nominal_row"]), int(b["n_nominal_rows"])
        ra, rb = mv[fa:fa + na], rv[fb:fb + nb]
        stats.add("V_ROWS.length", len(ra) == len(rb), {"key": key})
        if len(ra) == len(rb):
            _record_compare(stats, "V_ROWS", ra, rb, ("nominal_index", "bucket_id", "bucket_lo_ms", "bucket_hi_ms", "nominal_ms",
                                                      "actual_ms", "query_ref", "frame_index", "event_limit", "supported",
                                                      "support_reason_mask"), key)
            stats.add("V_ROWS.match_index/shard/local_match[position]",
                      bool(np.all(ra["match_index"] == mi[key]) and np.all(ra["shard"] == a["shard"]) and np.all(ra["local_match"] == a["local_match"])), {"key": key})
        ok, detail = _array_equal(np.asarray(mc[:, mi[key]]), np.asarray(rc[:, ri[key]]))
        stats.add("CONTEXT_MEMBERSHIP", ok, {"key": key, **(detail or {})})
    for array in (mt, rt, mv, rv, mc, rc):
        array._mmap.close()
    return {"fields": stats.result(), "exact": stats.exact(), "label": label,
            "position_fields": ["shard", "local_match", "first_nominal_row", "V_ROWS.match_index/shard/local_match"],
            "role_only_fields": {"patch_role": [C.EXTERNAL_PATCH_ROLE_CODE, 2], "role": [C.EXTERNAL_ROLE_CODE, 5]}}


def compare_items(keys: set) -> dict:
    import numpy as np

    stats = FieldStats()
    roots = (C.stage_dir(C.DATASET_PARITY, "items"), C.ACCEPTED["items"])
    tables = {}
    for side, root in zip(("mine", "ref"), roots):
        tables[side] = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in
                        ("MATCH_TABLE", "QUERY_TABLE", "NODE_OFFSETS", "NODE_SUPPORT", "NODE_REASON_MASK", "NOMINAL_TABLE", "CONTEXT_MEMBERSHIP")}
        for name in ("ITEM_IDS", "ITEM_COUNTS"):
            tables[side][name] = np.memmap(root / f"{name}.bin", dtype="<i8", mode="r")
        for name in ("EVENT_SOURCE_INDEX_ASSIGNED", "EVENT_SOURCE_INDEX_UNASSIGNED"):
            path = root / f"{name}.bin"
            tables[side][name] = np.memmap(path, dtype="<i4", mode="r") if path.stat().st_size else np.empty(0, dtype="<i4")
    m, r = tables["mine"], tables["ref"]
    for name in ("MATCH_TABLE", "QUERY_TABLE", "NOMINAL_TABLE"):
        stats.add(f"dtype:{name}", m[name].dtype == r[name].dtype, {"dtypes": [str(m[name].dtype), str(r[name].dtype)]})
    mi, ri = _key_index(m["MATCH_TABLE"], "canonical_key"), _key_index(r["MATCH_TABLE"], "canonical_key")
    for key in sorted(keys):
        if key not in mi or key not in ri:
            stats.add("present_in_both", False, {"key": key})
            continue
        stats.add("present_in_both", True)
        a, b = m["MATCH_TABLE"][mi[key]], r["MATCH_TABLE"][ri[key]]
        _record_compare(stats, "MATCH_TABLE", a, b, ("canonical_key", "original_index", "patch", "context_membership", "n_queries",
                                                     "n_nominal", "n_assigned_history", "n_unassigned_history"), key,
                        role_only={"role": (C.EXTERNAL_ROLE.encode("ascii"), b"TEST")})
        qa, qb = int(a["first_query"]), int(b["first_query"])
        n = int(a["n_queries"])
        if n != int(b["n_queries"]):
            continue
        A, B = m["QUERY_TABLE"][qa:qa + n], r["QUERY_TABLE"][qb:qb + n]
        _record_compare(stats, "QUERY_TABLE", A, B, ("query_index", "query_ms", "event_limit", "assigned_history_end",
                                                     "unassigned_history_end", "unassigned_observed", "unassigned_reason_mask"), key)
        stats.add("QUERY_TABLE.match_index/node_offset_row[position]",
                  bool(np.all(A["match_index"] == mi[key]) and np.array_equal(A["node_offset_row"], np.arange(qa, qa + n))), {"key": key})
        oa, ob = np.asarray(m["NODE_OFFSETS"][qa:qa + n]), np.asarray(r["NODE_OFFSETS"][qb:qb + n])
        ok, detail = _array_equal(oa - oa[:, :1], ob - ob[:, :1])
        stats.add("NODE_OFFSETS[relative to each query's first element]", ok, {"key": key, **(detail or {})})
        if n:
            for name in ("ITEM_IDS", "ITEM_COUNTS"):
                ok, detail = _array_equal(np.asarray(m[name][int(oa[0, 0]):int(oa[-1, -1])]), np.asarray(r[name][int(ob[0, 0]):int(ob[-1, -1])]))
                stats.add(f"{name}.bin[match slice]", ok, {"key": key, **(detail or {})})
        for name in ("NODE_SUPPORT", "NODE_REASON_MASK"):
            ok, detail = _array_equal(np.asarray(m[name][qa:qa + n]), np.asarray(r[name][qb:qb + n]))
            stats.add(name, ok, {"key": key, **(detail or {})})
        na, nb = int(a["first_nominal"]), int(b["first_nominal"])
        k = int(a["n_nominal"])
        _record_compare(stats, "NOMINAL_TABLE", m["NOMINAL_TABLE"][na:na + k], r["NOMINAL_TABLE"][nb:nb + k],
                        ("nominal_index", "query_ref", "supported", "actual_ms", "nominal_ms"), key)
        for part in ("assigned", "unassigned"):
            name = f"EVENT_SOURCE_INDEX_{part.upper()}"
            fa, fb, c = int(a[f"first_{part}_history"]), int(b[f"first_{part}_history"]), int(a[f"n_{part}_history"])
            ok, detail = _array_equal(np.asarray(m[name][fa:fa + c]), np.asarray(r[name][fb:fb + c]))
            stats.add(f"{name}.bin[match slice]", ok, {"key": key, **(detail or {})})
        ok, detail = _array_equal(np.asarray(m["CONTEXT_MEMBERSHIP"][:, mi[key]]), np.asarray(r["CONTEXT_MEMBERSHIP"][:, ri[key]]))
        stats.add("CONTEXT_MEMBERSHIP", ok, {"key": key, **(detail or {})})
    for side in (m, r):
        for value in side.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
    return {"fields": stats.result(), "exact": stats.exact(),
            "position_fields": ["MATCH_TABLE.shard/local_match/first_query/first_nominal/first_*_history",
                                "QUERY_TABLE.match_index/node_offset_row", "NOMINAL_TABLE.match_index", "NODE_OFFSETS absolute values"],
            "role_only_fields": {"MATCH_TABLE.role": [C.EXTERNAL_ROLE, "TEST"]},
            "not_compared": "VOCABULARY_* files (scope-level TRAIN vocabularies; not fitted for EXTERNAL rows), MATCH_STATS.jsonl worker affinity"}


def compare_projection(keys: set) -> dict:
    import numpy as np
    from studies.v_input_projection_20261002.bank import QUERY_SPECS, STATIC_SPECS

    stats = FieldStats()
    roots = {"mine": C.stage_dir(C.DATASET_PARITY, "projection"), "ref": C.ACCEPTED["projection"]}
    t = {side: {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in
                ("MATCH_TABLE", "QUERY_TABLE", "MATCH_QUERIES", *[n.upper() for n in (*QUERY_SPECS, *STATIC_SPECS)])}
         for side, root in roots.items()}
    m, r = t["mine"], t["ref"]
    mi, ri = _key_index(m["MATCH_TABLE"], "key"), _key_index(r["MATCH_TABLE"], "key")
    for key in sorted(keys):
        if key not in mi or key not in ri:
            stats.add("present_in_both", False, {"key": key})
            continue
        stats.add("present_in_both", True)
        a, b = mi[key], ri[key]
        qa, na = int(m["MATCH_QUERIES"][a]["first_query"]), int(m["MATCH_QUERIES"][a]["n_queries"])
        qb, nb = int(r["MATCH_QUERIES"][b]["first_query"]), int(r["MATCH_QUERIES"][b]["n_queries"])
        stats.add("MATCH_QUERIES.n_queries", na == nb, {"key": key, "values": [na, nb]})
        if na != nb:
            continue
        A, B = m["QUERY_TABLE"][qa:qa + na], r["QUERY_TABLE"][qb:qb + nb]
        _record_compare(stats, "QUERY_TABLE", A, B, ("local_query", "query_ms", "frame_index", "event_limit"), key)
        stats.add("QUERY_TABLE.match_index/item_query[position]",
                  bool(np.all(A["match_index"] == a) and np.array_equal(A["item_query"], np.arange(qa, qa + na))), {"key": key})
        for name in QUERY_SPECS:
            ok, detail = _array_equal(np.asarray(m[name.upper()][qa:qa + na]), np.asarray(r[name.upper()][qb:qb + nb]))
            stats.add(f"{name.upper()}.npy", ok, {"key": key, **(detail or {})})
        for name in STATIC_SPECS:
            ok, detail = _array_equal(np.asarray(m[name.upper()][a]), np.asarray(r[name.upper()][b]))
            stats.add(f"{name.upper()}.npy", ok, {"key": key, **(detail or {})})
    for side in (m, r):
        for value in side.values():
            value._mmap.close()
    tables = compare_preparation(keys, roots["mine"], roots["ref"], "projection table copies")
    manifest_equal = (roots["mine"] / "FIELD_MANIFEST.json").read_bytes() == (roots["ref"] / "FIELD_MANIFEST.json").read_bytes()
    stats.add("FIELD_MANIFEST.json[bytes]", manifest_equal)
    return {"fields": stats.result(), "table_copies": tables, "exact": stats.exact() and tables["exact"],
            "position_fields": ["MATCH_QUERIES.first_query", "QUERY_TABLE.match_index/item_query"]}


def compare_history(keys: set) -> dict:
    import numpy as np
    from studies.v_shared_history_production_20261003.store import ShardReader

    stats = FieldStats()
    sides = {}
    for side, hist_root, proj_root in (("mine", C.stage_dir(C.DATASET_PARITY, "history"), C.stage_dir(C.DATASET_PARITY, "projection")),
                                       ("ref", C.ACCEPTED["history"], C.ACCEPTED["projection"])):
        acceptance = C.read_json(hist_root / "ACCEPTANCE.json")
        plan = C.read_json(hist_root / "PLAN.json")
        matches = np.load(proj_root / "MATCH_TABLE.npy", mmap_mode="r")
        mq = np.load(proj_root / "MATCH_QUERIES.npy", mmap_mode="r")
        index = _key_index(matches, "key")
        spans = [(int(rcpt["match_range"][0]), int(rcpt["match_range"][1]), hist_root / rcpt["directory"]) for rcpt in acceptance["shards"]]
        sides[side] = {"plan": plan, "index": index, "mq": np.array(mq, copy=True), "spans": spans,
                       "complete": acceptance.get("complete") is True}  # copy: the memmap is closed below
        matches._mmap.close()
        mq._mmap.close()
    readers = {"mine": {}, "ref": {}}

    def records_for(side: str, key: str):
        s = sides[side]
        match_index = s["index"][key]
        for first, last, directory in s["spans"]:
            if first <= match_index < last:
                cache = readers[side]
                reader = cache.get(str(directory))
                if reader is None:
                    if len(cache) >= 4:
                        cache.clear()
                    reader = ShardReader(directory, identity=s["plan"])  # frozen reader: hashes binary + index
                    cache[str(directory)] = reader
                return match_index, reader.match_records(match_index)
        raise KeyError(key)

    by_ref_index = sorted(keys, key=lambda k: sides["ref"]["index"].get(k, -1))
    for key in by_ref_index:
        if key not in sides["mine"]["index"] or key not in sides["ref"]["index"]:
            stats.add("present_in_both", False, {"key": key})
            continue
        stats.add("present_in_both", True)
        ma, ra = records_for("mine", key)
        mb, rb = records_for("ref", key)
        stats.add("records.count", len(ra) == len(rb), {"key": key, "values": [len(ra), len(rb)]})
        if len(ra) != len(rb):
            continue
        fa, fb = int(sides["mine"]["mq"][ma]["first_query"]), int(sides["ref"]["mq"][mb]["first_query"])
        for x, y in zip(ra, rb):
            stats.add("record.query_id - first_query[local position]", int(x["query_id"]) - fa == int(y["query_id"]) - fb, {"key": key})
            stats.add("record.query_ms", x["query_ms"] == y["query_ms"], {"key": key, "values": [x["query_ms"], y["query_ms"]]})
            stats.add("record.nonitem", _canon(x["nonitem"]) == _canon(y["nonitem"]), {"key": key, "query_ms": x["query_ms"]})
            stats.add("record.item", _canon(x["item"]) == _canon(y["item"]), {"key": key, "query_ms": x["query_ms"]})
            stats.add("record.keys", set(x) == set(y), {"key": key})
    return {"fields": stats.result(), "exact": stats.exact() and sides["mine"]["complete"],
            "mine_acceptance_complete": sides["mine"]["complete"],
            "position_fields": ["record.query_id (global; compared as query_id - first_query)"]}


def run_parity(log) -> dict:
    C.runtime_hygiene()
    started = time.perf_counter()
    rows, by_key = _subset()
    keys = set(by_key)
    report = {"version": C.VERSION, "kind": "E2a_PARITY_REPORT", "created_utc": C.utc_now(),
              "parity_subset": str(C.PARITY_SUBSET), "parity_subset_sha256": C.sha256_file(C.PARITY_SUBSET),
              "n_matches": len(rows), "dataset": C.DATASET_PARITY, "role": C.EXTERNAL_ROLE,
              "rule": ("exact equality per match (arrays bitwise: dtype, shape, bytes; JSON by canonical text); global "
                       "positions compared through the local positions they encode; role fields may differ only to the "
                       "documented EXTERNAL codes; 'dataset' is an E2a provenance key"),
              "accepted_references": {k: str(v) for k, v in C.ACCEPTED.items()}, "artifacts": {}}
    for name, function in (("audit", lambda: compare_audit(keys)), ("bank", lambda: compare_bank(keys)),
                           ("preparation", lambda: compare_preparation(keys, C.stage_dir(C.DATASET_PARITY, "preparation"), C.ACCEPTED["preparation"], "preparation")),
                           ("items", lambda: compare_items(keys)), ("projection", lambda: compare_projection(keys)),
                           ("history", lambda: compare_history(keys))):
        t0 = time.perf_counter()
        try:
            result = function()
        except Exception as exc:  # noqa: BLE001 - recorded as a failed comparison
            import traceback

            result = {"exact": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
        result["seconds"] = round(time.perf_counter() - t0, 1)
        report["artifacts"][name] = result
        log(f"[parity] {name}: exact={result['exact']} ({result['seconds']} s)")
    report["exact_all"] = all(v["exact"] for v in report["artifacts"].values())
    first = None
    for name, result in report["artifacts"].items():
        if result["exact"]:
            continue
        if "error" in result:
            first = {"artifact": name, "error": result["error"]}
            break
        for field, row in (result.get("fields") or {}).items():
            if not row["exact"]:
                first = {"artifact": name, "field": field, "example": (row.get("examples") or [None])[0]}
                break
        if first:
            break
    report["first_difference"] = first
    report["status"] = "PASS" if report["exact_all"] else "BLOCKED"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    C.write_json(C.PARITY_REPORT, report)
    log(f"[parity] status={report['status']} first_difference={first}")
    return report
