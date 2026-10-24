"""External TEST preflight: read-only collector DB enumeration + raw file checks.

No training, scoring or label creation. DBs are opened with mode=ro. Only
columns needed for membership/provenance are exported (no puuid, no
winner/outcome fields). Raw detail/timeline bytes are hashed against the
DB-recorded sha256 in priority order until the deadline.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KST = dt.timezone(dt.timedelta(hours=9))
# Predefined TEST membership (api_patch string preserved, never remapped).
TARGETS = [
    # (set_id, db path, platform, api_patch, expected, prior_use)
    ("KR_16.15", r"D:\LOL_Project\data\raw\2026_current\_collector_state_1615\collector.sqlite3", "kr", "16.15", 926, "unknown"),
    ("KR_16.14_pilot", r"D:\LOL_Project\data\raw\2026_patch_16_14_pilot\_collector_state_kr_16_14_pilot\collector.sqlite3", "kr", "16.14", 200, "pilot_provenance"),
    ("NA1_16.13", r"D:\LOL_Project\data\raw\2026_current\_collector_state_na1_16_13\collector.sqlite3", "na1", "16.13", 10000, "unknown"),
    ("KR_16.13", r"D:\LOL_Project\data\raw\2026_current\_collector_state\collector.sqlite3", "kr", "16.13", 10064, "previously_used"),
    ("EUW1_complete", r"D:\LOL_Project\data\raw\2026_current\_collector_state_euw1_16_13\collector.sqlite3", "euw1", None, 0, "unknown"),
]
COLS = ["match_id", "platform", "status", "queue_id", "map_id", "game_mode", "game_version",
        "api_patch", "public_patch", "game_creation", "completed_at", "retrieved_at",
        "detail_sha256", "timeline_sha256", "detail_bytes", "timeline_bytes", "exclusion_reason"]


def now():
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8"))
    os.replace(tmp, p)


def ro(db):
    return sqlite3.connect(Path(db).as_uri() + "?mode=ro", uri=True)


def raw_root(db):
    # <raw>/<collection>/_collector_state*/collector.sqlite3 -> <raw>/<collection>
    return Path(db).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="outputs/external_test_inventory_20260915.json")
    ap.add_argument("--out", default="outputs/full_corpus_preflight_20260915")
    ap.add_argument("--hash-threads", type=int, default=1)
    ap.add_argument("--deadline", default="2026-09-15T08:06:00+09:00")
    a = ap.parse_args()
    ws = Path(os.environ["TEAMFIGHT_WORKSPACE"])
    out = ws / a.out
    ext = out / "external"
    ext.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(a.deadline)
    t0 = time.time()
    status_p = out / "status_external.json"
    status = {"stage": "db_enumeration", "started": now(), "argv": sys.argv, "complete": False}
    write_json(status_p, status)

    inv_p = ws / a.inventory
    inv = json.loads(inv_p.read_text(encoding="utf-8"))
    dbs = [d["db"] for d in inv]
    db_info = {}
    all_rows = {}  # db -> list of dict
    for db in dbs:
        info = {"exists": os.path.exists(db)}
        if info["exists"]:
            st = os.stat(db)
            info.update(bytes=st.st_size, mtime=dt.datetime.fromtimestamp(st.st_mtime, KST).isoformat())
            con = ro(db)
            try:
                info["groups"] = [dict(zip(["platform", "api_patch", "public_patch", "status", "queue_id", "map_id", "n"], r))
                                  for r in con.execute("select platform, api_patch, public_patch, status, queue_id, map_id, count(*) from matches group by 1,2,3,4,5,6 order by 1,2,3,4")]
                gvc = collections.Counter(
                    (s, ap_, ".".join(str(gv).split(".")[:2]))
                    for s, ap_, gv in con.execute("select status, api_patch, game_version from matches where game_version is not null"))
                info["gv_prefix_groups"] = [{"status": k[0], "api_patch": k[1], "game_version_prefix": k[2], "n": v}
                                            for k, v in sorted(gvc.items(), key=lambda kv: str(kv[0]))]
                cur = con.execute("select " + ",".join(COLS) + " from matches where status='complete'")
                all_rows[db] = [dict(zip(COLS, r)) for r in cur.fetchall()]
                info["metadata"] = dict(con.execute("select key, value from metadata").fetchall())
                info["all_status_ids"] = None
            finally:
                con.close()
        db_info[db] = info
    # all match ids by DB and status (for alias/overlap)
    status_ids = {}
    for db in dbs:
        if not db_info[db]["exists"]:
            continue
        con = ro(db)
        try:
            status_ids[db] = collections.defaultdict(set)
            for mid, stt in con.execute("select match_id, status from matches"):
                status_ids[db][stt].add(mid)
        finally:
            con.close()

    status.update(stage="file_checks", db_seconds=round(time.time() - t0, 1))
    write_json(status_p, status)

    # Build target manifests
    manifests = {}
    for set_id, db, plat, patch, expected, prior in TARGETS:
        rows = all_rows.get(db, [])
        sel = [r for r in rows if r["platform"] == plat and r["queue_id"] == 420 and r["map_id"] == 11
               and (patch is None or r["api_patch"] == patch)]
        sel_ids = {r["match_id"] for r in sel}
        other = collections.Counter((r["platform"], r["api_patch"], r["queue_id"], r["map_id"]) for r in rows if r["match_id"] not in sel_ids)
        root = raw_root(db) / plat
        for r in sel:
            dp = root / "detail" / (r["match_id"] + ".json")
            tp = root / "timeline" / (r["match_id"] + ".json")
            r["set_id"] = set_id
            r["collector_db"] = db
            r["raw_folder"] = str(root)
            r["prior_use"] = prior
            r["gv_prefix"] = ".".join(str(r["game_version"] or "").split(".")[:2])
            r["gv_prefix_eq_api_patch"] = int(r["gv_prefix"] == r["api_patch"])
            for kind, p in (("detail", dp), ("timeline", tp)):
                try:
                    sz = os.stat(p).st_size
                    r[kind + "_exists"] = 1
                    r[kind + "_size_ok"] = int(sz == r[kind + "_bytes"])
                except FileNotFoundError:
                    r[kind + "_exists"] = 0
                    r[kind + "_size_ok"] = 0
                r[kind + "_hash_ok"] = ""  # unchecked until hashed
        manifests[set_id] = {"rows": sel, "expected": expected, "db": db, "platform": plat,
                             "api_patch": patch, "prior_use": prior, "other_complete_in_db": {str(k): v for k, v in other.items()},
                             "raw_folder": str(root)}

    # On-disk files not in DB complete set, per raw folder
    disk = {}
    for set_id, m in manifests.items():
        root = Path(m["raw_folder"])
        if str(root) in disk:
            continue
        d = {}
        for kind in ("detail", "timeline", "quarantine"):
            kp = root / kind
            d[kind] = sorted(e.name[:-5] for e in os.scandir(kp) if e.name.endswith(".json")) if kp.exists() else None
        disk[str(root)] = d

    # Hash verification, priority order, until deadline
    stopped = False
    hashed = 0

    def verify(r):
        res = {}
        for kind in ("detail", "timeline"):
            if r[kind + "_exists"]:
                p = Path(r["raw_folder"]) / kind / (r["match_id"] + ".json")
                res[kind] = int(sha256_file(p) == r[kind + "_sha256"])
        return r, res

    with ThreadPoolExecutor(a.hash_threads) as ex:
        for set_id, m in manifests.items():
            if stopped:
                break
            batch = 200
            rows = m["rows"]
            for i in range(0, len(rows), batch):
                if dt.datetime.now(KST) >= deadline:
                    stopped = True
                    break
                for r, res in ex.map(verify, rows[i:i + batch]):
                    for kind, ok in res.items():
                        r[kind + "_hash_ok"] = ok
                    hashed += 1
                status.update(hashed=hashed, current_set=set_id, updated=now(), elapsed_s=round(time.time() - t0, 1))
                write_json(status_p, status)

    # Write manifests + summaries
    fields = ["set_id", "match_id", "platform", "api_patch", "public_patch", "game_version", "gv_prefix",
              "gv_prefix_eq_api_patch", "queue_id", "map_id", "game_mode", "game_creation", "status",
              "collector_db", "raw_folder", "prior_use", "detail_exists", "timeline_exists",
              "detail_bytes", "timeline_bytes", "detail_size_ok", "timeline_size_ok",
              "detail_sha256", "timeline_sha256", "detail_hash_ok", "timeline_hash_ok"]
    summary = {"generated": now(), "scope": "read-only SQLite (mode=ro) + raw file stat + sha256 of raw bytes vs DB-recorded sha256 where time allowed; no JSON content parsing, no outcome fields read",
               "inventory": {"path": str(inv_p), "sha256": sha256_file(inv_p)},
               "db_info": {k: {kk: vv for kk, vv in v.items() if kk != "all_status_ids"} for k, v in db_info.items()},
               "hash_stopped_at_deadline": stopped, "sets": {}}
    for set_id, m in manifests.items():
        rows = m["rows"]
        mp = ext / f"external_{set_id}.csv"
        with open(mp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["match_id"]))
        ids = [r["match_id"] for r in rows]
        gc = [r["game_creation"] for r in rows if r["game_creation"] is not None]
        root = m["raw_folder"]
        dsk = disk.get(root, {})
        det = set(dsk.get("detail") or [])
        tl = set(dsk.get("timeline") or [])
        summary["sets"][set_id] = {
            "manifest": str(mp.relative_to(ws)), "manifest_sha256": sha256_file(mp),
            "db": m["db"], "platform": m["platform"], "api_patch_filter": m["api_patch"],
            "raw_folder": root, "prior_use": m["prior_use"],
            "expected": m["expected"], "n_complete_q420_map11": len(rows), "n_unique_ids": len(set(ids)),
            "matches_expected": len(rows) == m["expected"],
            "api_patch_values": dict(collections.Counter(r["api_patch"] for r in rows)),
            "public_patch_values": dict(collections.Counter(r["public_patch"] for r in rows)),
            "gv_prefix_values": dict(collections.Counter(r["gv_prefix"] for r in rows)),
            "gv_prefix_ne_api_patch": sum(1 - r["gv_prefix_eq_api_patch"] for r in rows),
            "game_mode_values": dict(collections.Counter(r["game_mode"] for r in rows)),
            "id_prefix_values": dict(collections.Counter(r["match_id"].split("_")[0] for r in rows)),
            "game_creation_min_utc": dt.datetime.fromtimestamp(min(gc) / 1000, dt.timezone.utc).isoformat() if gc else None,
            "game_creation_max_utc": dt.datetime.fromtimestamp(max(gc) / 1000, dt.timezone.utc).isoformat() if gc else None,
            "detail_missing": sum(1 - r["detail_exists"] for r in rows),
            "timeline_missing": sum(1 - r["timeline_exists"] for r in rows),
            "detail_size_mismatch": sum(1 for r in rows if r["detail_exists"] and not r["detail_size_ok"]),
            "timeline_size_mismatch": sum(1 for r in rows if r["timeline_exists"] and not r["timeline_size_ok"]),
            "hash_checked_matches": sum(1 for r in rows if r["detail_hash_ok"] != "" or r["timeline_hash_ok"] != ""),
            "detail_hash_mismatch": sum(1 for r in rows if r["detail_hash_ok"] == 0),
            "timeline_hash_mismatch": sum(1 for r in rows if r["timeline_hash_ok"] == 0),
            "hash_unchecked_matches": sum(1 for r in rows if r["detail_hash_ok"] == "" and r["timeline_hash_ok"] == ""),
            "other_complete_rows_in_same_db_not_in_set": m["other_complete_in_db"],
            "folder_detail_files": len(det), "folder_timeline_files": len(tl),
            "folder_pairs": len(det & tl),
            "folder_quarantine_files": len(dsk.get("quarantine") or []),
        }
    # folder pairs not in any complete DB set for that folder
    for root, d in disk.items():
        pairs = set(d.get("detail") or []) & set(d.get("timeline") or [])
        in_sets = set()
        for set_id, m in manifests.items():
            if m["raw_folder"] == root:
                in_sets |= {r["match_id"] for r in m["rows"]}
        complete_any = set()
        for db, st in status_ids.items():
            if str(raw_root(db)) in root:
                complete_any |= st.get("complete", set())
        summary.setdefault("folders", {})[root] = {
            "pairs": len(pairs), "pairs_in_target_sets": len(pairs & in_sets),
            "pairs_not_in_target_sets": len(pairs - in_sets),
            "pairs_not_in_target_sets_but_complete_in_some_db_same_collection": len((pairs - in_sets) & complete_any),
            "detail_only": len(set(d.get("detail") or []) - set(d.get("timeline") or [])),
            "timeline_only": len(set(d.get("timeline") or []) - set(d.get("detail") or [])),
            "target_ids_missing_pair": len(in_sets - pairs),
        }
    # cross-DB alias / status overlap (exact match_id)
    id_dbs = collections.defaultdict(list)
    for db, st in status_ids.items():
        for stt, ids in st.items():
            for mid in ids:
                id_dbs[mid].append((db, stt))
    multi = {mid: v for mid, v in id_dbs.items() if len(v) > 1}
    target_ids = {set_id: {r["match_id"] for r in m["rows"]} for set_id, m in manifests.items()}
    alias = collections.Counter()
    for mid, v in multi.items():
        key = " + ".join(sorted(f"{Path(db).parent.name}:{stt}" for db, stt in v))
        alias[key] += 1
    summary["cross_db_same_id"] = {"n_ids_in_multiple_db_rows": len(multi), "patterns": dict(alias.most_common(50)),
                                   "target_ids_in_multiple_dbs": {s: sum(1 for i in ids if i in multi) for s, ids in target_ids.items()}}
    pend_excl = collections.defaultdict(set)
    for db, st in status_ids.items():
        for stt in ("pending", "excluded"):
            pend_excl[stt] |= st.get(stt, set())
    summary["target_ids_also_pending_or_excluded_elsewhere"] = {
        s: {stt: len(ids & pend_excl[stt]) for stt in ("pending", "excluded")} for s, ids in target_ids.items()}
    ids_p = ext / "external_target_ids.json"
    ids_p.write_bytes(json.dumps({s: sorted(v) for s, v in target_ids.items()}).encode())
    summary["external_target_ids_sha256"] = sha256_file(ids_p)
    write_json(out / "coverage_external.json", summary)
    status.update(stage="done", complete=not stopped, finished=now(), elapsed_s=round(time.time() - t0, 1),
                  script_sha256=sha256_file(Path(__file__)))
    write_json(status_p, status)


if __name__ == "__main__":
    main()
