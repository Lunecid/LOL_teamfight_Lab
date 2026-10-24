"""Full-corpus preflight (main cache 15.14/15.15/15.16): inventory only.

No training, no scoring, no label creation. Reads cache directory listing,
parses every *.meta.json, reads the npz zip central directory (member
names only), and checks events.json existence/size/first+last byte.
Joins the parent exposures.csv (read-only) per match/patch.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

KST = dt.timezone(dt.timedelta(hours=9))
ROLE = {"15.14": "TRAIN", "15.15": "VALIDATION", "15.16": "TEST"}
EXPECTED = {"15.14": 74673, "15.15": 74748, "15.16": 60579}
SUFFIXES = (".npz", ".meta.json", ".events.json")


def now():
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8"))
    os.replace(tmp, p)


def check_one(args):
    stem, cache = args
    out = {"match_id_stem": stem, "meta_ok": 0, "meta_match_id": "", "patch": "",
           "patch_full": "", "feature_version": "", "npz_zip_ok": 0,
           "npz_members": "", "events_brackets_ok": 0, "error": ""}
    errs = []
    meta = os.path.join(cache, stem + ".meta.json")
    try:
        with open(meta, "rb") as f:
            m = json.loads(f.read())
        out["meta_match_id"] = str(m.get("match_id", ""))
        out["patch"] = str(m.get("patch", ""))
        out["patch_full"] = str(m.get("patch_full", ""))
        out["feature_version"] = str(m.get("feature_version", ""))
        out["meta_ok"] = 1
    except FileNotFoundError:
        errs.append("meta_missing")
    except Exception as e:  # malformed
        errs.append("meta_error:" + type(e).__name__)
    npz = os.path.join(cache, stem + ".npz")
    try:
        with zipfile.ZipFile(npz) as z:
            out["npz_members"] = "|".join(sorted(z.namelist()))
        out["npz_zip_ok"] = 1
    except FileNotFoundError:
        errs.append("npz_missing")
    except Exception as e:
        errs.append("npz_error:" + type(e).__name__)
    ev = os.path.join(cache, stem + ".events.json")
    try:
        with open(ev, "rb") as f:
            first = f.read(64).lstrip()[:1]
            f.seek(-64, os.SEEK_END)
            last = f.read().rstrip()[-1:]
        ok = (first, last) in ((b"{", b"}"), (b"[", b"]"))
        out["events_brackets_ok"] = int(ok)
        if not ok:
            errs.append("events_bracket_mismatch")
    except FileNotFoundError:
        errs.append("events_missing")
    except Exception as e:
        errs.append("events_error:" + type(e).__name__)
    out["error"] = ";".join(errs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
    ap.add_argument("--exposures", default="outputs/postkill_objective_delay_full/exposures.csv")
    ap.add_argument("--out", default="outputs/full_corpus_preflight_20260915")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--deadline", default="2026-09-15T08:07:00+09:00")
    a = ap.parse_args()
    ws = Path(os.environ["TEAMFIGHT_WORKSPACE"])
    out = ws / a.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(a.deadline)
    t0 = time.time()
    status_p = out / "status_main.json"
    status = {"stage": "listing", "started": now(), "argv": sys.argv,
              "deadline": a.deadline, "complete": False}
    write_json(status_p, status)

    # 1. listing
    groups = collections.defaultdict(dict)
    other = collections.Counter()
    sizes = {}
    with os.scandir(a.cache) as it:
        for e in it:
            if not e.is_file():
                other["non_file_entry"] += 1
                continue
            name = e.name
            for suf in SUFFIXES:
                if name.endswith(suf):
                    stem = name[: -len(suf)]
                    groups[stem][suf] = e.stat().st_size
                    break
            else:
                other["unrecognized_file:" + (os.path.splitext(name)[1] or "<none>")] += 1
    stems = sorted(groups)
    status.update(stage="checking", n_stems=len(stems), listing_seconds=round(time.time() - t0, 1),
                  other_entries=dict(other))
    write_json(status_p, status)

    # 2. per-match header checks (checkpointed)
    rows = []
    done = 0
    stopped_at_deadline = False
    with Pool(a.workers) as pool:
        for r in pool.imap(check_one, ((s, a.cache) for s in stems), chunksize=256):
            rows.append(r)
            done += 1
            if done % 20000 == 0:
                status.update(n_checked=done, elapsed_s=round(time.time() - t0, 1), updated=now())
                write_json(status_p, status)
                if dt.datetime.now(KST) >= deadline:
                    stopped_at_deadline = True
                    pool.terminate()
                    break
    checked = {r["match_id_stem"]: r for r in rows}

    # 3. exposures join (parent output, original definition, read-only)
    exp_p = ws / a.exposures
    exp_rows = collections.Counter()
    exp_by_patch = collections.Counter()
    exp_overlap_by_patch = collections.Counter()
    exp_nan_next_by_patch = collections.Counter()
    exp_same_match_overlap = collections.Counter()
    exp_end_observed0 = collections.Counter()
    exp_patch_values = collections.defaultdict(set)
    exp_bad_rows = 0
    with open(exp_p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        exp_header = rd.fieldnames
        for row in rd:
            mid = row["match"]
            p = row["patch"]
            exp_rows[mid] += 1
            exp_patch_values[mid].add(p)
            exp_by_patch[p] += 1
            try:
                L = float(row["L"])
                ns = row["next_start"]
                if ns in ("", "nan", "NaN", "None"):
                    exp_nan_next_by_patch[p] += 1
                elif float(ns) <= L:
                    exp_overlap_by_patch[p] += 1
            except Exception:
                exp_bad_rows += 1
            if row.get("same_match_overlap") not in ("0", "", None):
                exp_same_match_overlap[p] += 1
            if row.get("end_observed") == "0":
                exp_end_observed0[p] += 1

    # 4. manifest
    fields = ["match_id", "patch", "patch_full", "role", "has_npz", "has_meta", "has_events",
              "npz_bytes", "meta_bytes", "events_bytes", "checked", "meta_ok", "meta_id_matches_stem",
              "npz_zip_ok", "events_brackets_ok", "feature_version", "exposure_rows",
              "exposure_patch_values", "error"]
    member_sets = collections.Counter()
    feat_versions = collections.Counter()
    man_rows = []
    for s in stems:
        g = groups[s]
        c = checked.get(s)
        patch = c["patch"] if c else ""
        role = ROLE.get(patch, "UNKNOWN_PATCH" if c else "UNCHECKED")
        if c:
            member_sets[c["npz_members"]] += 1
            feat_versions[c["feature_version"]] += 1
        man_rows.append({
            "match_id": s, "patch": patch, "patch_full": c["patch_full"] if c else "",
            "role": role, "has_npz": int(".npz" in g), "has_meta": int(".meta.json" in g),
            "has_events": int(".events.json" in g), "npz_bytes": g.get(".npz", ""),
            "meta_bytes": g.get(".meta.json", ""), "events_bytes": g.get(".events.json", ""),
            "checked": int(c is not None), "meta_ok": c["meta_ok"] if c else "",
            "meta_id_matches_stem": int(c["meta_match_id"] == s) if c else "",
            "npz_zip_ok": c["npz_zip_ok"] if c else "",
            "events_brackets_ok": c["events_brackets_ok"] if c else "",
            "feature_version": c["feature_version"] if c else "",
            "exposure_rows": exp_rows.get(s, 0),
            "exposure_patch_values": "|".join(sorted(exp_patch_values.get(s, ()))),
            "error": c["error"] if c else "unchecked",
        })
    man_p = out / "main_matches.csv"
    with open(man_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(man_rows)

    # 5. coverage summary
    by_role = collections.defaultdict(collections.Counter)
    id_list = []
    for r in man_rows:
        k = r["patch"] or r["role"]
        c = by_role[k]
        c["matches"] += 1
        c["complete_triplet"] += int(r["has_npz"] and r["has_meta"] and r["has_events"])
        c["missing_npz"] += 1 - r["has_npz"]
        c["missing_meta"] += 1 - r["has_meta"]
        c["missing_events"] += 1 - r["has_events"]
        c["checked"] += r["checked"]
        c["errors"] += int(bool(r["error"]))
        c["meta_id_mismatch"] += int(r["meta_id_matches_stem"] == 0)
        c["matches_with_exposure"] += int(r["exposure_rows"] > 0)
        c["matches_without_exposure"] += int(r["exposure_rows"] == 0)
        c["exposure_rows_joined"] += r["exposure_rows"]
        c["exposure_patch_disagree"] += int(bool(r["exposure_patch_values"]) and r["exposure_patch_values"] != r["patch"])
        id_list.append(r["match_id"])
    manifest_ids = set(id_list)
    exp_ids = set(exp_rows)
    exp_not_in_manifest = sorted(exp_ids - manifest_ids)
    err_counter = collections.Counter()
    for r in man_rows:
        for e in filter(None, r["error"].split(";")):
            err_counter[e] += 1
    ids_sorted = sorted(manifest_ids)
    candidate_hashes = {
        "sha256_newline_join_sorted": hashlib.sha256("\n".join(ids_sorted).encode()).hexdigest(),
        "sha256_newline_join_sorted_trailing": hashlib.sha256(("\n".join(ids_sorted) + "\n").encode()).hexdigest(),
        "sha256_json_sorted_list": hashlib.sha256(json.dumps(ids_sorted).encode()).hexdigest(),
    }
    parent_run = json.loads((ws / "outputs/postkill_objective_delay_full/run.json").read_text(encoding="utf-8"))
    coverage = {
        "generated": now(),
        "scope": ("directory listing + meta.json full parse + npz zip central directory member "
                  "names + events.json first/last byte; NOT full semantic validation of npz arrays "
                  "or events content"),
        "cache": a.cache,
        "n_stems": len(stems),
        "n_unique_ids": len(manifest_ids),
        "n_checked": len(checked),
        "stopped_at_deadline": stopped_at_deadline,
        "unchecked_remaining": len(stems) - len(checked),
        "other_dir_entries": dict(other),
        "expected_prior_census": EXPECTED,
        "by_patch": {k: dict(v) for k, v in sorted(by_role.items())},
        "role_map": ROLE,
        "error_counts": dict(err_counter),
        "feature_versions": dict(feat_versions),
        "npz_member_sets": {k: v for k, v in member_sets.most_common(20)},
        "n_distinct_npz_member_sets": len(member_sets),
        "exposures": {
            "path": str(exp_p), "sha256": sha256_file(exp_p), "header": exp_header,
            "rows_total": sum(exp_rows.values()),
            "rows_by_patch_column": dict(exp_by_patch),
            "rows_next_start_le_L_by_patch": dict(exp_overlap_by_patch),
            "rows_next_start_missing_by_patch": dict(exp_nan_next_by_patch),
            "rows_next_start_gt_L_by_patch": {p: exp_by_patch[p] - exp_overlap_by_patch[p] - exp_nan_next_by_patch[p] for p in exp_by_patch},
            "rows_same_match_overlap_nonzero_by_patch": dict(exp_same_match_overlap),
            "rows_end_observed_0_by_patch": dict(exp_end_observed0),
            "unparseable_rows": exp_bad_rows,
            "distinct_matches": len(exp_ids),
            "matches_not_in_manifest": len(exp_not_in_manifest),
            "matches_not_in_manifest_first20": exp_not_in_manifest[:20],
        },
        "parent_run_json": {"stats": parent_run.get("stats"), "match_list_sha256": parent_run.get("match_list_sha256"),
                            "script_sha256": parent_run.get("script_sha256"), "errors": parent_run.get("errors")},
        "match_list_hash_candidates": candidate_hashes,
        "match_list_hash_reproduced": any(v == parent_run.get("match_list_sha256") for v in candidate_hashes.values()),
    }
    write_json(out / "coverage_main.json", coverage)
    ids_p = out / "main_match_ids_sorted.txt"
    ids_p.write_bytes(("\n".join(ids_sorted) + "\n").encode())
    status.update(stage="done", complete=not stopped_at_deadline, n_checked=len(checked),
                  finished=now(), elapsed_s=round(time.time() - t0, 1),
                  outputs={"main_matches.csv": sha256_file(man_p), "coverage_main.json": sha256_file(out / "coverage_main.json"),
                           "main_match_ids_sorted.txt": sha256_file(ids_p)},
                  script_sha256=sha256_file(Path(__file__)))
    write_json(status_p, status)


if __name__ == "__main__":
    main()
