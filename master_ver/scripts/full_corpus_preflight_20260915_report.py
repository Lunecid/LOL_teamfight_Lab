"""Combine main + external preflight outputs: overlap checks, coverage.json,
validation.json and REPORT.md. No training, scoring or label creation."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

KST = dt.timezone(dt.timedelta(hours=9))
EXPECTED_MAIN = {"15.14": 74673, "15.15": 74748, "15.16": 60579}
ROLE = {"15.14": "TRAIN", "15.15": "VALIDATION", "15.16": "TEST"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ws = Path(os.environ["TEAMFIGHT_WORKSPACE"])
    out = ws / "outputs/full_corpus_preflight_20260915"
    cm = load(out / "coverage_main.json")
    ce = load(out / "coverage_external.json")
    sm = load(out / "status_main.json") or {}
    se = load(out / "status_external.json") or {}
    checks = []

    def chk(name, state, detail):
        checks.append({"check": name, "state": state, "detail": detail})

    # ---------- main ----------
    main_ids = collections.defaultdict(set)
    if cm:
        with open(out / "main_matches.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                main_ids[r["role"]].add(r["match_id"])
        full = cm["unchecked_remaining"] == 0
        chk("main_file_count_210000", "verified" if cm["n_stems"] == 210000 else "error", cm["n_stems"])
        chk("main_unique_ids_eq_stems", "verified" if cm["n_unique_ids"] == cm["n_stems"] else "error",
            [cm["n_unique_ids"], cm["n_stems"]])
        bp = cm["by_patch"]
        for p, exp in EXPECTED_MAIN.items():
            got = bp.get(p, {}).get("matches", 0)
            chk(f"main_patch_{p}_{ROLE[p]}_count", ("verified" if got == exp else "error") if full else "unchecked",
                {"expected": exp, "actual_meta_patch": got})
        other = {k: v["matches"] for k, v in bp.items() if k not in EXPECTED_MAIN}
        chk("main_unknown_or_unchecked_patch", "verified" if not other else ("unchecked" if not full else "error"), other)
        miss = {k: {m: v.get(m, 0) for m in ("missing_npz", "missing_meta", "missing_events")} for k, v in bp.items()}
        chk("main_required_files_present", "verified" if all(sum(x.values()) == 0 for x in miss.values()) else "error", miss)
        chk("main_header_checks_errors", ("verified" if not cm["error_counts"] else "error") if full else "unchecked",
            {"error_counts": cm["error_counts"], "unchecked_remaining": cm["unchecked_remaining"]})
        chk("main_meta_id_eq_filename", "verified" if full and all(v.get("meta_id_mismatch", 0) == 0 for v in bp.values()) else ("unchecked" if not full else "error"),
            {k: v.get("meta_id_mismatch", 0) for k, v in bp.items()})
        ex = cm["exposures"]
        chk("exposures_matches_all_in_manifest", "verified" if ex["matches_not_in_manifest"] == 0 else "error",
            {"not_in_manifest": ex["matches_not_in_manifest"], "sample": ex["matches_not_in_manifest_first20"]})
        chk("exposures_patch_eq_meta_patch", "verified" if full and all(v.get("exposure_patch_disagree", 0) == 0 for v in bp.values()) else ("unchecked" if not full else "error"),
            {k: v.get("exposure_patch_disagree", 0) for k, v in bp.items()})
        unk_exp = {p: n for p, n in ex["rows_by_patch_column"].items() if p not in EXPECTED_MAIN}
        chk("exposures_unknown_patch_rows", "verified" if not unk_exp else "error", unk_exp)
        chk("exposures_unparseable_rows", "verified" if ex["unparseable_rows"] == 0 else "error", ex["unparseable_rows"])
        chk("parent_match_list_sha256_reproduced", "verified" if cm["match_list_hash_reproduced"] else "unchecked",
            {"parent": cm["parent_run_json"]["match_list_sha256"], "tried": cm["match_list_hash_candidates"],
             "note": "parent hashing formula unknown; non-reproduction is not evidence of a different list"})
        chk("npz_member_name_sets", "verified" if cm["n_distinct_npz_member_sets"] == 1 else "error",
            {"n_distinct": cm["n_distinct_npz_member_sets"]})
        chk("feature_version_single", "verified" if len(cm["feature_versions"]) == 1 else "error", cm["feature_versions"])
    else:
        chk("main_scan_outputs", "error" if sm.get("stage") == "done" else "unchecked", {"status_main": sm})

    # ---------- external ----------
    ext_ids = {}
    if ce:
        for s, v in ce["sets"].items():
            with open(ws / v["manifest"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ext_ids[s] = {r["match_id"] for r in rows}
            chk(f"ext_{s}_count", "verified" if v["matches_expected"] else "error",
                {"expected": v["expected"], "actual": v["n_complete_q420_map11"], "unique": v["n_unique_ids"]})
            chk(f"ext_{s}_status_complete_only", "verified" if all(r["status"] == "complete" for r in rows) else "error",
                dict(collections.Counter(r["status"] for r in rows)))
            chk(f"ext_{s}_gameVersion_prefix_eq_api_patch", "verified" if v["gv_prefix_ne_api_patch"] == 0 else "error",
                {"gv_prefix": v["gv_prefix_values"], "api_patch": v["api_patch_values"], "public_patch_db": v["public_patch_values"]})
            chk(f"ext_{s}_files_present_size", "verified" if v["detail_missing"] + v["timeline_missing"] + v["detail_size_mismatch"] + v["timeline_size_mismatch"] == 0 else "error",
                {k: v[k] for k in ("detail_missing", "timeline_missing", "detail_size_mismatch", "timeline_size_mismatch")})
            hs = "verified" if v["hash_unchecked_matches"] == 0 and v["detail_hash_mismatch"] + v["timeline_hash_mismatch"] == 0 else (
                "error" if v["detail_hash_mismatch"] + v["timeline_hash_mismatch"] else "unchecked")
            chk(f"ext_{s}_sha256_vs_db", hs, {k: v[k] for k in ("hash_checked_matches", "hash_unchecked_matches", "detail_hash_mismatch", "timeline_hash_mismatch")})
            chk(f"ext_{s}_other_complete_rows_in_db", "verified" if not v["other_complete_rows_in_same_db_not_in_set"] else "unchecked",
                {"recorded_not_added": v["other_complete_rows_in_same_db_not_in_set"]})
    else:
        chk("external_scan_outputs", "error" if se.get("stage") == "done" else "unchecked", {"status_external": se})

    # ---------- overlap ----------
    all_sets = {f"main_{k}": v for k, v in main_ids.items()}
    all_sets.update({f"ext_{k}": v for k, v in ext_ids.items()})
    names = sorted(all_sets)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(all_sets[a] & all_sets[b])
            overlap[f"{a}&{b}"] = n
    ok = all(n == 0 for n in overlap.values())
    chk("exact_match_id_overlap_all_sets", ("verified" if ok else "error") if cm and ce else "unchecked", overlap)
    # numeric-suffix collisions across different region prefixes (info only; not deduplicated)
    by_num = collections.defaultdict(set)
    for s, ids in all_sets.items():
        for mid in ids:
            pre, _, num = mid.partition("_")
            by_num[num].add(pre)
    cross = sum(1 for v in by_num.values() if len(v) > 1)
    chk("numeric_suffix_shared_across_region_prefixes_info", "verified", {"n": cross, "note": "different regions are distinct games; not deduplicated"})
    main_prefix = collections.Counter(mid.split("_")[0] for ids in main_ids.values() for mid in ids)

    coverage = {"generated": dt.datetime.now(KST).isoformat(timespec="seconds"),
                "main": cm, "external": ce, "overlap_exact_match_id": overlap,
                "main_region_prefixes": dict(main_prefix),
                "status_main": sm, "status_external": se}
    (out / "coverage.json").write_bytes(json.dumps(coverage, indent=2, sort_keys=True, default=str).encode("utf-8"))
    states = collections.Counter(c["state"] for c in checks)
    validation = {"generated": coverage["generated"], "summary": dict(states), "checks": checks,
                  "full_model_task_complete": False,
                  "semantic_content_validation": "not performed (header/metadata scope only)"}
    (out / "validation.json").write_bytes(json.dumps(validation, indent=2, sort_keys=True, default=str).encode("utf-8"))

    # ---------- source hashes ----------
    srcs = ["scripts/full_corpus_preflight_20260915_main.py", "scripts/full_corpus_preflight_20260915_external.py",
            "scripts/full_corpus_preflight_20260915_schema.py", "scripts/full_corpus_preflight_20260915_report.py",
            "docs/CLAUDE_FULL_CORPUS_PREFLIGHT_20260915.md", "docs/FULL_CORPUS_SPLIT_CORRECTION_20260915.md",
            "docs/EXTERNAL_TEST_PLAN_20260915.md", "outputs/external_test_inventory_20260915.json",
            "outputs/postkill_objective_delay_full/exposures.csv", "outputs/postkill_objective_delay_full/run.json",
            "outputs/postkill_objective_delay_full/results.json"]
    src_hash = {s: sha(ws / s) for s in srcs if (ws / s).exists()}
    (out / "source_hashes.json").write_bytes(json.dumps(src_hash, indent=2, sort_keys=True).encode("utf-8"))

    # ---------- REPORT ----------
    L = ["# Full-corpus preflight 2026-09-15 (inventory only, no training)", "",
         "Status: PREFLIGHT ONLY. The full-model task (V/q retraining, labels, TEST evaluation, SHAP) is NOT complete and was not started.", "",
         f"Validation summary: {dict(states)} (see validation.json).", ""]
    if cm:
        L += ["## 1. Main cache census (meta.json patch field)", "",
              f"Scope: {cm['scope']}.", "",
              f"Files grouped into {cm['n_stems']} stems, {cm['n_unique_ids']} unique IDs; header-checked {cm['n_checked']}; unchecked remaining {cm['unchecked_remaining']}. Region prefixes: {dict(main_prefix)}.", "",
              "| patch | role | expected | matches | complete npz/meta/events | header errors | meta id mismatch | matches with >=1 exposure row | matches with 0 rows | exposure rows joined |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for p, v in cm["by_patch"].items():
            L.append(f"| {p} | {ROLE.get(p, 'UNKNOWN')} | {EXPECTED_MAIN.get(p, '-')} | {v.get('matches', 0)} | {v.get('complete_triplet', 0)} | {v.get('errors', 0)} | {v.get('meta_id_mismatch', 0)} | {v.get('matches_with_exposure', 0)} | {v.get('matches_without_exposure', 0)} | {v.get('exposure_rows_joined', 0)} |")
        ex = cm["exposures"]
        L += ["", f"Header error counts: {cm['error_counts'] or 'none'}. Feature versions: {cm['feature_versions']}. Distinct npz member-name sets: {cm['n_distinct_npz_member_sets']}.", "",
              "## 2. Parent exposures join (original definition, not re-detected)", "",
              f"exposures.csv sha256 {ex['sha256']}; {ex['rows_total']} rows over {ex['distinct_matches']} matches; matches not in manifest: {ex['matches_not_in_manifest']}; unparseable rows: {ex['unparseable_rows']}.", "",
              "Raw-match counts (section 1) are separate from engagement rows below. Row = one detected engagement exposure in the parent output. 'next_start<=L' is the overlap exclusion; 'missing next_start' is reported separately and is NOT classified here as eligible or excluded.", "",
              "| patch column | rows (detected) | next_start<=L (overlap excl.) | next_start missing | next_start>L | same_match_overlap!=0 | end_observed=0 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for p in sorted(ex["rows_by_patch_column"]):
            L.append(f"| {p} | {ex['rows_by_patch_column'][p]} | {ex['rows_next_start_le_L_by_patch'].get(p, 0)} | {ex['rows_next_start_missing_by_patch'].get(p, 0)} | {ex['rows_next_start_gt_L_by_patch'].get(p, 0)} | {ex['rows_same_match_overlap_nonzero_by_patch'].get(p, 0)} | {ex['rows_end_observed_0_by_patch'].get(p, 0)} |")
        L += ["", f"Parent run.json stats: {cm['parent_run_json']['stats']}. The parent 'unknown_team' count (1417) is recorded from the parent run and was not re-derived here; its effect on eligibility is unknown and must be resolved by the next spec.",
              (f"Parent match_list_sha256 reproduced: {cm['match_list_hash_reproduced']}. " + (
                  "It equals sha256('\\n'.join(sorted(match_ids))) of this manifest, i.e. the parent full run used exactly this 210,000-ID list."
                  if cm['match_list_hash_reproduced'] else "Not reproduced by the tried formulas (formula unknown; not evidence either way).")),
              "",
              "Eligible-engagement caveat: 'next_start>L' rows are the rows NOT removed by the overlap rule only. Any further eligibility rules of the downstream label definition (e.g. the parent 'unknown_team' handling, state availability at L) were not applied here, so these are upper-bound candidate counts, not final training rows.", ""]
    else:
        L += ["## 1-2. Main cache", "", f"Main scan did not produce outputs. status_main: {sm.get('stage')} n_checked={sm.get('n_checked')}", ""]
    if ce:
        L += ["## 3. External TEST sets (collector DB mode=ro, raw files)", "", f"Scope: {ce['scope']}.", "",
              "| set | platform | api_patch (raw) | DB public_patch (raw) | gameVersion prefix | expected | actual | detail/timeline missing | size mismatch | sha256 checked | sha256 mismatch | sha256 unchecked | game_creation UTC range | prior use |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for s, v in ce["sets"].items():
            L.append(f"| {s} | {v['platform']} | {v['api_patch_values']} | {v['public_patch_values']} | {v['gv_prefix_values']} | {v['expected']} | {v['n_complete_q420_map11']} | {v['detail_missing']}/{v['timeline_missing']} | {v['detail_size_mismatch'] + v['timeline_size_mismatch']} | {v['hash_checked_matches']} | {v['detail_hash_mismatch'] + v['timeline_hash_mismatch']} | {v['hash_unchecked_matches']} | {v['game_creation_min_utc']} .. {v['game_creation_max_utc']} | {v['prior_use']} |")
        L += ["", "Folder identity (raw pairs vs predefined set membership):", ""]
        for root, f in ce.get("folders", {}).items():
            L.append(f"- `{root}`: {f}")
        L += ["", f"Cross-DB same match_id rows: {ce['cross_db_same_id']['n_ids_in_multiple_db_rows']}; patterns: {ce['cross_db_same_id']['patterns']}; target IDs present in >1 DB: {ce['cross_db_same_id']['target_ids_in_multiple_dbs']}.",
              f"Target IDs also pending/excluded in some DB row: {ce['target_ids_also_pending_or_excluded_elsewhere']}. Membership was taken only from each set's own DB 'complete' rows.", "",
              "The DB's public_patch column (e.g. 26.xx) is a collector-recorded string, exported verbatim; this preflight does not assert any 16.xx/26.xx mapping.",
              "Prior use: KR_16.13 previously used (pilot adaptation/evaluation); KR_16.14_pilot has pilot provenance; KR_16.15 and NA1_16.13 prior use UNKNOWN (docs/OVERNIGHT_* only record inventory registration). Unknown is not 'untouched'.", ""]
    else:
        L += ["## 3. External", "", f"External scan did not produce outputs. status_external: {se.get('stage')}", ""]
    L += ["## 4. Exact match_id overlap", "", "Exact string equality with region prefix retained.", ""]
    L += [f"- {k}: {v}" for k, v in overlap.items()]
    L += ["", f"Numeric suffix shared across different region prefixes (info only, not deduplicated): {cross}.", "",
          "## 5. Runtime schemas and design dependencies", "",
          "runtime_schemas.json records npz array names/dtypes/shapes, meta.json keys, events.json structure and raw detail/timeline key names for one sample each (structure only).", "",
          "- The existing frozen V was trained on a mixed-patch pilot split and CANNOT be reused to label a clean 15.16 TEST (or 15.15 VALIDATION) in the full-corpus design.",
          "- TRAIN labels require out-of-fold V by match; the fold construction, VALIDATION calibration/selection dependency and their verification need a subsequent Codex specification. No cross-fitting was designed or executed here.",
          "- External raw sets (detail/timeline JSON) are not yet in the cache npz/meta/events format; cache build compatibility for 16.xx data is unverified.", "",
          "## 6. Verified / unchecked / error", ""]
    for c in checks:
        d = json.dumps(c["detail"], default=str)
        L.append(f"- [{c['state']}] {c['check']}: {d[:400]}")
    L += ["", "## 7. Notes, scope limits and discrepancies", "",
          "- Checkpointing: progress was checkpointed to status_main.json (every 20,000 matches) and status_external.json (every 200 hash batches); both runs finished before their self-stop deadlines (08:07 / 08:06 KST), so no partial manifests were needed. The checkpoints/ directory was created empty and left in place (no deletes).",
          "- Collector DBs with no predefined TEST membership were enumerated but not added: 2026_current/_collector_state_kr_16_14 (0 rows), 2026_patch_16_14/_collector_state_kr_16_14 (pending + excluded only), and EUW1 excluded 16.14/16.15 rows. Per-DB status groups are in coverage_external.json db_info.",
          "- KR_16.13 and KR_16.15 share raw folder 2026_current/kr (10,990 pairs = 10,064 + 926); set identity comes from the collector DB, not the folder.",
          "- Semantic validation NOT done: npz arrays were not decoded (only zip member names), events.json and raw detail/timeline JSON were not parsed for content (raw bytes were hash-verified against DB records).",
          "- No training, scoring, label creation, hyperparameter selection, or outcome/winner reading was performed. No files outside outputs/full_corpus_preflight_20260915 and the four new scripts were written.",
          "", "## Artifacts and hashes", "", "See source_hashes.json (scripts, governing docs, inputs), status_main.json/status_external.json (output hashes), commands.txt (exact commands).", ""]
    (out / "REPORT.md").write_bytes("\n".join(L).encode("utf-8"))


if __name__ == "__main__":
    main()
