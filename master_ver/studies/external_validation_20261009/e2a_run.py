"""E2a driver: run stages per dataset, the parity gate, and E2a_REPORT.json.

README
  Usage (Git Bash / PowerShell, from the workspace root):
    python -B studies/external_validation_20261009/e2a_parity_subset.py                       # step 1 (frozen subset)
    python -B studies/external_validation_20261009/e2a_run.py run --dataset parity_15_16 --workers 6
    python -B studies/external_validation_20261009/e2a_run.py parity                            # E2a_PARITY_REPORT.json
    python -B studies/external_validation_20261009/e2a_run.py run --dataset kr_15_18_22 --workers 6
    python -B studies/external_validation_20261009/e2a_run.py run --dataset kr_16_13 --workers 6
    python -B studies/external_validation_20261009/e2a_run.py run --dataset na1_16_13 --workers 6   # added 10-09 17:04
    python -B studies/external_validation_20261009/e2a_run.py unk --workers 6                       # E2a_UNK_CENSUS.json
    python -B studies/external_validation_20261009/e2a_run.py report                            # E2a_REPORT.json
  'run' refuses the external datasets unless E2a_PARITY_REPORT.json has status PASS, and na1_16_13 unless its
  preflight (e2a_na1.preflight: post-build check passed, manifests/index/cache agree) passes.  Each stage reuses a
  complete output of an earlier invocation (resumable); stage times go to E2a_STAGE_TIMES.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2a_na1  # noqa: E402  (import registers na1_16_13 additively)

STAGE_TIMES = C.REPORTS / "E2a_STAGE_TIMES.jsonl"
RUN_STAGES = ("audit", "bank", "preparation", "items", "projection", "history")


def _record_time(row: dict) -> None:
    STAGE_TIMES.parent.mkdir(parents=True, exist_ok=True)
    with STAGE_TIMES.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_dataset(dataset: str, stages: list[str], workers: int, log) -> None:
    from studies.external_validation_20261009 import (e2a_audit, e2a_bank, e2a_history, e2a_items, e2a_preparation,
                                                      e2a_projection)

    if dataset != C.DATASET_PARITY:
        if not C.PARITY_REPORT.exists() or C.read_json(C.PARITY_REPORT).get("status") != "PASS":
            raise SystemExit("BLOCKED: the parity gate has not passed; external datasets are not run")
    if workers < 1 or workers > C.MAX_WORKERS:
        raise SystemExit(f"workers must be 1..{C.MAX_WORKERS}")
    if dataset == e2a_na1.DATASET_NA1 and not e2a_na1.preflight(log)["passed"]:
        raise SystemExit("BLOCKED: na1_16_13 preflight failed (see E2a_NA1_PREFLIGHT.json)")
    functions = {
        "audit": lambda: e2a_audit.run_audit(dataset, workers, log),
        "bank": lambda: e2a_bank.run_bank(dataset, workers, log),
        "preparation": lambda: e2a_preparation.run_preparation(dataset, log),
        "items": lambda: e2a_items.run_items(dataset, workers, log),
        "projection": lambda: e2a_projection.run_projection(dataset, log),
        "history": lambda: e2a_history.run_history(dataset, workers, log),
    }
    for stage in stages:
        started_utc, t0 = C.utc_now(), time.perf_counter()
        status = "ok"
        error = None
        try:
            functions[stage]()
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _record_time({"dataset": dataset, "stage": stage, "started_utc": started_utc, "completed_utc": C.utc_now(),
                          "seconds": round(time.perf_counter() - t0, 1), "status": status, "error": error,
                          "workers": workers, "argv": sys.argv})
            log(f"[run] {dataset}/{stage}: {status} {round(time.perf_counter() - t0, 1)} s {error or ''}")


def _stage_disk(dataset: str) -> dict:
    out = {}
    for stage in C.STAGES:
        path = C.stage_dir(dataset, stage)
        if path.exists():
            files, size = C.dir_bytes(path)
            out[stage] = {"path": str(path), "files": files, "bytes": size, "GiB": round(size / (1 << 30), 3)}
    return out


def _dataset_summary(dataset: str) -> dict:
    summary = {"dataset": dataset, "cache_dir": str(C.CACHE[dataset]), "patch_index": str(C.PATCH_INDEX[dataset])}
    audit = C.stage_dir(dataset, "audit") / "AUDIT_MANIFEST.json"
    if not audit.exists():
        return {**summary, "status": "not run"}
    a = C.read_json(audit)
    summary["audit"] = {"n_matches_in": a["n_matches"], "status_counts": a["status_counts"],
                        "reason_counts": a["reason_counts"], "elapsed_seconds": a["elapsed_seconds"]}
    bank_manifest = C.stage_dir(dataset, "bank") / "RUN_MANIFEST.json"
    if bank_manifest.exists():
        b = C.read_json(bank_manifest)
        excluded = C.read_json(C.stage_dir(dataset, "bank") / "EXCLUDED.json")
        counts = b["counts"]
        summary["bank"] = {
            "n_entries_after_audit": b["n_entries_after_audit"], "n_bank_matches": b["n_bank_matches"],
            "n_excluded": b["n_excluded"], "excluded_reason_counts": excluded["reason_counts"],
            "excluded_examples": excluded["excluded"][:20],
            "patch_counts": b["entries_binding"]["patch_counts"],
            "n_frames": counts.get("n_frames"), "n_events": counts.get("n_events"), "n_kills": counts.get("n_kills"),
            "n_components": counts.get("n_components"), "n_zero_kill_matches": counts.get("n_zero_kill_matches"),
            "frame_proximity_timing_candidates": counts.get("n_timing_candidates"),
            "candidate_eligible_isolated_opposed": counts.get("n_candidate_eligible"),
            "primary_candidates_pre_ge_120s": counts.get("n_primary_candidates"),
            "supported_primary_engagements": counts.get("n_supported_primary"),
            "early_candidates": counts.get("n_early_candidates"),
            "n_unique_queries": counts.get("n_unique_queries"), "n_nominal_V_queries": counts.get("n_nominal_V_queries"),
            "n_supported_V_queries": counts.get("n_supported_V_queries"),
            "static_repair_counts": b["static_repair_counts"], "n_shards": len(b["shards"]),
            "static_lookup_tables": b["static_lookup_tables"], "elapsed_seconds": b["elapsed_seconds"],
            "independent_counts_equal": all(counts.get(k) == v for k, v in b["independent_counts"].items()),
        }
        support = C.read_json(C.stage_dir(dataset, "bank") / "FIELD_SUPPORT_COUNTS.json")["role_counts"].get(C.EXTERNAL_ROLE, {})
        static = Counter()
        reasons = Counter()
        for name, value in support.items():
            if name.startswith("C."):
                static[name] += value
            elif name.startswith("component_reason:"):
                reasons[name.split(":", 1)[1]] += value
        summary["bank"]["static_lookup_status_counts"] = dict(sorted(static.items()))
        summary["bank"]["component_reason_counts"] = dict(sorted(reasons.items()))
    for stage, filename in (("preparation", "RUN_MANIFEST.json"), ("items", "RUN_MANIFEST.json"),
                            ("projection", "RUN_MANIFEST.json"), ("history", "ACCEPTANCE.json")):
        path = C.stage_dir(dataset, stage) / filename
        if path.exists():
            m = C.read_json(path)
            if stage == "history":
                summary[stage] = {"matches": m["matches"], "queries": m["queries"], "shards": len(m["shards"]),
                                  "empty_query_matches": m["empty_query_matches"], "elapsed_seconds": m["elapsed_seconds"],
                                  "identity_sha256": m["identity_sha256"], "complete": m["complete"]}
            else:
                summary[stage] = {"counts": m.get("counts"), "elapsed_seconds": m.get("elapsed_seconds"), "complete": m.get("complete")}
    if "history" in summary and summary["history"].get("complete"):
        try:
            from studies.v_full_preprocess_20261003.surface import BankSurface

            with BankSurface(C.stage_dir(dataset, "history"), C.stage_dir(dataset, "projection")) as surface:
                opened = {"matches": int(len(surface.loader.matches)), "queries": int(len(surface.loader.queries))}
            summary["frozen_reader_check"] = {"reader": "studies.v_full_preprocess_20261003.surface.BankSurface "
                                                        "(acceptance, PLAN identity, receipts, shard manifests, projection hashes)",
                                              "opens": True, **opened}
        except Exception as exc:  # noqa: BLE001 - reported
            summary["frozen_reader_check"] = {"opens": False, "error": f"{type(exc).__name__}: {exc}"}
    times = []
    if STAGE_TIMES.exists():
        for line in STAGE_TIMES.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["dataset"] == dataset:
                times.append({k: row[k] for k in ("stage", "started_utc", "seconds", "status", "workers")})
    summary["stage_invocations"] = times
    summary["wall_seconds_ok_invocations"] = round(sum(r["seconds"] for r in times if r["status"] == "ok"), 1)
    summary["disk"] = _stage_disk(dataset)
    summary["disk_total_GiB"] = round(sum(v["bytes"] for v in summary["disk"].values()) / (1 << 30), 3)
    return summary


def _code_drift() -> dict:
    """B2: producer code of the accepted parent bank (snapshot) versus the current files the E2a run imports."""
    import hashlib
    import zipfile

    snapshot = C.ACCEPTED["bank_parent"] / "OBSERVABLE_CODE_SNAPSHOT.zip"
    out = {"parent_snapshot": str(snapshot), "files": {}}
    with zipfile.ZipFile(snapshot) as archive:
        for name in archive.namelist():
            parent = hashlib.sha256(archive.read(name)).hexdigest()
            current_path = C.ROOT / name
            current = C.sha256_file(current_path) if current_path.is_file() else None
            if parent != current:
                out["files"][name] = {"parent_run_sha256": parent, "current_sha256": current}
    repaired = C.read_json(C.ACCEPTED["bank"] / "RUN_MANIFEST.json")["bindings"]
    out["accepted_static_preserved_binds_current"] = all(repaired.get(name) == row["current_sha256"] for name, row in out["files"].items())
    out["explanation"] = ("kernel.py: the parent coalesced nonpositive cached selection IDs (negative bans) to 0; the current file "
                          "preserves them (the purpose of the static-preservation repair). producer.py: only "
                          "_independent_credit_check's duplicate-identity bookkeeping changed (a check that raises on mismatch; "
                          "it writes no value). The accepted static-preserved bank binds the current files; the E2a repair step "
                          "re-derives the literal IDs and changed 0 values on the parity subset.")
    return out


def _frozen_bindings_check() -> dict:
    """Every code/contract file bound by the accepted 2025 releases still has its bound sha256 (nothing was edited)."""
    sources = {
        "bank.bindings": C.read_json(C.ACCEPTED["bank"] / "RUN_MANIFEST.json")["bindings"],
        "bank.repair_bindings": C.read_json(C.ACCEPTED["bank"] / "RUN_MANIFEST.json")["repair_bindings"],
        "items.code_bindings_after": C.read_json(C.ACCEPTED["items"] / "RUN_MANIFEST.json")["code_bindings_after"],
        "projection.code_bindings_after": C.read_json(C.ACCEPTED["projection"] / "RUN_MANIFEST.json")["code_bindings_after"],
        "history.PLAN.code": C.read_json(C.ACCEPTED["history"] / "PLAN.json")["code"],
    }
    out, changed = {}, []
    for label, bindings in sources.items():
        checked = 0
        for name, expected in bindings.items():
            if not isinstance(expected, str) or len(expected) != 64:
                continue
            path = C.ROOT / name
            actual = C.sha256_file(path) if path.is_file() else None
            checked += 1
            if actual != expected:
                import subprocess

                committed = subprocess.run(["git", "-C", str(C.ROOT), "diff", "--quiet", "HEAD", "--", name],
                                           capture_output=True).returncode == 0
                changed.append({"binding": label, "file": name, "bound": expected, "current": actual,
                                "current_equals_git_HEAD": committed,
                                "note": KNOWN_PRE_EXISTING_DRIFT.get(name)})
        out[label] = checked
    return {"files_checked": out, "changed": changed, "all_unchanged": not changed,
            "changed_outside_E2a": all(row["current_equals_git_HEAD"] for row in changed)}


def _failures() -> list[dict]:
    rows = []
    if STAGE_TIMES.exists():
        for line in STAGE_TIMES.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] != "ok":
                rows.append({k: row[k] for k in ("dataset", "stage", "started_utc", "seconds", "status", "error")})
    for dataset in C.DATASETS:
        path = C.stage_dir(dataset, "bank") / "EXCLUDED.json"
        if path.exists() and C.read_json(path)["n_excluded"]:
            rows.append({"dataset": dataset, "stage": "bank", "status": "matches_excluded",
                         "n": C.read_json(path)["n_excluded"], "file": str(path)})
    return rows


KNOWN_PRE_EXISTING_DRIFT = {
    "studies/v_model_fit_input_20261003/models.py":
        "bound by the accepted shared-history PLAN (2026-10-03) through the runner's folder-wide hash list; the current "
        "file is the one in the 2026-10-04 baseline commit 4d7374c (no working-tree change), so the drift predates E2a. "
        "It is not on the history path: common.py imports .models only inside transform_rows, which semantic_row, "
        "query_view and iter_match_views never call; the parity gate reproduces the accepted history records exactly.",
}


def _stage_modules_check() -> dict:
    """The stage modules used for every dataset were last written before the first parity stage started."""
    import datetime

    first = None
    if STAGE_TIMES.exists():
        rows = [json.loads(line) for line in STAGE_TIMES.read_text(encoding="utf-8").splitlines()]
        starts = [row["started_utc"] for row in rows if row["dataset"] == C.DATASET_PARITY]
        first = min(starts) if starts else None
    out = {"first_parity_stage_started_utc": first, "modules": {}}
    for name in ("e2a_common.py", "e2a_audit.py", "e2a_bank.py", "e2a_preparation.py", "e2a_items.py",
                 "e2a_projection.py", "e2a_history.py"):
        path = C.PACKAGE / name
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out["modules"][name] = {"sha256": C.sha256_file(path), "last_written_utc": mtime,
                                "before_first_parity_stage": bool(first and mtime < first)}
    out["all_before_parity"] = all(row["before_first_parity_stage"] for row in out["modules"].values())
    return out


def _unk_summary() -> dict | None:
    from studies.external_validation_20261009.e2a_unk import UNK_CENSUS

    if not UNK_CENSUS.exists():
        return None
    census = C.read_json(UNK_CENSUS)
    out = {"file": str(UNK_CENSUS), "sha256": C.sha256_file(UNK_CENSUS), "vocabulary_bundle": census["vocabulary_bundle"],
           "datasets": {}}
    for dataset, row in census["datasets"].items():
        per_match = row["per_match"]["groups"]
        out["datasets"][dataset] = {
            "champion_slots_unk_share": per_match["static:champion_id"]["unk_share"],
            "ban_slots_unk_share": per_match["static:bans"]["unk_share"],
            "ban_slots_unk_unseen_entries": per_match["static:bans"]["unk_unseen_entries"],
            "static_fields_unk_share_per_match": {k: v["unk_share"] for k, v in per_match.items()},
            "all_queries": row["all_queries"]["namespaces"],
            "nominal_rows": row["nominal_rows"]["namespaces"],
            "item_current_unk_share_all_queries": row["all_queries"]["groups"]["item:current"]["unk_share"],
            "top_unseen_items_current": dict(list(row["top_unseen"]["item_current_entries"].items())[:10]),
            "top_unseen_champions": row["top_unseen"]["static"].get("static:champion_id"),
            "top_unseen_nonitem": dict(list(row["top_unseen"]["nonitem_weighted"].items())[:5]),
        }
    return out


def write_report(log) -> dict:
    parity = C.read_json(C.PARITY_REPORT) if C.PARITY_REPORT.exists() else None
    report = {
        "version": C.VERSION, "kind": "E2a_REPORT", "created_utc": C.utc_now(),
        "goal": "per-match caches -> observable bank (static-preserved), V input projection (+ preparation, item-node "
                "bank), shared history store, for an EXTERNAL role, with a parity gate against the accepted 2025 artifacts",
        "role_coding": {"ledger_role": C.EXTERNAL_ROLE, "preparation_patch_role": C.EXTERNAL_PATCH_ROLE_CODE,
                        "preparation_role": C.EXTERNAL_ROLE_CODE, "context_membership": 0,
                        "never_TEST_code_5": True},
        "layout": {"data_root": str(C.DATA_ROOT), "parity": f"{C.DATA_ROOT}/parity_15_16/<stage>",
                   "external": f"{C.DATA_ROOT}/<stage>/<dataset>", "stages": list(C.STAGES)},
        "code": {"package": C.package_code_hashes(), "frozen_dependencies": C.frozen_code_hashes()},
        "code_drift_B2": _code_drift(),
        "frozen_release_bindings_unchanged": _frozen_bindings_check(),
        "hooks": C.FROZEN_HOOKS,
        "exact_calls": {
            "parity_subset": "python -B studies/external_validation_20261009/e2a_parity_subset.py",
            "run": "python -B studies/external_validation_20261009/e2a_run.py run --dataset <dataset> --workers 6",
            "parity": "python -B studies/external_validation_20261009/e2a_run.py parity",
            "report": "python -B studies/external_validation_20261009/e2a_run.py report",
            "stage_functions": {
                "audit": "e2a_audit.run_audit -> cache_source_audit_20261002.run_audit._worker + _ledger_record per match",
                "bank": "e2a_bank.run_bank -> per shard: producer.produce_match per match, ShardBuilder, storage.write_shard, "
                        "run._verify_shard, repair.repair_shard, run._verify_shard",
                "preparation": "e2a_preparation.run_preparation -> partition.nominal_reference_rows, run.verify_nominal_sampler",
                "items": "e2a_items.run_items -> AcceptedSourceReader.iter_matches, PreparationIndex.rows_for, audit._source_task, "
                         "audit._BankWriter.write/finalize, audit.independent_reload_check",
                "projection": "e2a_projection.run_projection -> load_source_shard, local_source, projection.project_match",
                "history": "e2a_history.run_history -> iter_match_views, semantic_row/query_view equality, store.write_shard, "
                           "store.ShardReader reload + fingerprint re-check",
            },
        },
        "parity": None if parity is None else {
            "status": parity["status"], "exact_all": parity["exact_all"], "first_difference": parity["first_difference"],
            "per_artifact": {name: {"exact": value["exact"], **({"error": value["error"]} if "error" in value else {}),
                                    "n_fields": len(value.get("fields", {}))} for name, value in parity["artifacts"].items()},
            "report": str(C.PARITY_REPORT), "report_sha256": C.sha256_file(C.PARITY_REPORT)},
        "datasets": {dataset: _dataset_summary(dataset) for dataset in C.DATASETS},
        "na1_16_13_registration": {
            "decision": e2a_na1.DECISION, "module": "studies/external_validation_20261009/e2a_na1.py",
            "preflight": C.read_json(e2a_na1.PREFLIGHT) if e2a_na1.PREFLIGHT.exists() else None,
            "parity_gate": "none separate: same stage modules as kr_16_13; the 15.16 parity PASS covers the code path"},
        "stage_modules_unchanged_since_parity_run": _stage_modules_check(),
        "unk_census": _unk_summary(),
        "static_data": {"policy": "no network; championFull/item local Data Dragon tables only for diagnostics; "
                                  "summoner/runesReforged tables exist offline only for 15.14-15.16; raw IDs never changed",
                        "datadragon_dir": str(C.ROOT / "outputs/reest_exact_v4_20260925/datadragon_raw")},
        "identifiers": "match ids, canonical keys and counts only; no player identifiers are written",
        "failures": _failures(),
        "notes": [
            "The first E2a_PARITY_REPORT attempt crashed inside the comparison code (e2a_parity.compare_history read a "
            "view of a memmap after closing it; process exit 139). Fixed by copying MATCH_QUERIES before closing; the "
            "comparison was re-run in full and the report written by that run is the one recorded here.",
            "bank_raw/<dataset> keeps the producer shards before the static-preservation repair (input of the repair); "
            "the repair changed 0 static IDs and 0 bans in every dataset because the current kernel already preserves "
            "nonpositive cached IDs. bank_raw is not read by any later stage and can be deleted.",
            "Item-node vocabularies (VOCABULARY_*) are not fitted for EXTERNAL rows (empty FIT/REFIT scopes); raw item and "
            "champion IDs are kept (e.g. KR 16.13 champions 805 and 904 stay raw in static_ids) so the frozen V/q "
            "vocabularies map unseen IDs to UNK downstream.",
            "Shared history ran on a <=6-process pool at below-normal priority without CPU pinning (the frozen runner "
            "requires exactly eight leased CPU ids); every per-shard semantic check of the frozen runner is kept.",
            "Not part of E2a: labels, V scoring and q scoring of the external rows. Inventory blocker B6 (CONTRACT "
            "split.fresh_external_test = null) still needs the author amendment before external labels are opened.",
        ],
    }
    C.write_json(C.E2A_REPORT, report)
    log(f"[report] wrote {C.E2A_REPORT}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True, choices=C.DATASETS)
    run.add_argument("--stages", default=",".join(RUN_STAGES))
    run.add_argument("--workers", type=int, default=C.MAX_WORKERS)
    sub.add_parser("parity")
    unk = sub.add_parser("unk")
    unk.add_argument("--datasets", default=",".join(C.DATASETS))
    unk.add_argument("--workers", type=int, default=C.MAX_WORKERS)
    sub.add_parser("report")
    args = parser.parse_args(argv)
    C.runtime_hygiene()
    if args.command == "unk":
        from studies.external_validation_20261009.e2a_unk import run_unk

        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        if any(d not in C.DATASETS for d in datasets) or not 1 <= args.workers <= C.MAX_WORKERS:
            raise SystemExit("unknown dataset or worker count outside 1..6")
        log = C.Log(C.LOGS / "E2a_unk.log")
        t0, started_utc = time.perf_counter(), C.utc_now()
        run_unk(datasets, args.workers, log)
        _record_time({"dataset": ",".join(datasets), "stage": "unk_census", "started_utc": started_utc,
                      "completed_utc": C.utc_now(), "seconds": round(time.perf_counter() - t0, 1), "status": "ok",
                      "error": None, "workers": args.workers, "argv": sys.argv})
    elif args.command == "run":
        log = C.Log(C.LOGS / f"E2a_{args.dataset}.log")
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in stages if s not in RUN_STAGES]
        if unknown:
            raise SystemExit(f"unknown stages {unknown}")
        log(f"[run] start dataset={args.dataset} stages={stages} workers={args.workers}")
        run_dataset(args.dataset, stages, args.workers, log)
    elif args.command == "parity":
        from studies.external_validation_20261009.e2a_parity import run_parity

        log = C.Log(C.LOGS / "E2a_parity.log")
        report = run_parity(log)
        return 0 if report["status"] == "PASS" else 2
    else:
        write_report(C.Log(C.LOGS / "E2a_report.log"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
