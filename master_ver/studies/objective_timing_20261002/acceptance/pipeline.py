"""Orchestrate scientific acceptance for a completed objective_timing run."""
from __future__ import annotations

from .. import thread_limits as TL  # noqa: F401

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .. import config as C
from .. import independent_verify as IV
from .. import io_util as IO
from . import arrays as ARR
from . import source as SRC

ARTIFACT_KEYS = (
    "objective_events_sha256",
    "component_queries_sha256",
    "aggregates_sha256",
    "schema_census_sha256",
)

CODE_FILES = (
    Path(__file__).resolve().parent / "arrays.py",
    Path(__file__).resolve().parent / "source.py",
    Path(__file__).resolve().parent / "pipeline.py",
    Path(__file__).resolve().parent / "__main__.py",
    C.STUDY / "independent_verify.py",
    C.STUDY / "io_util.py",
    C.STUDY / "config.py",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_resolved(path: Path) -> Path:
    """Resolve and normalize for containment checks (Windows case-insensitive)."""
    p = Path(path).resolve()
    if os.name == "nt":
        return Path(os.path.normcase(str(p)))
    return p


def is_same_or_descendant(path: Path, ancestor: Path) -> bool:
    """True if path is ancestor or a descendant (pathlib, case-normalized on Windows)."""
    p = _norm_resolved(path)
    a = _norm_resolved(ancestor)
    if p == a:
        return True
    try:
        p.relative_to(a)
        return True
    except ValueError:
        return False


def _write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return IO.file_sha256(path)


def _reject_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"verification out_dir already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def assert_safe_out_dir(run_dir: Path, out_dir: Path) -> None:
    """Reject writing into the target run or a descendant; no filesystem writes."""
    if is_same_or_descendant(out_dir, run_dir):
        raise ValueError(
            "out_dir must not be the target completed run or a descendant of it"
        )


def verify_artifact_hashes(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    arts = dict(manifest.get("artifacts") or {})
    names = {
        "objective_events_sha256": "objective_events.npz",
        "component_queries_sha256": "component_queries.npz",
        "aggregates_sha256": "aggregates.json",
        "schema_census_sha256": "schema_census.json",
    }
    out: dict[str, Any] = {}
    all_ok = True
    for key, name in names.items():
        path = run_dir / name
        exists = path.is_file()
        actual = IO.file_sha256(path) if exists else ""
        expected = str(arts.get(key) or "")
        ok = bool(exists and expected and actual == expected)
        out[key] = {"exists": exists, "actual": actual, "expected": expected, "ok": ok}
        all_ok = all_ok and ok
    out["all_ok"] = all_ok
    return out


def verify_frozen_baselines() -> dict[str, Any]:
    checks = {
        "anchor_manifest": {
            "actual": IO.file_sha256(C.ANCHOR_MANIFEST),
            "expected": C.EXPECTED_ANCHOR_MANIFEST_SHA256,
        },
        "components": {
            "actual": IO.file_sha256(C.COMPONENTS_NPZ),
            "expected": C.EXPECTED_COMPONENTS_SHA256,
        },
        "terminals": {
            "actual": IO.file_sha256(C.TERMINALS_JSON),
            "expected": C.EXPECTED_TERMINALS_SHA256,
        },
        "match_ids": {
            "actual": IO.file_sha256(C.ST_SOURCE / "train_match_ids.json"),
            "expected": C.EXPECTED_MATCH_IDS_SHA256,
        },
        "match_meta": {
            "actual": IO.file_sha256(C.ST_SOURCE / "train_match_meta.npz"),
            "expected": C.EXPECTED_MATCH_META_SHA256,
        },
    }
    for v in checks.values():
        v["ok"] = v["actual"] == v["expected"]
    return {"checks": checks, "all_ok": all(v["ok"] for v in checks.values())}


class _IncrementalLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.path.write_text("", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{_utc()} {msg}"
        self.lines.append(line)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


def _failure_result(
    *,
    run_dir: Path,
    out_dir: Path,
    command: list[str],
    stage: str,
    error: str,
    detail: str,
    evidence: Mapping[str, Any] | None = None,
    wall_s: float = 0.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "objective_timing_scientific_acceptance_v1",
        "accepted": False,
        "complete": False,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "stage": stage,
        "error": error,
        "detail": detail,
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "command": command,
        "environment": IO.environment_record(),
        "wall_s": float(wall_s),
        "finished_at": _utc(),
        "future_run_requirement": (
            "Run python -m studies.objective_timing_20261002.acceptance "
            "--run-dir <completed_run> --out-dir <new_verification_dir> "
            "before treating extracted statistics as scientifically accepted. "
            "Extraction RUN_MANIFEST complete flags remain historical and are not rewritten."
        ),
    }
    if evidence is not None:
        result["failure_evidence"] = dict(evidence)
    return result


def _persist_failure(out_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    _write_json(out_dir / "ACCEPTANCE_MANIFEST.json", result)
    return dict(result)


def run_acceptance(
    run_dir: Path,
    out_dir: Path,
    *,
    argv: list[str] | None = None,
    skip_raw_source: bool = False,
) -> dict[str, Any]:
    """Run full scientific acceptance; write all evidence under out_dir; never write into run_dir."""
    t0 = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    command = list(argv) if argv is not None else list(sys.argv)

    # Safety first: never write on unsafe / colliding paths
    assert_safe_out_dir(run_dir, out_dir)
    _reject_out_dir(out_dir)

    log = _IncrementalLog(out_dir / "acceptance.log")
    stage = "init"

    def fail(stage_name: str, error: str, detail: str, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
        log(f"acceptance failed stage={stage_name} error={error} detail={detail}")
        result = _failure_result(
            run_dir=run_dir,
            out_dir=out_dir,
            command=command,
            stage=stage_name,
            error=error,
            detail=detail,
            evidence=evidence,
            wall_s=time.perf_counter() - t0,
        )
        return _persist_failure(out_dir, result)

    try:
        stage = "load_manifest"
        man_path = run_dir / "RUN_MANIFEST.json"
        if not man_path.is_file():
            return fail(stage, "missing_manifest", f"missing RUN_MANIFEST.json: {man_path}")
        try:
            manifest = IO.load_json(man_path)
        except Exception as exc:  # noqa: BLE001
            return fail(
                stage,
                "corrupt_manifest",
                f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(manifest, dict):
            return fail(stage, "corrupt_manifest", "RUN_MANIFEST.json root must be an object")

        # Incompatible baseline/schema: require this study identity
        if str(manifest.get("study") or "") != "objective_timing_20261002":
            result = _failure_result(
                run_dir=run_dir,
                out_dir=out_dir,
                command=command,
                stage="study_identity",
                error="incompatible_study",
                detail=str(manifest.get("study")),
                evidence={"study": manifest.get("study")},
                wall_s=time.perf_counter() - t0,
            )
            return _persist_failure(out_dir, result)

        log(f"acceptance start run_dir={run_dir} out_dir={out_dir}")

        # Target must declare complete/full/non-smoke scope + source identities/counts
        # (reuse existing verify_reusable_source_run contract; fail before raw scan).
        stage = "target_scope"
        scope_checks = IO.verify_reusable_source_run(run_dir)
        _write_json(out_dir / "target_scope_checks.json", scope_checks)
        if not scope_checks.get("all_ok"):
            return fail(
                stage,
                "target_scope_rejected",
                "target RUN_MANIFEST failed verify_reusable_source_run contract",
                evidence=scope_checks,
            )
        log("target_scope_ok via verify_reusable_source_run")

        stage = "artifact_hashes"
        artifact_hashes = verify_artifact_hashes(run_dir, manifest)
        _write_json(out_dir / "artifact_hash_checks.json", artifact_hashes)
        if not artifact_hashes["all_ok"]:
            return fail(
                stage,
                "artifact_hash_mismatch",
                "one or more target artifacts missing or hash mismatch vs manifest",
                evidence=artifact_hashes,
            )

        stage = "frozen_baselines"
        baselines = verify_frozen_baselines()
        _write_json(out_dir / "baseline_identity.json", baselines)
        if not baselines["all_ok"]:
            return fail(
                stage,
                "frozen_baseline_mismatch",
                "local frozen baseline identity check failed",
                evidence=baselines,
            )

        # Load artifacts
        stage = "load_artifacts"
        try:
            with np.load(run_dir / "objective_events.npz", allow_pickle=False) as z:
                events = {k: np.asarray(z[k]) for k in z.files}
            with np.load(run_dir / "component_queries.npz", allow_pickle=False) as z:
                query = {k: np.asarray(z[k]) for k in z.files}
            aggregates = IO.load_json(run_dir / "aggregates.json")
            components = IO.load_components()
        except Exception as exc:  # noqa: BLE001
            return fail(
                stage,
                "corrupt_or_missing_artifact",
                f"{type(exc).__name__}: {exc}",
            )

        stage = "query_identity"
        log("query identity verification start")
        t_q = time.perf_counter()
        query_review = ARR.verify_query_identity(components, events, query)
        query_review["wall_s"] = float(time.perf_counter() - t_q)
        query_review["run_dir"] = str(run_dir)
        _write_json(out_dir / "full_query_identity_review.json", query_review)
        log(
            f"query_review all_ok={query_review['all_ok']} "
            f"n_failures={query_review['n_failures']} wall_s={query_review['wall_s']:.3f}"
        )

        stage = "aggregates"
        log("independent aggregate verification start")
        t_a = time.perf_counter()
        agg_review = IV.verify_aggregates_independent(components, query, aggregates)
        support = IV.verify_events_and_query_support(
            components, events, query, expect_full=True
        )
        agg_bundle = {
            "independent_aggregate_checks": agg_review,
            "events_and_query_support": support,
            "wall_s": float(time.perf_counter() - t_a),
            "all_ok": bool(agg_review.get("all_ok") and support.get("all_ok")),
        }
        _write_json(out_dir / "independent_aggregate_review.json", agg_bundle)
        log(f"aggregate_review all_ok={agg_bundle['all_ok']}")

        stage = "raw_source"
        if skip_raw_source:
            source_review = {
                "skipped": True,
                "all_ok": False,
                "reason": "skip_raw_source",
            }
        else:
            log("full raw source audit start")
            match_ids = IO.load_match_ids()
            terminals = IO.load_json(C.TERMINALS_JSON)
            ledger = C.ST_SOURCE / "event_ledger_train.npz"
            with np.load(ledger, allow_pickle=False) as z:
                kill_ml = np.asarray(z["match_local"])
                kill_t = np.asarray(z["t_ms"])
            source_review = SRC.audit_raw_sources(
                match_ids=match_ids,
                terminals=terminals,
                kill_match_local=kill_ml,
                kill_t_ms=kill_t,
                events=events,
                cache_dir=C.CACHE_DIR,
                progress=log,
            )
            source_review["identities"] = {
                k: bool(artifact_hashes[k]["ok"]) for k in ARTIFACT_KEYS
            }
            source_review["all_ok"] = bool(
                source_review["all_ok"] and all(source_review["identities"].values())
            )
        _write_json(out_dir / "full_raw_source_review.json", source_review)
        log(
            f"source_review all_ok={source_review.get('all_ok')} "
            f"n_failures={source_review.get('n_failures')}"
        )

        stage = "finalize"
        code_hashes = {
            str(p.relative_to(C.ROOT)): IO.file_sha256(p) for p in CODE_FILES if p.is_file()
        }
        wall_s = float(time.perf_counter() - t0)
        accepted = bool(
            scope_checks.get("all_ok")
            and artifact_hashes["all_ok"]
            and baselines["all_ok"]
            and query_review["all_ok"]
            and agg_bundle["all_ok"]
            and source_review.get("all_ok")
        )
        result = {
            "schema": "objective_timing_scientific_acceptance_v1",
            "accepted": accepted,
            "complete": accepted,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "scope": (
                "frozen TRAIN15.14 objective_timing baseline; "
                "scientific acceptance separate from extraction complete flags; "
                "no training or horizon adoption"
            ),
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "command": command,
            "environment": IO.environment_record(),
            "wall_s": wall_s,
            "finished_at": _utc(),
            "code_hashes": code_hashes,
            "input_hashes": {
                "run_manifest_sha256": IO.file_sha256(man_path),
                "objective_events_sha256": artifact_hashes["objective_events_sha256"]["actual"],
                "component_queries_sha256": artifact_hashes["component_queries_sha256"]["actual"],
                "aggregates_sha256": artifact_hashes["aggregates_sha256"]["actual"],
                "schema_census_sha256": artifact_hashes["schema_census_sha256"]["actual"],
            },
            "target_scope_checks": {
                "all_ok": scope_checks.get("all_ok"),
                "complete": scope_checks.get("complete"),
                "not_smoke": scope_checks.get("not_smoke"),
                "full_corpus": scope_checks.get("full_corpus"),
                "n_matches_ok": scope_checks.get("n_matches_ok"),
                "n_components_ok": scope_checks.get("n_components_ok"),
                "baseline_refs_ok": scope_checks.get("baseline_refs_ok"),
                "events_sha_matches_manifest": scope_checks.get("events_sha_matches_manifest"),
                "failures": scope_checks.get("failures"),
            },
            "frozen_baseline_identity": baselines,
            "artifact_identities": {k: bool(artifact_hashes[k]["ok"]) for k in ARTIFACT_KEYS},
            "source_raw_review": {
                k: source_review.get(k)
                for k in (
                    "n_matches",
                    "counts",
                    "identities",
                    "n_failures",
                    "failures_head",
                    "all_ok",
                    "wall_s",
                    "skipped",
                    "reason",
                )
                if k in source_review
            },
            "query_review_summary": {
                "n_queries": query_review.get("n_queries"),
                "n_failures": query_review.get("n_failures"),
                "all_ok": query_review.get("all_ok"),
                "wall_s": query_review.get("wall_s"),
                "method": query_review.get("method"),
            },
            "independent_aggregate_checks": agg_review,
            "events_and_query_support": support,
            "future_run_requirement": (
                "Run python -m studies.objective_timing_20261002.acceptance "
                "--run-dir <completed_run> --out-dir <new_verification_dir> "
                "before treating extracted statistics as scientifically accepted. "
                "Extraction RUN_MANIFEST complete flags remain historical and are not rewritten."
            ),
            "checks_all_ok": {
                "target_scope": bool(scope_checks.get("all_ok")),
                "artifact_hashes": artifact_hashes["all_ok"],
                "frozen_baselines": baselines["all_ok"],
                "query_identity": query_review["all_ok"],
                "aggregates": agg_bundle["all_ok"],
                "raw_source": bool(source_review.get("all_ok")),
            },
        }
        if not accepted:
            result["stage"] = "checks_failed"
            result["error"] = "scientific_checks_failed"
            result["detail"] = "one or more acceptance checks returned all_ok=false"
        _write_json(out_dir / "ACCEPTANCE_MANIFEST.json", result)
        log(f"acceptance complete accepted={accepted} wall_s={wall_s:.2f}")
        return result
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        log(f"acceptance exception stage={stage} {detail}")
        evidence = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "stage": stage,
        }
        result = _failure_result(
            run_dir=run_dir,
            out_dir=out_dir,
            command=command,
            stage=stage,
            error="validation_exception",
            detail=detail,
            evidence=evidence,
            wall_s=time.perf_counter() - t0,
        )
        return _persist_failure(out_dir, result)
