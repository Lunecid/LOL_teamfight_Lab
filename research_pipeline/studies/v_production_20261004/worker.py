"""Scientific V worker: one gate, one candidate, one device check, or one selection.

This module is the only code path between the scheduler and the unchanged
scientific runner (``train.py``).  Its import closure is what results are bound
to (``SCIENTIFIC_<digest>.json``); the scheduler (``reports/v_production_20261004/
pipeline.py``) and the remote dispatcher can change without invalidating gates,
candidates or V_LOCK files.  Every path in a spec is resolved against
``spec["run_root"]`` and the workspace root, so the same spec semantics run on
the main PC and on the RTX 2070 SUPER worker.

Usage: python -B studies/v_production_20261004/worker.py --spec <spec.json>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from studies.execution_progress_20261003.progress import _atomic_write_json  # noqa: E402
from studies.v_production_20261004.cache import InputCache, sha256_file  # noqa: E402

VERSION = "v_production_worker_20261004_v2"
AUDIT_REL = "reports/v_full_preprocess_20261003/completion_audit_20261004_1/ACCEPTANCE.json"
CONTRACT_REL = "reports/research_execution_one_seed_20261003/CONTRACT.json"
CONTEXTS = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
SCOPES = ("FIT", "REFIT_TRAIN")
SEED = 20261001
EXPECTED_CANDIDATES = 20
GiB = 1 << 30
RSS_LIMIT = 40 * GiB
TABLE_FAMILIES = ("logistic", "lightgbm")
NEURAL_FAMILIES = ("plain_mlp", "shared_slot_mlp", "team_set_mlp")
# A gate's own sample points must not trip on memory the scheduler already
# accounts for; they still bound the gate's own RSS and the free disk.
GATE_LIMITS = {"rss_bytes": RSS_LIMIT, "available_ram_bytes": 2 * GiB, "free_vram_bytes": 1 * GiB, "free_disk_bytes": 8 * GiB}
DEVICE_CHECK_BATCHES = 8


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rel(path: Path, base: Path) -> str:
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()


# ------------------------------------------------------------- scientific release
def scientific_digest(evidence: dict[str, str]) -> str:
    canonical = json.dumps(sorted(evidence.items()), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_scientific(path: str | Path) -> dict[str, Any]:
    """Check a SCIENTIFIC_<digest>.json against this machine's workspace files."""

    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    record = read_json(path)
    evidence = record["evidence"]
    if scientific_digest(evidence) != record["digest"]:
        raise ValueError("scientific release digest does not match its evidence")
    for name, digest in evidence.items():
        target = ROOT / name
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"stale scientific evidence: {name}")
    if read_json(ROOT / AUDIT_REL)["complete"] is not True:
        raise ValueError("full preprocessing audit is incomplete")
    return record


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"host": platform.node(), "python": platform.python_version()}
    try:
        import torch

        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001 - informational only
        pass
    return info


# --------------------------------------------------------------- reuse rules
def archive(path: Path, context_root: Path, reason: str) -> Path:
    """Move a prior non-reusable output under stale/ and log why; never delete."""

    path = Path(path)
    relative = path.resolve().relative_to(Path(context_root).resolve())
    destination = Path(context_root) / "stale" / utc_compact() / relative
    while destination.exists():
        destination = destination.with_name(destination.name + "_")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    with (Path(context_root) / "STALE_LOG.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": utc(), "from": str(path), "to": str(destination), "reason": reason}) + "\n")
    return destination


def gate_reusable(gate_root: Path, scientific_sha: str, manifest_sha: str) -> bool:
    gate, binding = gate_root / "CONSUMPTION_GATE.json", gate_root / "GATE_BINDING.json"
    if not gate.is_file() or not binding.is_file():
        return False
    record, bound = read_json(gate), read_json(binding)
    return (record.get("status") == "passed" and all(row.get("reload_equal") for row in record.get("families", ()))
            and bound.get("scientific_sha256") == scientific_sha and bound.get("cache_manifest_sha256") == manifest_sha
            and bound.get("gate_sha256") == sha256_file(gate))


def terminal_candidate(output_dir: Path, scientific_sha: str, acceptance_sha: str) -> dict[str, Any] | None:
    """Return a reusable terminal candidate record bound to this scientific release/acceptance."""

    result, binding = output_dir / "candidate_result.json", output_dir / "CANDIDATE_BINDING.json"
    if not result.is_file() or not binding.is_file():
        return None
    record, bound = read_json(result), read_json(binding)
    if bound.get("scientific_sha256") != scientific_sha or bound.get("acceptance_sha256") != acceptance_sha:
        return None
    if record.get("status") == "completed":
        return record
    if record.get("status") == "failed" and (record.get("error") or {}).get("reason") == "nonconverged":
        return record
    return None


def local_checkpoint(record: dict[str, Any], candidate_dir: Path) -> dict[str, Any]:
    """Point a (possibly remote-produced) record at its files inside ``candidate_dir``."""

    checkpoint = record.get("checkpoint")
    return {**record, "output_dir": str(candidate_dir),
            "checkpoint": str(candidate_dir / Path(str(checkpoint).replace("\\", "/")).name) if checkpoint else None}


def vlock_reusable(context_root: Path, scientific_sha: str, acceptance_sha: str) -> bool:
    path = Path(context_root) / "V_LOCK.json"
    if not path.is_file():
        return False
    record = read_json(path)
    checkpoint = (record.get("selected") or {}).get("checkpoint")
    return (record.get("complete") is True and record.get("scientific_sha256") == scientific_sha
            and record.get("training_acceptance_sha256") == acceptance_sha
            and bool(checkpoint) and Path(checkpoint).is_file() and sha256_file(checkpoint) == record.get("checkpoint_sha256"))


def check_candidate_grid(records: list[dict[str, Any]]) -> dict[str, Any]:
    """All twenty (family, card) candidates must be terminal; only logistic nonconvergence may fail."""

    from studies.v_production_20261004.train import FAMILIES, family_cards

    expected = {(family, card["id"]) for family in FAMILIES for card in family_cards(family)}
    seen = [(record["family"], record["card_id"]) for record in records]
    if len(expected) != EXPECTED_CANDIDATES or len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError(f"candidate grid is incomplete: {len(set(seen))}/{len(expected)}")
    nonconverged = []
    for record in records:
        if record["status"] == "completed":
            continue
        reason = (record.get("error") or {}).get("reason")
        if record["status"] == "failed" and record["family"] == "logistic" and reason == "nonconverged":
            nonconverged.append(f"{record['family']}/{record['card_id']}")
            continue
        raise ValueError(f"candidate {record['family']}/{record['card_id']} is not terminal: {record['status']}")
    completed = len(records) - len(nonconverged)
    if not completed:
        raise ValueError("no completed V candidate")
    return {"expected": len(expected), "completed": completed, "nonconverged": sorted(nonconverged)}


# ------------------------------------------------------- context plan and acceptance
PREPROCESS_REL = "outputs/v_full_preprocess_20261003/full_1"


def cache_name(context: str, scope: str) -> str:
    return f"{context}_{scope}_{'fit_eval' if scope == 'FIT' else 'refit_eval'}"


def dimensions(cache, context: str, scope: str, family: str):
    from studies.v_full_preprocess_20261003.artifacts import load_bundle

    if family in TABLE_FAMILIES:
        return None
    if family == "plain_mlp":
        return cache.table_width
    bundle = load_bundle(ROOT / PREPROCESS_REL / f"{context}_{scope}")
    return {"node_numeric_dim": 102, "team_numeric_dim": 8, "global_numeric_dim": 76,
            "vocab_sizes": {name: vocabulary.num_embeddings for name, vocabulary in bundle.categorical}}


def excluded_keys(context: str) -> tuple[str, ...]:
    """OOF exclusions must be identical in the FIT and REFIT preprocessing bundles."""

    from studies.v_full_preprocess_20261003.artifacts import load_bundle

    fit = tuple(load_bundle(ROOT / PREPROCESS_REL / f"{context}_FIT").evidence.excluded_game_keys)
    refit = tuple(load_bundle(ROOT / PREPROCESS_REL / f"{context}_REFIT_TRAIN").evidence.excluded_game_keys)
    if set(fit) != set(refit) or len(set(fit)) != len(fit):
        raise ValueError(f"{context} FIT/REFIT bundles disagree on excluded OOF games")
    if context != "final" and not fit:
        raise ValueError(f"{context} requires excluded OOF games")
    return fit


def cache_inventory(cache, destination: Path, run_root: Path) -> dict[str, str]:
    """Hash every cache file; keys are run-root-relative so the record is portable."""

    files = [cache.root / name for name in cache.manifest["numeric_files"].values()]
    files += [cache.root / name for name in cache.manifest["metadata_files"].values()]
    for batch in cache.manifest["batches"]:
        files.extend(cache.root / batch[name] for name in ("structured_file", "csr_file"))
    hashes = {rel(path, run_root): sha256_file(path) for path in files}
    _atomic_write_json(destination, {"rows": cache.rows, "hash_evidence": hashes})
    return hashes


def build_plan(run_root: Path, context: str) -> dict[str, Any]:
    """Verify both caches and bind exclusions, inventories, split counts and model dimensions."""

    from studies.v_production_20261004.source import CacheSource
    from studies.v_production_20261004.train import FAMILIES

    run_root = Path(run_root)
    context_root = run_root / context
    context_root.mkdir(parents=True, exist_ok=True)
    names = {scope: cache_name(context, scope) for scope in SCOPES}
    fit = InputCache(run_root / "cache" / names["FIT"], verify=True)
    refit = InputCache(run_root / "cache" / names["REFIT_TRAIN"], verify=True)
    try:
        excluded = excluded_keys(context)
        _atomic_write_json(context_root / "EXCLUDED_GAME_KEYS.json", {"context": context, "excluded_game_keys": list(excluded)})
        for cache, scope in ((fit, "FIT"), (refit, "REFIT_TRAIN")):
            cache_inventory(cache, context_root / f"CACHE_{scope}_INVENTORY.json", run_root)
        counts = CacheSource(fit_cache=fit, refit_cache=refit, excluded_game_keys=excluded).row_counts
        gate_counts = {scope: CacheSource(cache=cache, excluded_game_keys=excluded).row_counts["gate"]
                       for cache, scope in ((fit, "FIT"), (refit, "REFIT_TRAIN"))}
        dims = {f"{scope}:{family}": dimensions(cache, context, scope, family)
                for cache, scope in ((fit, "FIT"), (refit, "REFIT_TRAIN")) for family in FAMILIES}
        plan = {"context": context, "fit_cache_name": names["FIT"], "refit_cache_name": names["REFIT_TRAIN"],
                "fit_manifest_sha256": sha256_file(fit.root / "MANIFEST.json"),
                "refit_manifest_sha256": sha256_file(refit.root / "MANIFEST.json"),
                "counts": counts, "gate_counts": gate_counts, "dimensions": dims}
    finally:
        fit.close()
        refit.close()
    _atomic_write_json(context_root / "CONTEXT_PLAN.json", plan)
    return plan


def write_acceptance(run_root: Path, context: str, plan: dict[str, Any], scientific_rel: str) -> str:
    """Write (or keep) the context's training acceptance with portable relative evidence."""

    run_root = Path(run_root)
    context_root = run_root / context
    evidence: dict[str, str] = {}
    for name in (AUDIT_REL, CONTRACT_REL, scientific_rel):
        evidence[name] = sha256_file(ROOT / name)
    for key in ("fit_cache_name", "refit_cache_name"):
        evidence[f"cache/{plan[key]}/MANIFEST.json"] = sha256_file(run_root / "cache" / plan[key] / "MANIFEST.json")
    for scope in SCOPES:
        evidence[f"{context}/CACHE_{scope}_INVENTORY.json"] = sha256_file(context_root / f"CACHE_{scope}_INVENTORY.json")
    for gate in sorted((context_root / "gates").glob("*/*/CONSUMPTION_GATE.json")):
        evidence[rel(gate, run_root)] = sha256_file(gate)
    payload = {"P2b_complete": True, "training_ready": True, "full_corpus_preprocessors_fitted": True,
               "full_corpus_model_consumption_verified": True, "full_batch_resource_verified": True,
               "scientific_runner_verified": True, "seed": SEED, "context": context, "hash_evidence": evidence}
    acceptance = context_root / "TRAINING_ACCEPTANCE.json"
    if acceptance.is_file() and read_json(acceptance) != json.loads(json.dumps(payload)):
        archive(acceptance, context_root, "acceptance evidence changed")
    if not acceptance.is_file():
        _atomic_write_json(acceptance, payload)
    return sha256_file(acceptance)


# --------------------------------------------------------------------- sources
def cache_paths(spec: dict[str, Any]) -> tuple[Path, Path]:
    run_root, plan = Path(spec["run_root"]), spec["plan"]
    return run_root / "cache" / plan["fit_cache_name"], run_root / "cache" / plan["refit_cache_name"]


def _open_source(spec: dict[str, Any], family: str | None, scope: str | None = None):
    from studies.v_production_20261004.source import CacheSource

    context_root = Path(spec["run_root"]) / spec["context"]
    keys = tuple(read_json(context_root / "EXCLUDED_GAME_KEYS.json")["excluded_game_keys"])
    fit_path, refit_path = cache_paths(spec)
    if scope is None:
        fit, refit = InputCache(fit_path, verify=False), InputCache(refit_path, verify=False)
        return CacheSource(fit_cache=fit, refit_cache=refit, family=family, excluded_game_keys=keys), (fit, refit)
    cache = InputCache(fit_path if scope == "FIT" else refit_path, verify=False)
    return CacheSource(cache=cache, family=family, excluded_game_keys=keys), (cache,)


def _acceptance(spec: dict[str, Any]) -> Path:
    return Path(spec["run_root"]) / spec["context"] / "TRAINING_ACCEPTANCE.json"


def _candidate_job(spec: dict[str, Any], source, *, family: str, card_id: str, output_dir: Path):
    from studies.v_production_20261004.train import CandidateJob, RunnerConfig

    plan, context = spec["plan"], spec["context"]
    counts = plan["counts"]
    return CandidateJob(
        family=family, card_id=card_id, seed=SEED, acceptance_path=_acceptance(spec),
        output_dir=output_dir, fit_source=source, stop_source=source, refit_source=source, dev_source=source,
        context=context, fold=None if context == "final" else int(context[-1]),
        dimensions=plan["dimensions"][f"FIT:{family}"], refit_dimensions=plan["dimensions"][f"REFIT_TRAIN:{family}"],
        config=RunnerConfig(cpu_threads=3 if family == "lightgbm" else 1, device="cpu" if family in TABLE_FAMILIES else "cuda"),
        expected_fit_rows=counts["fit"], expected_stop_rows=counts["stop"],
        expected_refit_rows=counts["refit"], expected_dev_rows=counts["dev"],
        excluded_game_keys=tuple(source.excluded_game_keys))


# ------------------------------------------------------------------- workers
def run_gate(spec: dict[str, Any]) -> int:
    from studies.v_production_20261004.train import run_full_input_consumption_gate

    family, scope = spec["family"], spec["scope"]
    gate_root = Path(spec["run_root"]) / spec["context"] / "gates" / scope / family
    source, caches = _open_source(spec, family, scope)
    try:
        gate = run_full_input_consumption_gate(
            source, output_dir=gate_root, families=(family,), card_id=2,
            dimensions=spec["plan"]["dimensions"][f"{scope}:{family}"], expected_rows=spec["plan"]["gate_counts"][scope],
            seed=SEED, device="cpu" if family in TABLE_FAMILIES else "cuda", resource_limits=GATE_LIMITS)
    finally:
        for cache in caches:
            cache.close()
    if gate["status"] != "passed" or any(not row["reload_equal"] for row in gate["families"]):
        raise RuntimeError(f"full-input gate failed for {scope}/{family}")
    gate_path = gate_root / "CONSUMPTION_GATE.json"
    _atomic_write_json(gate_root / "GATE_BINDING.json", {
        "scientific_sha256": spec["scientific_sha256"], "gate_sha256": sha256_file(gate_path),
        "cache_manifest_sha256": spec["plan"]["fit_manifest_sha256" if scope == "FIT" else "refit_manifest_sha256"],
        "device": device_info(), "utc": utc()})
    return 0


DEVICE_KEYS = ("host", "gpu", "torch", "cuda")
# Failures of the machine, not of the science: the scheduler retries them, and a
# candidate directory that ended with one is continued from its checkpoints.
RETRYABLE_MARKERS = ("resource limit exceeded", "out of memory", "memoryerror", "cublas_status_alloc_failed",
                     "cuda error: out of memory", "unable to allocate", "exceeded the process rss limit",
                     "winerror 1450", "paging file is too small", "not enough space on the disk", "no space left on device")


def is_terminal(record: dict[str, Any]) -> bool:
    return record.get("status") == "completed" or (record.get("status") == "failed" and (record.get("error") or {}).get("reason") == "nonconverged")


def resource_failure(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in RETRYABLE_MARKERS)


def continuable_result(record: dict[str, Any]) -> bool:
    """A non-terminal result whose directory may be continued: a resource failure only.

    Any other failure (a data check, a reload mismatch, a solver error) is a
    scientific outcome; its directory is archived and the candidate recomputed,
    so a resumed run can never turn a rejected candidate into a completed one.
    """

    error = record.get("error") or {}
    return record.get("status") == "failed" and resource_failure(f"{error.get('type', '')}: {error.get('message', '')}")


def prepare_candidate_dir(output: Path, context_root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """Decide what an existing candidate directory is worth before a (re)launch.

    A directory is continued only when it was started under this scientific
    release, this acceptance and the same host/GPU/torch/CUDA, and its last
    attempt ended without a result (crash, power loss, kill) or with a
    resource failure; its checkpoints are then verified again by the runner.
    Anything else is archived.  The earlier attempt's result goes to ``attempts/``.
    """

    path = output / "CANDIDATE_BINDING.json"
    if not output.exists() or not any(output.iterdir()):
        return {"resumed": False, "launches": []}
    try:
        prior = read_json(path) if path.is_file() else {}
    except (OSError, ValueError):
        prior = {}
    same = (prior.get("scientific_sha256") == binding["scientific_sha256"] and prior.get("acceptance_sha256") == binding["acceptance_sha256"]
            and all((prior.get("device") or {}).get(key) == binding["device"].get(key) for key in DEVICE_KEYS))
    if not same:
        archive(output, context_root, "worker: candidate directory not bound to this release/acceptance/device")
        return {"resumed": False, "launches": []}
    result = output / "candidate_result.json"
    if result.is_file():
        try:
            record = read_json(result)
        except (OSError, ValueError):
            record = {}
        if is_terminal(record):
            return {"terminal": record}
        if not continuable_result(record):
            archive(output, context_root, "worker: earlier attempt failed scientifically; recomputed from scratch")
            return {"resumed": False, "launches": []}
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        shutil.move(str(result), str(attempts / f"candidate_result_{utc_compact()}.json"))
    launches = prior.get("launches") or [{key: prior.get(key) for key in ("worker_pid", "spec", "started_utc")}]
    return {"resumed": True, "launches": launches}


def keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs; released automatically at exit."""

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        except Exception:  # noqa: BLE001 - best effort; never blocks training
            pass


def run_candidate(spec: dict[str, Any], spec_path: Path) -> int:
    from studies.v_production_20261004.train import run_candidate_job

    context_root = Path(spec["run_root"]) / spec["context"]
    output = context_root / "candidates" / spec["family"] / spec["card_id"]
    binding = {"scientific_sha256": spec["scientific_sha256"], "acceptance_sha256": sha256_file(_acceptance(spec)), "device": device_info()}
    state = prepare_candidate_dir(output, context_root, binding)
    if "terminal" in state:  # finished before this launch (e.g. while the scheduler was down)
        return 0
    launch = {"worker_pid": os.getpid(), "spec": str(spec_path), "started_utc": utc(), "resumed": state["resumed"]}
    launches = [*state["launches"], launch]
    source, caches = _open_source(spec, spec["family"])
    try:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output / "CANDIDATE_BINDING.json", {
            **binding, "worker_pid": os.getpid(), "spec": str(spec_path),
            "started_utc": launches[0].get("started_utc") or launch["started_utc"], "launches": launches})
        result = run_candidate_job(_candidate_job(spec, source, family=spec["family"], card_id=spec["card_id"], output_dir=output))
    finally:
        for cache in caches:
            cache.close()
    return 0 if is_terminal(result.to_dict()) else 3


def run_device_check(spec: dict[str, Any]) -> int:
    """Bounded device probe for a neural family on this machine (no model is kept).

    Full-input consumption was verified by the context's gates on the main PC;
    this checks the device side: CUDA under deterministic algorithms, a short
    training run repeated bitwise, checkpoint reload equality, and memory.
    """

    import numpy as np
    import torch
    from studies.v_production_20261004 import train

    try:  # optional on the remote worker; the scheduler watches host memory itself
        import psutil
    except ImportError:
        psutil = None

    family = spec["family"]
    out = Path(spec["run_root"]) / spec["context"] / "device_checks" / f"{family}.json"
    train.configure_runtime(SEED, "cuda")
    source, caches = _open_source(spec, family)
    dims = spec["plan"]["dimensions"][f"FIT:{family}"]

    def short_run():
        train.configure_runtime(SEED, "cuda")
        model = train._probe_model(family, train.family_cards(family)[0]["id"], dimensions=dims, seed=SEED).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=train.NEURAL_LR, betas=(0.9, 0.999), eps=1e-8)
        losses = []
        for index, item in enumerate(source.iter_batches("fit")):
            inputs, labels, _ = train._normalise_batch(item)
            logits = train._model_logits(model, family, inputs, torch.device("cuda"))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(labels, device="cuda"))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.GRADIENT_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if index + 1 >= DEVICE_CHECK_BATCHES:
                break
        return model, losses

    try:
        model, first = short_run()
        weights = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])
        _, second = short_run()
        repeat_model, _ = short_run()
        repeat_weights = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in repeat_model.parameters()])
        probe = next(iter(source.iter_batches("dev")))
        inputs, _, _ = train._normalise_batch(probe)
        restored = train._probe_model(family, train.family_cards(family)[0]["id"], dimensions=dims, seed=SEED).to("cuda")
        model.eval()
        restored.eval()
        with torch.no_grad():
            train._model_logits(restored, family, inputs, torch.device("cuda"))
            restored.load_state_dict(model.state_dict(), strict=True)
            left = train._model_logits(model, family, inputs, torch.device("cuda")).cpu().numpy()
            right = train._model_logits(restored, family, inputs, torch.device("cuda")).cpu().numpy()
        free_vram, total_vram = torch.cuda.mem_get_info()
        record = {"version": VERSION, "family": family, "context": spec["context"], "batches": DEVICE_CHECK_BATCHES,
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "bitwise_repeat": first == second and bool(torch.equal(weights, repeat_weights)),
                  "reload_equal": bool(np.array_equal(left, right)), "losses": first,
                  "rss_bytes": psutil.Process().memory_info().rss if psutil else None,
                  "available_ram_bytes": psutil.virtual_memory().available if psutil else None,
                  "free_vram_bytes": int(free_vram), "total_vram_bytes": int(total_vram),
                  "scientific_sha256": spec["scientific_sha256"], "device": device_info(), "utc": utc()}
        record["status"] = "passed" if (record["deterministic_algorithms"] and record["bitwise_repeat"] and record["reload_equal"]) else "failed"
    finally:
        for cache in caches:
            cache.close()
    _atomic_write_json(out, record)
    return 0 if record["status"] == "passed" else 4


def run_select(spec: dict[str, Any]) -> int:
    import numpy as np
    import torch
    from studies.v_production_20261004.train import (
        CandidateResult, V_FAMILY_TIE_TOLERANCE, _fresh_model, _model_logits, _normalise_batch, _sigmoid,
        configure_runtime, fit_positive_slope_calibrator, select_v_dev,
    )

    plan, root = spec["plan"], Path(spec["run_root"]) / spec["context"]
    acceptance = _acceptance(spec)
    acceptance_sha = sha256_file(acceptance)
    for name in ("SELECTION.json", "V_LOCK.json"):
        if (root / name).is_file():
            archive(root / name, root, "selection rerun for the current scientific release/acceptance")
    records = []
    for result_path in sorted((root / "candidates").glob("*/*/candidate_result.json")):
        record = terminal_candidate(result_path.parent, spec["scientific_sha256"], acceptance_sha)
        if record is None:
            raise RuntimeError(f"candidate is not terminal for this release: {result_path.parent}")
        records.append(local_checkpoint(record, result_path.parent))
    grid = check_candidate_grid(records)
    results = [CandidateResult.from_dict(record) for record in records]
    selected = select_v_dev(results, tie_tolerance=V_FAMILY_TIE_TOLERANCE)
    _atomic_write_json(root / "SELECTION.json", {"selected": selected.to_dict(), "candidates": [row.to_dict() for row in results],
                                                 "grid": grid, "tie_tolerance": V_FAMILY_TIE_TOLERANCE,
                                                 "hosts": {f"{row['family']}/{row['card_id']}": (read_json(Path(row['output_dir']) / 'CANDIDATE_BINDING.json').get('device') or {}).get('host')
                                                           for row in records}})
    source, caches = _open_source(spec, selected.family)
    try:
        configure_runtime(SEED, "cpu" if selected.family in TABLE_FAMILIES else "cuda")
        job = _candidate_job(spec, source, family=selected.family, card_id=selected.card_id, output_dir=Path(selected.output_dir))
        if selected.family in TABLE_FAMILIES:
            with Path(selected.checkpoint).open("rb") as stream:
                model = pickle.load(stream)
            device = None
        else:
            device = torch.device("cuda")
            model = _fresh_model(job, dimensions=job.refit_dimensions)
            model.to(device)
            model.load_state_dict(torch.load(selected.checkpoint, map_location=device, weights_only=False)["model_state"])
            model.eval()
        probabilities, labels = [], []
        for item in source.iter_batches("cal"):
            inputs, target, metadata = _normalise_batch(item)
            if set(metadata["roles"]) != {"V_CAL"}:
                raise ValueError("V_CAL source served a non-V_CAL row")
            with torch.no_grad():
                logits = _model_logits(model, selected.family, inputs, device)
            values = logits.detach().cpu().numpy() if hasattr(logits, "detach") else logits
            probabilities.append(_sigmoid(values))
            labels.append(target)
        if sum(map(len, labels)) != plan["counts"]["cal"]:
            raise ValueError("V_CAL row coverage failed")
        calibrator = fit_positive_slope_calibrator(np.concatenate(probabilities), np.concatenate(labels))
    finally:
        for cache in caches:
            cache.close()
    for inventory in (root / f"CACHE_{scope}_INVENTORY.json" for scope in SCOPES):
        for path, digest in read_json(inventory)["hash_evidence"].items():
            target = Path(path) if Path(path).is_absolute() else Path(spec["run_root"]) / path
            if sha256_file(target) != digest:
                raise RuntimeError(f"production cache changed during training: {path}")
    # No wall-clock field: the lock's bytes depend only on its scientific content.
    _atomic_write_json(root / "V_LOCK.json", {
        "context": spec["context"], "seed": SEED, "selected": selected.to_dict(), "calibrator": calibrator.to_dict(),
        "grid": grid, "tie_tolerance": V_FAMILY_TIE_TOLERANCE, "checkpoint_sha256": sha256_file(selected.checkpoint),
        "contract_sha256": sha256_file(ROOT / CONTRACT_REL), "scientific_sha256": spec["scientific_sha256"],
        "training_acceptance_sha256": acceptance_sha, "phase": "one-seed phase 1 (seed 20261001)", "complete": True})
    return 0


def run_spec(spec_path: Path) -> int:
    spec = read_json(spec_path)
    scientific = verify_scientific(spec["scientific_release"])
    if sha256_file(ROOT / spec["scientific_release"]) != spec["scientific_sha256"] or not scientific:
        raise RuntimeError("scientific release changed after the task was scheduled")
    kind = spec["kind"]
    if kind == "gate":
        return run_gate(spec)
    if kind == "candidate":
        return run_candidate(spec, spec_path)
    if kind == "device_check":
        return run_device_check(spec)
    if kind == "select":
        return run_select(spec)
    raise ValueError(f"unknown worker kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)  # relative acceptance/release evidence resolves against the workspace root
    keep_awake()
    return run_spec(Path(args.spec))


if __name__ == "__main__":
    raise SystemExit(main())
