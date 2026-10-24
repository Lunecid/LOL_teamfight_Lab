"""Tests for the E6 run (reports/tog_experiments_20261012/PRESPEC_E6_KO.md section 12).

Synthetic data only.  The selection, build checks, guards, score-v environment, worker capture and acceptance rule are
checked on small inputs under tmp_path; the real raw folder, the external store, the training root and the
repository ledgers are never touched.  One test starts a real spawn pool (one worker) to check the worker records end
to end; set E6_SKIP_SPAWN_TESTS=1 to skip it while a pipeline stage holds the CPU budget.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from studies.external_validation_20261009 import e2a_common as C
from studies.external_validation_20261009 import e2bc_common as B
import studies.external_validation_20261009 as EXT
from studies.tog_experiments_20261012 import e6_run as E6

SPAWN = pytest.mark.skipif(os.environ.get("E6_SKIP_SPAWN_TESTS") == "1", reason="spawn tests disabled while a stage runs")


@pytest.fixture
def clean_stage():
    """No active stage before or after a test; E6_STAGE / E6_OUT restored; the executor names restored."""
    saved = {name: os.environ.get(name) for name in ("E6_STAGE", "E6_OUT")}
    for name in saved:
        os.environ.pop(name, None)
    E6._STAGE = None
    yield
    E6._STAGE = None
    E6.restore_executor()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def bound_state():
    """Snapshot and restore every module attribute bind_outputs / register_dataset may change."""
    saved = {
        "EXT": {k: getattr(EXT, k) for k in ("REPORTS", "LOGS", "WORK", "RUNTIME_REPO", "SAMPLE_MANIFEST", "SAMPLE_MANIFEST_SHA")},
        "EXT_dicts": {k: dict(getattr(EXT, k)) for k in ("RAW", "CACHE_DIR", "PATCH_INDEX")},
        "C": {k: getattr(C, k) for k in ("LOGS", "DATASETS", "stage_dir")},
        "C_dicts": {k: dict(getattr(C, k)) for k in ("CACHE", "PATCH_INDEX", "EXPECTED_PATCHES")},
        "B": {k: getattr(B, k) for k in ("labels_dir", "q_inputs_dir", "scores_dir", "STAGE_TIMES")},
    }
    yield saved
    for k, v in saved["EXT"].items():
        setattr(EXT, k, v)
    for k, v in saved["EXT_dicts"].items():
        getattr(EXT, k).clear()
        getattr(EXT, k).update(v)
    for k, v in saved["C"].items():
        setattr(C, k, v)
    for k, v in saved["C_dicts"].items():
        getattr(C, k).clear()
        getattr(C, k).update(v)
    for k, v in saved["B"].items():
        setattr(B, k, v)


# ----------------------------------------------------------------------------- selection (PRESPEC 2.2)

def test_h6_is_sha256_of_the_new_tag_and_the_match_id():
    expected = hashlib.sha256("tog_e6_20261014:KR_7812345678".encode("utf-8")).hexdigest()
    assert E6.match_hash6("KR_7812345678") == expected
    assert E6.HASH_TAG == "tog_e6_20261014:"
    assert E6.match_hash6("KR_1") != EXT.match_hash("KR_1")          # not the old sample's order


def test_largest_remainder_reproduces_the_prespec_shares():
    remaining = {"15.18": 49153, "15.19": 65909, "15.20": 63706, "15.21": 54106, "15.22": 28794}
    quotas = E6.largest_remainder(remaining, 40000)
    assert quotas == {"15.18": 7514, "15.19": 10075, "15.20": 9738, "15.21": 8271, "15.22": 4402}
    assert quotas == E6.QUOTAS and sum(quotas.values()) == 40000
    assert E6.REMAINING == remaining


def test_largest_remainder_breaks_equal_remainders_by_patch_name():
    assert E6.largest_remainder({"15.20": 1, "15.18": 1, "15.19": 1}, 2) == {"15.18": 1, "15.19": 1, "15.20": 0}
    assert E6.largest_remainder({"b": 3, "a": 3}, 3) == {"a": 2, "b": 1}
    with pytest.raises(ValueError):
        E6.largest_remainder({"a": 1}, 2)


def _ids(prefix, n):
    return [f"{prefix}_{i}" for i in range(n)]


def test_select_slice_excludes_the_existing_sample_and_takes_the_smallest_h6():
    cands = {"15.18": _ids("A", 40), "15.19": _ids("B", 20)}
    existing = set(cands["15.18"][:10]) | set(cands["15.19"][:5])
    out = E6.select_slice(cands, existing, total=9)
    assert out["remaining"] == {"15.18": 30, "15.19": 15}
    assert out["quotas"] == E6.largest_remainder({"15.18": 30, "15.19": 15}, 9) == {"15.18": 6, "15.19": 3}
    for patch, chosen in out["selected"].items():
        pool = sorted(set(cands[patch]) - existing, key=lambda m: (E6.match_hash6(m), m))
        assert chosen == pool[:out["quotas"][patch]]
        assert not set(chosen) & existing
    assert out["existing"] == {"15.18": 10, "15.19": 5}


def test_select_slice_blocks_when_an_existing_match_is_not_a_candidate():
    with pytest.raises(E6.Blocked, match="not candidates"):
        E6.select_slice({"15.18": _ids("A", 5)}, {"A_1", "Z_9"}, total=2)


@pytest.mark.parametrize("what,found,expected", [
    ("detail files", 301743, 301744), ("timeline files", 301725, 301724),
    ("candidates", {"15.18": 57152}, {"15.18": 57153}), ("quotas", {"15.22": 4401}, {"15.22": 4402})])
def test_count_checks_block_on_any_difference(what, found, expected):
    with pytest.raises(E6.Blocked, match=what):
        E6.check_equal(what, found, expected)
    E6.check_equal(what, expected, expected)


def test_selection_overlap_checks_block():
    with pytest.raises(E6.Blocked, match="2025 cache"):
        E6.check_disjoint("2025 cache index", {"KR_1", "KR_2"}, {"KR_2"})
    E6.check_disjoint("2025 cache index", {"KR_1"}, {"KR_2"})


# ----------------------------------------------------------------------------- build, audit and bank checks (PRESPEC 2.3, 9)

def test_any_build_failure_is_blocked():
    with pytest.raises(E6.Blocked, match="build failure"):
        E6.check_build_manifest({"summary": {"ok": 39999, "failed": 1}})
    assert E6.check_build_manifest({"summary": {"ok": 40000, "failed": 0}}) == 40000


def test_the_postbuild_check_must_pass():
    with pytest.raises(E6.Blocked, match="post-build"):
        E6.check_postbuild({"passed": False})
    E6.check_postbuild({"passed": True})


def test_audit_and_bank_exclusions_count_against_built_matches_apart_from_build_failures():
    assert E6.check_exclusions(400, 40000)["share"] == pytest.approx(0.01)       # 1 % is allowed
    with pytest.raises(E6.Blocked, match="1 %"):
        E6.check_exclusions(401, 40000)
    with pytest.raises(ValueError):
        E6.check_exclusions(0, 0)


# ----------------------------------------------------------------------------- guards and write once (PRESPEC 4, 9, 12)

def test_sealed_names_are_refused_for_reading_but_may_be_hashed(tmp_path):
    for name in ("KA_LABELS_E6_SEALED.npz", "EXTERNAL_LABELS_SEALED_kr_e6.npz", "x_sealed.json"):
        path = tmp_path / name
        path.write_bytes(b"abc")
        with pytest.raises(E6.SealedPathError):
            E6.guard_read(path)
        assert E6.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_the_svi_seal_and_pred_final_are_refused_always(tmp_path):
    for rel in ("EXTERNAL_LABELS_SEALED_kr_e6.npz", "PRED_final.npz", "PRED_final/part_00000.npz", "PRED_final_MANIFEST.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        with pytest.raises(E6.SealedPathError):
            E6.guard_read(path, sealed_ok=True)
        E6.sha256_file(path)                                           # byte hash only (Ruling 52)
    ok = tmp_path / "LABELS_PUBLIC_h90.npz"
    ok.write_bytes(b"y")
    assert E6.guard_read(ok) == ok


def test_outputs_are_written_once_with_lf_only(tmp_path):
    target = tmp_path / "a" / "X.json"
    sha = E6.write_json_once(target, {"b": 1, "a": [1.5, float("nan")], "c": np.int64(3)})
    raw = target.read_bytes()
    assert sha == hashlib.sha256(raw).hexdigest() and b"\r" not in raw and b"\t" not in raw
    assert json.loads(raw) == {"a": [1.5, None], "b": 1, "c": 3}
    with pytest.raises(FileExistsError):
        E6.write_json_once(target, {"b": 2})
    npz = tmp_path / "Y.npz"
    E6.write_npz_once(npz, {"v": np.arange(3)})
    with pytest.raises(FileExistsError):
        E6.write_npz_once(npz, {"v": np.arange(3)})


# ----------------------------------------------------------------------------- V scoring boundary (PRESPEC 4 GPU, 12)

def test_score_v_environment_is_the_allowlist_plus_the_fixed_values():
    base = {"SYSTEMROOT": r"C:\Windows", "PATH": "p", "TEMP": "t", "CONDA_PREFIX": "c", "PYTHONPATH": "leak",
            "CUDA_VISIBLE_DEVICES": "", "E6_STAGE": "history", "OMP_NUM_THREADS": "8", "HOME": "h", "LOL_OUTPUT_ROOT": "x",
            "USERNAME": "u", "USERDOMAIN": "leak"}
    env = E6.score_v_env(base)
    fixed = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "KMP_DUPLICATE_LIB_OK": "TRUE",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "E6_STAGE": "score_v"}
    assert env == {"SYSTEMROOT": r"C:\Windows", "PATH": "p", "TEMP": "t", "CONDA_PREFIX": "c", "USERNAME": "u", **fixed}
    # PRESPEC 4 list plus USERNAME (2026-10-14 운영 정정, Ruling 69): torch's getpass.getuser() in the frozen V path
    assert set(E6.SCORE_V_ALLOW) == {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP",
                                     "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
                                     "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "CONDA_PREFIX", "USERNAME"}


def _fake_torch(available=True, count=1, name="NVIDIA GeForce RTX 5080"):
    cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count, get_device_name=lambda i=0: name,
                           get_device_capability=lambda i=0: (12, 0))
    return SimpleNamespace(cuda=cuda, __version__="2.10.0+cu128", version=SimpleNamespace(cuda="12.8"),
                           backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)))


@pytest.mark.parametrize("kwargs,reason", [({"available": False}, "not available"), ({"count": 2}, "device count"),
                                           ({"name": "NVIDIA GeForce RTX 4090"}, "RTX 5080")])
def test_device_check_blocks_and_never_falls_back_to_cpu(kwargs, reason):
    with pytest.raises(E6.Blocked, match=reason) as info:
        E6.check_device(_fake_torch(**kwargs))
    assert info.value.record["passed"] is False and "cpu" not in json.dumps(info.value.record).lower()


def test_device_check_passes_on_one_visible_rtx_5080():
    record = E6.check_device(_fake_torch())
    assert record["passed"] is True and record["device_count"] == 1 and "RTX 5080" in record["device_name"]
    assert record["torch"] == "2.10.0+cu128" and record["cuda"] == "12.8"


def test_score_v_stage_does_not_score_when_the_device_check_fails(tmp_path, monkeypatch, clean_stage):
    calls = []
    monkeypatch.setattr(E6, "_import_torch", lambda: _fake_torch(available=False))
    monkeypatch.setattr(E6, "_frozen_run_scoring", lambda *a, **k: calls.append(a))
    paths = E6.Paths(out=tmp_path)
    code = E6.score_v_stage(paths, workers=1, require_committed=False)
    assert code == 2 and calls == []
    record = json.loads((tmp_path / "V_SCORING_DEVICE.json").read_text(encoding="utf-8"))
    assert record["passed"] is False
    assert list(tmp_path.glob("SCORE_V_BLOCKED_*.json"))


# ----------------------------------------------------------------------------- dataset registration and output binding (PRESPEC 4)

def test_registration_is_additive_and_idempotent(bound_state):
    E6.register_dataset()
    E6.register_dataset()
    assert C.DATASETS.count("kr_e6") == 1 and C.DATASETS[:3] == bound_state["C"]["DATASETS"][:3]
    assert C.EXPECTED_PATCHES["kr_e6"] == ("15.18", "15.19", "15.20", "15.21", "15.22")
    assert EXT.RAW["kr_e6"] == E6.RAW_ROOT
    assert C.EXPECTED_PATCHES["kr_15_18_22"] == bound_state["C_dicts"]["EXPECTED_PATCHES"]["kr_15_18_22"]


def test_bind_outputs_redirects_every_package_output_to_out(tmp_path, bound_state):
    rebound = E6.bind_outputs(tmp_path)
    assert EXT.REPORTS == tmp_path and EXT.LOGS == tmp_path / "logs" and EXT.WORK == tmp_path / "work"
    assert EXT.RUNTIME_REPO == tmp_path / "runtime_repo"
    assert EXT.SAMPLE_MANIFEST == tmp_path / "E6_SAMPLE_MANIFEST.json"
    assert EXT.SAMPLE_MANIFEST_SHA == tmp_path / "E6_SAMPLE_MANIFEST.sha256"
    assert EXT.CACHE_DIR["kr_e6"] == C.CACHE["kr_e6"] == tmp_path / "cache_kr_e6"
    assert EXT.PATCH_INDEX["kr_e6"] == C.PATCH_INDEX["kr_e6"] == tmp_path / "match_cache_kr_e6_patch_index.json"
    assert C.stage_dir("kr_e6", "history") == tmp_path / "ext" / "history" / "kr_e6"
    assert B.labels_dir("kr_e6") == tmp_path / "ext" / "labels" / "kr_e6"
    assert B.q_inputs_dir("kr_e6") == tmp_path / "ext" / "q_inputs" / "kr_e6"
    assert B.sealed_labels_path("kr_e6") == tmp_path / "ext" / "labels" / "kr_e6" / "EXTERNAL_LABELS_SEALED_kr_e6.npz"
    assert C.LOGS == tmp_path / "logs" and B.STAGE_TIMES.parent == tmp_path
    # the other datasets keep their frozen layout
    assert C.stage_dir("kr_15_18_22", "bank") == C.DATA_ROOT / "bank" / "kr_15_18_22"
    assert B.labels_dir("kr_15_18_22") == C.DATA_ROOT / "labels" / "kr_15_18_22"
    with pytest.raises(ValueError):
        C.stage_dir("kr_e6", "nonsense")
    assert {"stage_dir", "labels_dir", "RUNTIME_REPO", "ProcessPoolExecutor"} <= {r["name"] for r in rebound["rebindings"]}
    assert E6.bind_outputs(tmp_path)["out"] == str(tmp_path)          # idempotent


# ----------------------------------------------------------------------------- worker capture (PRESPEC 4 module records)

def _stage(tmp_path, name="audit"):
    return E6.begin_stage(E6.Paths(out=tmp_path), name)


def test_install_executor_replaces_both_names_and_the_frozen_check(clean_stage):
    import concurrent.futures as cf
    import concurrent.futures.process as cfp
    original = cfp.ProcessPoolExecutor
    E6.install_executor()
    assert cf.ProcessPoolExecutor is E6.E6Executor and cfp.ProcessPoolExecutor is E6.E6Executor
    assert issubclass(E6.E6Executor, E6._ORIGINAL_PPE) and E6._ORIGINAL_PPE is not E6.E6Executor
    good = SimpleNamespace(__name__="m1", ProcessPoolExecutor=E6.E6Executor)
    bad = SimpleNamespace(__name__="m2", ProcessPoolExecutor=E6._ORIGINAL_PPE)
    assert E6.check_frozen_executors([good]) == {"m1": True}
    with pytest.raises(E6.Blocked, match="m2"):
        E6.check_frozen_executors([good, bad])
    E6.restore_executor()
    assert cfp.ProcessPoolExecutor is (original if original is not E6.E6Executor else E6._ORIGINAL_PPE)


def test_executor_needs_a_stage_and_at_most_four_workers(tmp_path, clean_stage):
    with pytest.raises(RuntimeError, match="stage"):
        E6.E6Executor(max_workers=1)
    _stage(tmp_path)
    for n in (None, 5):
        with pytest.raises(E6.Blocked, match="4"):
            E6.E6Executor(max_workers=n)


def _init_mark(path, token):
    Path(path).write_text(token, encoding="utf-8")


def test_initializer_writes_a_start_marker_registers_finalize_100_then_runs_the_original(tmp_path, monkeypatch, clean_stage):
    seen = []
    monkeypatch.setattr(E6.multiprocessing.util, "Finalize",
                        lambda obj, cb, args=(), kwargs=None, exitpriority=None: seen.append((obj, cb, exitpriority)))
    monkeypatch.setenv("E6_STAGE", "bank")
    records = tmp_path / "worker_modules" / "bank"
    mark = tmp_path / "mark.txt"
    E6._worker_initializer("bank", "run1", 0, str(records), _init_mark, (str(mark), "tok"))
    try:
        assert seen == [(None, E6._write_worker_record, 100)]
        assert mark.read_text(encoding="utf-8") == "tok"                 # the original initializer with its own args
        start = json.loads((records / f"run1_0_{os.getpid()}.start.json").read_text(encoding="utf-8"))
        assert start["stage"] == "bank" and start["run_id"] == "run1" and start["executor_no"] == 0
        assert start["pid"] == os.getpid() and start["os_ppid"] == os.getppid()
        assert E6._E6Task(pow)(2, 5) == 32 and E6._E6Task(divmod)(7, 2) == (3, 1)
        E6._write_worker_record()
        record = json.loads((records / f"run1_0_{os.getpid()}.json").read_text(encoding="utf-8"))
        assert record["tasks"] == 2 and record["stage"] == "bank" and record["env_stage"] == "bank"
        assert any(p.endswith("e6_run.py") for p in record["modules"])
        assert all(len(h) == 64 for h in record["modules"].values())
    finally:
        E6._WORKER = None


def _double(x):
    return 2 * x


def _read_mark(path):
    return Path(path).read_text(encoding="utf-8")


@SPAWN
def test_e6_executor_end_to_end_records_reconcile(tmp_path, clean_stage):
    ctx = _stage(tmp_path, "select")
    mark = tmp_path / "spawn_mark.txt"
    with E6.E6Executor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                       initializer=_init_mark, initargs=(str(mark), "spawned")) as pool:
        assert list(pool.map(_double, [1, 2, 3])) == [2, 4, 6]
        assert pool.submit(_read_mark, str(mark)).result() == "spawned"
    assert ctx.executors[0]["submitted"] == 4
    assert ctx.records_dir == tmp_path / "worker_modules" / "select" / ctx.run_id     # a fresh subfolder per attempt
    runs = {ctx.run_id: {"stage": "select", "pid": os.getpid(), "executors": {0: 4}}}
    rec = E6.reconcile(tmp_path / "worker_modules", runs)
    assert rec["passed"] is True, rec["problems"]
    assert rec["start_markers"] == rec["termination_records"] == 1 and rec["tasks"] == {f"{ctx.run_id}:0": 4}
    assert all(p.parent == ctx.records_dir for p in sum(E6.worker_record_files(tmp_path / "worker_modules"), []))


def _write_pair(root, stage, run, no, pid, *, tasks=1, ppid=100, env_stage=None, end=True, nested=False, modules=None):
    folder = root / stage / run if nested else root / stage
    folder.mkdir(parents=True, exist_ok=True)
    ident = {"stage": stage, "run_id": run, "executor_no": no, "pid": pid, "os_ppid": ppid, "mp_parent_pid": ppid,
             "process_name": "SpawnProcess-1", "env_stage": stage if env_stage is None else env_stage}
    (folder / f"{run}_{no}_{pid}.start.json").write_text(json.dumps(ident), encoding="utf-8")
    if end:
        (folder / f"{run}_{no}_{pid}.json").write_text(json.dumps({**ident, "tasks": tasks, "modules": modules or {}}),
                                                       encoding="utf-8")



def test_reconcile_checks_module_versions_within_each_run_not_across_stages(tmp_path):
    """A stage's workers must agree on every module's version.  Stages may have run different committed versions of
    e6_run.py (the lock's drift check covers every other module), so the lock's reconciliation over the final attempts
    of all stages does not compare versions across runs (2026-10-14: select-audit ran 81fbb37, later stages 86e1541)."""
    root = tmp_path / "worker_modules"
    _write_pair(root, "audit", "r1", 0, 11, modules={"e6_run.py": "a", "frozen.py": "f"})
    _write_pair(root, "bank", "r2", 0, 21, ppid=200, modules={"e6_run.py": "b", "frozen.py": "f"})
    runs = {"r1": {"stage": "audit", "pid": 100, "executors": {0: 1}}, "r2": {"stage": "bank", "pid": 200, "executors": {0: 1}}}
    rec = E6.reconcile(root, runs)
    assert rec["passed"] is True, rec["problems"]
    assert E6.reconcile(root, runs, run_ids={"r1"})["modules"] == {"e6_run.py": "a", "frozen.py": "f"}
    assert E6.reconcile(root, runs, run_ids={"r2"})["modules"] == {"e6_run.py": "b", "frozen.py": "f"}
    _write_pair(root, "bank", "r2", 0, 22, ppid=200, modules={"e6_run.py": "c", "frozen.py": "f"})
    runs["r2"]["executors"] = {0: 2}
    rec = E6.reconcile(root, runs)
    assert rec["passed"] is False
    assert [p for p in rec["problems"] if "two versions" in p] == ["workers of run r2 loaded two versions of e6_run.py"]

def test_reconcile_scans_subfolders_and_matches_tasks_to_submissions(tmp_path):
    root = tmp_path / "worker_modules"
    _write_pair(root, "audit", "r1", 0, 11, tasks=3)
    _write_pair(root, "audit", "r1", 0, 12, tasks=2)
    _write_pair(root, "bank", "r2", 0, 21, tasks=4, ppid=200)
    runs = {"r1": {"stage": "audit", "pid": 100, "executors": {0: 5}}, "r2": {"stage": "bank", "pid": 200, "executors": {0: 4}}}
    rec = E6.reconcile(root, runs)
    assert rec["passed"] and rec["start_markers"] == 3 and rec["termination_records"] == 3
    starts, ends = E6.worker_record_files(root)
    assert len(starts) == 3 and len(ends) == 3 and all(p.name.endswith(".start.json") for p in starts)


@pytest.mark.parametrize("damage,needle", [
    ("missing_end", "termination"), ("stage_mismatch", "invalid"), ("ppid", "parent pid"), ("tasks", "submissions"),
    ("unknown_run", "no stage run"), ("corrupt", "invalid")])
def test_reconcile_blocks_every_mismatch(tmp_path, damage, needle):
    root = tmp_path / "worker_modules"
    _write_pair(root, "audit", "r1", 0, 11, tasks=3, end=damage != "missing_end",
                env_stage="bank" if damage == "stage_mismatch" else None, ppid=999 if damage == "ppid" else 100)
    runs = {"r1": {"stage": "audit", "pid": 100, "executors": {0: 4 if damage == "tasks" else 3}}}
    if damage == "unknown_run":
        runs = {}
    if damage == "corrupt":
        (root / "audit" / "r1_0_11.json").write_text("{not json", encoding="utf-8")
    rec = E6.reconcile(root, runs)
    assert rec["passed"] is False and any(needle in p for p in rec["problems"]), rec["problems"]


def test_reconcile_counts_an_executor_with_submissions_but_no_record(tmp_path):
    rec = E6.reconcile(tmp_path / "worker_modules", {"r9": {"stage": "items", "pid": 1, "executors": {0: 7}}})
    assert rec["passed"] is False and any("submissions" in p for p in rec["problems"])


def test_reconcile_accepts_per_attempt_subfolders_and_refuses_a_record_in_another_attempts_folder(tmp_path):
    root = tmp_path / "worker_modules"
    _write_pair(root, "history", "r7", 0, 31, tasks=2, nested=True)
    _write_pair(root, "audit", "r1", 0, 11, tasks=1)                              # the flat layout of earlier stages
    runs = {"r7": {"stage": "history", "pid": 100, "executors": {0: 2}}, "r1": {"stage": "audit", "pid": 100, "executors": {0: 1}}}
    assert E6.reconcile(root, runs)["passed"] is True
    other = root / "history" / "r8"
    other.mkdir()
    (root / "history" / "r7" / "r7_0_31.json").rename(other / "r7_0_31.json")
    rec = E6.reconcile(root, runs)
    assert rec["passed"] is False and any("folder" in p for p in rec["problems"])


def _run_record(out, stage, run, status, submitted, error=None):
    E6.write_json_once(out / "stage_runs" / stage / f"{run}.json",
                       {"run_id": run, "stage": stage, "pid": 100, "status": status, "error": error,
                        "executors": {"0": {"submitted": submitted}}})


def test_the_lock_reconciles_final_attempts_only_and_lists_superseded_ones(tmp_path):
    """Ruling 69: an attempt that died before the lock is superseded by a rerun; its records stay, are listed with
    their reason and are left out of reconciliation; only each stage's final attempt must reconcile."""
    out = tmp_path / "out"
    root = out / "worker_modules"
    _write_pair(root, "history", "old1", 0, 41, end=False, nested=True)          # host ended the job: no exit record
    _write_pair(root, "score_v", "old2", 0, 42, tasks=0, nested=True)            # broken pool: a submission never ran
    _run_record(out, "score_v", "old2", "failed", 1, error="BrokenProcessPool: a worker died")
    for stage, run, submitted in (("history", "new1", 3), ("score_v", "new2", 1)):
        _write_pair(root, stage, run, 0, 50 + submitted, tasks=submitted, nested=True)
        _run_record(out, stage, run, "completed", submitted)
        E6.write_json_once(out / f"STAGE_{stage}.json", {"status": "completed", "run_id": run})
    paths = E6.Paths(out=out)
    rec = E6.attempts_record(paths, ("history", "score_v"))
    assert rec["final"] == {"history": "new1", "score_v": "new2"}
    assert rec["reconciliation"]["passed"] is True, rec["reconciliation"]["problems"]
    sup = {s["run_id"]: s for s in rec["superseded"]}
    assert set(sup) == {"old1", "old2"}
    assert sup["old1"]["stage"] == "history" and sup["old1"]["status"] is None and "no stage run record" in sup["old1"]["reason"]
    assert sup["old2"]["status"] == "failed" and "BrokenProcessPool" in sup["old2"]["reason"]
    assert sup["old1"]["worker_records"] == ["worker_modules/history/old1/old1_0_41.start.json"]
    assert E6.reconcile(root, E6.load_stage_runs(paths))["passed"] is False      # all attempts together would not
    with pytest.raises(E6.Blocked):
        E6.attempts_record(paths, ("history", "score_v", "missing"))


def _git_commit(repo, text, when):
    import subprocess
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    path = repo / E6.CODE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    for args in (("add", E6.CODE_REL), ("commit", "-q", "-m", f"v {when}")):
        subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, env=env, capture_output=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True)
    return head.stdout.decode("ascii").strip(), hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_stage_code_versions_take_the_version_a_stage_ran_from_its_start_and_flag_a_file_edited_during_it(tmp_path):
    """run_stage starts a real stage only when e6_run.py equals HEAD, so a stage ran the last commit made before it
    started.  Module records hash files at the end of the stage; a file edited during the stage shows its end version
    there and the stage is listed (2026-10-14: bank ran 81fbb37 while the Ruling 69 edit, later 86e1541, was made)."""
    import subprocess
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    c1, v1 = _git_commit(repo, "version one\n", "2026-10-14T12:00:00+09:00")
    c2, v2 = _git_commit(repo, "version two\n", "2026-10-14T13:00:00+09:00")
    code_path = str((repo / E6.CODE_REL).resolve())
    starts = {"before": ("2026-10-14T03:30:00+00:00", v1), "edited": ("2026-10-14T03:50:00+00:00", v2),
              "after": ("2026-10-14T04:10:00+00:00", v2), "unknown": ("2026-10-14T04:20:00+00:00", "f" * 64),
              "early": ("2026-10-14T02:00:00+00:00", v1)}
    docs = {}
    for stage, (started, sha) in starts.items():
        E6.write_json_once(out / f"STAGE_{stage}.json", {"status": "completed", "run_id": stage, "started_utc": started})
        docs[stage] = {"stage_process_modules": {code_path: sha}}
    versions = E6.stage_code_versions(E6.Paths(out=out, repo=repo), docs)
    stages = versions["stages"]
    assert stages["before"]["commit_at_start"] == c1 and stages["before"]["record_commit"] == c1
    assert stages["before"]["record_equals_start"] is True
    assert stages["edited"]["commit_at_start"] == c1 and stages["edited"]["sha256_at_start"] == v1
    assert stages["edited"]["record_commit"] == c2 and stages["edited"]["record_equals_start"] is False
    assert stages["after"]["commit_at_start"] == c2 and stages["after"]["record_equals_start"] is True
    assert versions["edited_during_stage"] == ["edited", "unknown"]              # record differs from the start version
    assert versions["now_sha256"] == v2
    assert {p.split(":", 1)[0] for p in versions["problems"]} == {"unknown", "early"}   # not a commit; no commit before


# ----------------------------------------------------------------------------- acceptance rule (PRESPEC 4 module records)

def test_acceptance_rule_three_records_old_repo_thirty_parent_record_and_otherwise_blocked():
    ws, old = E6.ROOT, E6.OLD_REPO
    refs = E6.Refs(ws={"studies/a.py": "a" * 64}, old30={str(old / "core" / "x.py"): "b" * 64},
                   parent={str(old / "core" / "x.py"): "b" * 64, str(old / "core" / "y.py"): "c" * 64})
    good = {str(ws / "studies" / "a.py"): "a" * 64, str(ws / "studies" / "free.py"): "d" * 64,
            str(old / "core" / "x.py"): "b" * 64, str(old / "core" / "y.py"): "c" * 64}
    out = E6.acceptance(good, refs)
    assert out["passed"] is True
    rules = {Path(p).name: v["rule"] for p, v in out["files"].items()}
    assert rules == {"a.py": "three_records", "free.py": "workspace_not_in_records", "x.py": "old_repo_30",
                     "y.py": "parent_record"}
    for bad in ({str(ws / "studies" / "a.py"): "e" * 64}, {str(old / "core" / "x.py"): "e" * 64},
                {str(old / "core" / "y.py"): "e" * 64}, {str(old / "core" / "z.py"): "f" * 64}):
        assert E6.acceptance(bad, refs)["passed"] is False


def test_the_real_reference_records_have_the_prespec_sizes():
    refs = E6.load_refs(E6.Paths())
    assert len(refs.old30) == 30 and len(refs.parent) == 41
    assert refs.sizes == {"EXTERNAL_SCORE_LOCK.code.external": 18, "EXTERNAL_SCORE_LOCK.code.frozen_imported": 41,
                          "E2a_REPORT.code.package": 19, "E2a_REPORT.code.frozen_dependencies": 67,
                          "E1_REPORT.code.old_repo_builder_modules": 30}


# ============================================================================= phase 2: players, KA labels, evaluation, lock, opening

# ----------------------------------------------------------------------------- KA_v1 label function (PRESPEC 3, 8.1 A9)

def _ka_inputs(seed=3, matches=12):
    """A q-row table, its label table (shuffled, plus non-q rows) and cumulative team kills per query."""
    rng = np.random.default_rng(seed)
    per = rng.integers(1, 5, size=matches)
    match = np.repeat(np.arange(matches) * 5 + 2, per).astype(np.int64)
    n = match.size
    mkey = np.asarray([f"{m:064d}".encode() for m in match], dtype="S64")
    ckey = np.asarray([f"c{i:03d}".encode() for i in range(n)], dtype="S8")
    kills = np.zeros((4 * n + 10, 2), dtype=np.int32)
    pre = np.arange(n, dtype=np.int64) * 4 + 1
    post = pre + 2
    for i in range(n):
        base = rng.integers(0, 20, size=2)
        gain = rng.integers(0, 3, size=2)
        kills[pre[i]] = base
        kills[pre[i] + 1] = base
        kills[post[i]] = base + gain
    rows = {"canonical_match_key": mkey, "component_key": ckey, "query_pre": pre, "match_index": match,
            "engagement_key": np.char.add(np.char.add(mkey, b":"), ckey)}
    order = rng.permutation(n)
    extra = 3
    labels = {"canonical_match_key": np.concatenate([mkey[order], np.asarray([b"9" * 64] * extra, dtype="S64")]),
              "component_key": np.concatenate([ckey[order], np.asarray([b"zz"] * extra, dtype="S8")]),
              "query_pre": np.concatenate([pre[order], np.full(extra, 7)]),
              "query_post": np.concatenate([post[order], np.full(extra, 9)])}
    return rows, labels, kills


def _kill_label_reference(rows, labels, kills):
    """kill_label.cmd_prepare lines 124-141, verbatim on in-memory arrays."""
    index = {(m, c): i for i, (m, c) in enumerate(zip(labels["canonical_match_key"].tolist(), labels["component_key"].tolist()))}
    pick = np.asarray([index[(m, c)] for m, c in zip(rows["canonical_match_key"].tolist(), rows["component_key"].tolist())], dtype=np.int64)
    pre, post = np.asarray(labels["query_pre"], dtype=np.int64)[pick], np.asarray(labels["query_post"], dtype=np.int64)[pick]
    blue = np.asarray(kills[post, 0], dtype=np.int64) - np.asarray(kills[pre, 0], dtype=np.int64)
    red = np.asarray(kills[post, 1], dtype=np.int64) - np.asarray(kills[pre, 1], dtype=np.int64)
    ka = blue - red
    eligible = ka != 0
    y_ka = (ka > 0).astype(np.float64)
    match = np.asarray(rows["match_index"], dtype=np.int64)
    m = np.bincount(match[eligible], minlength=int(match.max()) + 1).astype(np.float64)
    weight = np.where(eligible, 1.0 / np.maximum(m[match], 1.0), 0.0)
    return {"engagement_key": rows["engagement_key"], "ka": ka.astype(np.int32), "blue_kills": blue.astype(np.int32),
            "red_kills": red.astype(np.int32), "eligible": eligible, "y_ka": y_ka, "weight_raw_ka": weight}


def test_ka_labels_equal_kill_label_prepare_with_window_kills_ties_and_match_weights():
    rows, labels, kills = _ka_inputs()
    out = E6.ka_labels(rows, labels, kills)
    ref = _kill_label_reference(rows, labels, kills)
    assert tuple(out) == E6.KA_COLUMNS
    for name in E6.KA_COLUMNS:
        assert out[name].dtype == ref[name].dtype and np.array_equal(out[name], ref[name]), name
    assert (out["ka"] == out["blue_kills"] - out["red_kills"]).all()
    assert np.array_equal(out["eligible"], out["ka"] != 0) and np.array_equal(out["y_ka"] == 1.0, out["ka"] > 0)
    assert (~out["eligible"]).any() and out["eligible"].any()                     # ties are present and kept apart
    assert (out["weight_raw_ka"][~out["eligible"]] == 0.0).all()
    match = rows["match_index"]
    for m in np.unique(match[out["eligible"]]):
        mask = (match == m) & out["eligible"]
        assert out["weight_raw_ka"][mask] == pytest.approx(np.full(mask.sum(), 1.0 / mask.sum()))


@pytest.mark.parametrize("damage,needle", [("pre", "pre query"), ("negative", "decrease"), ("key", "key"),
                                           ("ref", "query reference")])
def test_ka_labels_stop_on_pre_mismatch_negative_windows_missing_keys_and_bad_references(damage, needle):
    rows, labels, kills = _ka_inputs()
    if damage == "pre":
        rows["query_pre"] = rows["query_pre"].copy()
        rows["query_pre"][0] += 1
    elif damage == "negative":
        kills = kills.copy()
        i = int(np.flatnonzero(labels["component_key"] == rows["component_key"][0])[0])
        kills[labels["query_post"][i], 0] = kills[labels["query_pre"][i], 0] - 1
    elif damage == "key":
        rows["component_key"] = rows["component_key"].copy()
        rows["component_key"][1] = b"missing"
    else:
        labels["query_post"] = labels["query_post"].copy()
        labels["query_post"][:] = -1
    with pytest.raises(E6.Blocked, match=needle):
        E6.ka_labels(rows, labels, kills)


def test_gate_compare_requires_every_array_equal_on_the_kept_rows():
    rows, labels, kills = _ka_inputs()
    full = _kill_label_reference(rows, labels, kills)
    keep = np.ones(rows["match_index"].size, dtype=bool)
    keep[-2:] = False                                                            # e.g. TEST rows: not computed, not compared
    sub = {k: v[keep] for k, v in rows.items()}
    mine = E6.ka_labels(sub, labels, kills)
    ref = {k: v.copy() for k, v in full.items()}
    ref["y_ka"][-1] = 1.0 - ref["y_ka"][-1]                                      # a difference outside the kept rows
    assert E6.gate_compare(mine, ref, keep)["passed"] is True
    for name, change in (("ka", lambda a: a + (np.arange(a.size) == 0)), ("weight_raw_ka", lambda a: a * (1 + 1e-15) + 1e-17),
                         ("eligible", lambda a: a.astype(np.int8))):
        bad = {k: v.copy() for k, v in full.items()}
        bad[name] = change(bad[name])
        out = E6.gate_compare(mine, bad, keep)
        assert out["passed"] is False and out["arrays"][name] is False


# ----------------------------------------------------------------------------- player identification branches (PRESPEC 5.3)

def test_incomplete_identification_counts_every_match_below_ten_players():
    idx = np.arange(40, dtype=np.int64).reshape(4, 10)
    idx[1, 3] = -1                                       # puuid missing without a failure reason
    idx[2, :] = -1                                       # a failed match (k_m = 0)
    out = E6.incomplete_identification(idx, 4)
    assert out == {"available": True, "incomplete_matches": 2, "q_row_matches": 4, "share": 0.5}
    with pytest.raises(E6.Blocked):
        E6.incomplete_identification(idx, 5)              # the matrix must cover every q-row match


@pytest.mark.parametrize("block,branch,primary,reason", [
    ({"available": True, "incomplete_matches": 10, "q_row_matches": 1000, "share": 0.01}, "가", "twoway", None),
    ({"available": True, "incomplete_matches": 11, "q_row_matches": 1000, "share": 0.011}, "나", "match_bootstrap",
     "incomplete_identification_over_1pct"),
    ({"available": False, "reason": "player_identification_unavailable", "q_row_matches": 1000}, "다", "match_bootstrap",
     "player_identification_unavailable")])
def test_branch_rule(block, branch, primary, reason):
    out = E6.choose_branch(block)
    assert (out["branch"], out["primary_interval"], out["twoway_unavailable_reason"]) == (branch, primary, reason)
    assert out["incomplete_identification"] == block


def _players_tree(tmp_path, *, detail=True, missing_puuid=0, failed=0):
    """A small E6 OUT with a q-row table, a patch index and fake detail files (fake puuids, never real ones)."""
    out = tmp_path / "out"
    q = out / "ext" / "q_inputs" / "kr_e6"
    q.mkdir(parents=True)
    ids = [f"KR_{900 + i}" for i in range(30)]
    keys = [C.canonical_key(m).encode() for m in ids]
    match = np.repeat(np.arange(30, dtype=np.int64) * 3, 2)
    mk = np.asarray([keys[i // 2] for i in range(60)], dtype="S64")
    np.savez(q / "Q_ROWS_h90.npz", match_index=match, canonical_match_key=mk, patch=np.full(60, b"15.18", dtype="S8"),
             engagement_key=np.asarray([k + b":" + str(i).encode() for i, k in enumerate(mk)]))
    (out / "match_cache_kr_e6_patch_index.json").write_text(json.dumps({m: "15.18" for m in ids}), encoding="utf-8")
    E6.write_json_once(out / "STAGE_e2c.json", {"status": "completed", "note": "test predecessor"})
    raw = tmp_path / "raw"
    if detail:
        (raw / "detail").mkdir(parents=True)
        for i, m in enumerate(ids):
            parts = [{"participantId": j + 1, "puuid": f"fake-{(i * 3 + j) % 25:04d}-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"} for j in range(10)]
            if i < missing_puuid:
                parts[0].pop("puuid")
            if missing_puuid <= i < missing_puuid + failed:
                parts = parts[:9]
            (raw / "detail" / f"{m}.json").write_text(json.dumps({"metadata": {"matchId": m}, "info": {"participants": parts}}),
                                                       encoding="utf-8")
    return E6.Paths(out=out, raw=raw)


def test_players_stage_branch_a_writes_the_matrix_without_identifiers(tmp_path, clean_stage):
    paths = _players_tree(tmp_path)
    assert E6.players_stage(paths, require_committed=False) == 0
    doc = json.loads((paths.out / "PLAYERS_E6.json").read_text(encoding="utf-8"))
    assert doc["branch"]["branch"] == "가" and doc["branch"]["primary_interval"] == "twoway"
    assert doc["branch"]["incomplete_identification"]["incomplete_matches"] == 0
    with np.load(paths.out / "PLAYERS_E6.npz") as z:
        assert z["player_idx"].shape == (30, 10) and z["player_idx"].dtype == np.int64 and (z["player_idx"] >= 0).all()
    for name in ("PLAYERS_E6.json", "PLAYERS_E6.npz", "STAGE_players.json"):
        assert b"fake-" not in (paths.out / name).read_bytes()


def test_players_stage_branch_b_counts_missing_puuids_and_failed_matches(tmp_path, clean_stage):
    paths = _players_tree(tmp_path, missing_puuid=1, failed=1)                  # 2 of 30 matches incomplete
    assert E6.players_stage(paths, require_committed=False) == 0
    doc = json.loads((paths.out / "PLAYERS_E6.json").read_text(encoding="utf-8"))
    assert doc["branch"]["branch"] == "나" and doc["branch"]["twoway_unavailable_reason"] == "incomplete_identification_over_1pct"
    assert doc["branch"]["incomplete_identification"] == {"available": True, "incomplete_matches": 2, "q_row_matches": 30,
                                                          "share": 2 / 30}


def test_players_stage_branch_c_only_when_the_detail_folder_is_absent(tmp_path, clean_stage, monkeypatch):
    paths = _players_tree(tmp_path, detail=False)
    called = []
    monkeypatch.setattr(E6, "_e3_build", lambda *a, **k: called.append(1))
    assert E6.players_stage(paths, require_committed=False) == 0
    doc = json.loads((paths.out / "PLAYERS_E6_UNAVAILABLE.json").read_text(encoding="utf-8"))
    assert called == [] and not (paths.out / "PLAYERS_E6.npz").exists()
    assert doc["reason"] == "player_identification_unavailable" and doc["detail_folder"]["exists"] is False
    assert doc["incomplete_identification"] == {"available": False, "reason": "player_identification_unavailable",
                                                "q_row_matches": 30}


def test_players_stage_blocks_on_missing_mandatory_inputs_or_a_raising_build(tmp_path, clean_stage, monkeypatch):
    paths = _players_tree(tmp_path)
    (paths.out / "match_cache_kr_e6_patch_index.json").unlink()
    assert E6.players_stage(paths, require_committed=False) == 2
    paths2 = _players_tree(tmp_path / "b")

    def boom(*a, **k):
        raise ValueError("a match_index of the row table carries two canonical keys")

    monkeypatch.setattr(E6, "_e3_build", boom)
    assert E6.players_stage(paths2, require_committed=False) == 2
    assert not (paths2.out / "PLAYERS_E6_UNAVAILABLE.json").exists()


def test_a_rerun_moves_a_crashed_attempts_outputs_aside_and_starts_clean(tmp_path, clean_stage):
    """Ruling 69: outputs left by an attempt that died before its stage record are kept (moved, never read or
    changed) under OUT/superseded/<stage>/<new run>/ and the new attempt starts from scratch."""
    paths = _players_tree(tmp_path)
    (paths.out / "PLAYERS_E6.npz").write_bytes(b"partial output of a crashed attempt")
    assert E6.players_stage(paths, require_committed=False) == 0
    moved = list((paths.out / "superseded" / "players").rglob("PLAYERS_E6.npz"))
    assert len(moved) == 1 and moved[0].read_bytes() == b"partial output of a crashed attempt"
    with np.load(paths.out / "PLAYERS_E6.npz") as z:
        assert z["player_idx"].shape == (30, 10)
    record = json.loads((paths.out / "STAGE_players.json").read_text(encoding="utf-8"))
    assert [Path(e["to"]).name for e in record["superseded_outputs"]] == ["PLAYERS_E6.npz"]
    assert record["superseded_outputs"][0]["sha256"] == hashlib.sha256(b"partial output of a crashed attempt").hexdigest()
    assert E6.STAGE_OUTPUTS["labels"] == ("KA_GATE.json", "KA_LABELS_E6_SEALED.npz", "KA_LABELS_E6_MANIFEST.json")


# ----------------------------------------------------------------------------- evaluation and decision (PRESPEC 5-7)

def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _eval_data(seed=5, matches=80):
    rng = np.random.default_rng(seed)
    per = rng.integers(1, 6, size=matches)
    match = np.repeat(np.arange(matches) * 4 + 1, per).astype(np.int64)
    n = match.size
    y = rng.integers(0, 2, size=n).astype(np.float64)
    preds = {key: _sig((0.3 + 0.1 * (i % 3)) * (2 * y - 1) + rng.normal(0, 1, n)) for i, key in enumerate(E6.CELLS)}
    kills = rng.integers(1, 7, size=n).astype(np.int16)
    players = np.asarray([rng.permutation(120)[:10] for _ in range(matches)], dtype=np.int64)
    return match, y, preds, kills, players


@pytest.mark.parametrize("interval,decision", [((-0.03, -0.01), "confirmed"), ((-0.01, 0.0), "unresolved"),
                                               ((-0.01, 0.02), "unresolved"), ((0.0, 0.01), "unresolved"),
                                               ((0.001, 0.02), "opposite")])
def test_decision_rule_on_the_primary_interval(interval, decision):
    assert E6.decide(interval) == decision


def test_evaluate_branch_a_primary_is_the_twoway_interval_of_boot():
    from studies.tog_experiments_20261012 import boot
    from studies.q_production_20261005 import qeval

    match, y, preds, kills, players = _eval_data()
    unique = np.unique(match)
    pin = E6.PlayerInput(available=True, reason=None, match_index=unique, player_idx=players, record=None)
    out = E6.evaluate(y, preds, match, kills, kills, pin, s_scored=True)
    losses = {k: qeval.per_match_loss(y, preds[k], match) for k in (E6.P1, E6.PT)}
    shared, aligned = qeval.align_cells(losses)
    D = aligned[E6.P1] - aligned[E6.PT]
    tw = boot.twoway_interval(D, players, 0.95)
    assert out["primary_interval"] == "twoway"
    assert out["primary"]["interval"] == list(tw["interval"]) and out["twoway"]["available"] is True
    assert out["twoway"]["V1"] == tw["V1"] and out["twoway"]["VU"] == tw["VU"] and out["twoway"]["V2"] == tw["V2"]
    assert out["decision"] == E6.decide(tw["interval"])
    boot_p1 = qeval.paired_match_bootstrap(shared, aligned, {"P1": {E6.P1: 1.0, E6.PT: -1.0}}, levels=0.95)["contrasts"]["P1"]
    assert out["bootstrap"]["interval"] == boot_p1["interval"] and out["theta_hat"] == boot_p1["point"]
    assert out["secondary"]["a_position_share"]["scored"] is True and "twoway" in out["secondary"]["a_position_share"]


def test_evaluate_without_players_has_no_twoway_values_and_bootstrap_primary():
    match, y, preds, kills, _ = _eval_data()
    for reason in ("incomplete_identification_over_1pct", "player_identification_unavailable", "no_player_ids"):
        out = E6.evaluate(y, preds, match, kills, kills, E6.PlayerInput(False, reason, None, None, None), s_scored=False)
        assert out["primary_interval"] == "match_bootstrap" and out["primary"]["interval"] == out["bootstrap"]["interval"]
        assert out["twoway"] == {"available": False, "reason": reason}
        assert out["secondary"]["a_position_share"]["scored"] is False
        assert "twoway" not in out["secondary"]["a_position_share"]


def test_evaluate_reproduces_the_ka_report_block_of_kill_label():
    from reports.q_production_20261005 import kill_label as KL

    match, y, preds, kills, _ = _eval_data(seed=11, matches=150)
    out = E6.evaluate(y, preds, match, kills, kills, E6.PlayerInput(False, "no_player_ids", None, None, None), s_scored=True)
    block = KL._contrast_block(y, {k: preds[k] for k in KL.CELLS}, match)
    for cell in KL.CELLS:
        assert out["cells"][cell]["auc"] == block["metrics"][cell]["auc"]
        assert out["cells"][cell]["brier"] == block["metrics"][cell]["brier"]
    for name, mine in (("P1", out["bootstrap"]), ("S_effect_meanD", out["secondary"]["a_position_share"]["bootstrap"])):
        ref = block["contrasts"]["contrasts"][name]
        assert mine["point"] == ref["point"] and mine["interval"] == ref["interval"]
    assert E6.CELLS == tuple(KL.CELLS) and E6.S_CONTRAST == KL.CONTRASTS["S_effect_meanD"]


def test_kill_strata_split_non_tie_rows_and_count_ties_apart():
    match, y, preds, kills, _ = _eval_data(seed=2)
    all_kills = np.concatenate([kills, np.asarray([1, 2, 5, 5], dtype=np.int16)])   # four tie rows outside the evaluation
    out = E6.evaluate(y, preds, match, kills, all_kills, E6.PlayerInput(False, "no_player_ids", None, None, None),
                      s_scored=False)["secondary"]["c_kill_strata"]["strata"]
    strata = np.where(kills == 1, "1", np.where(kills <= 3, "2-3", "4+"))
    for s, ties in (("1", 1), ("2-3", 1), ("4+", 2)):
        assert out[s]["rows"] == int((strata == s).sum()) and out[s]["ties"] == ties
        assert out[s]["matches"] == int(np.unique(match[strata == s]).size)
    with pytest.raises(E6.Blocked):
        E6.evaluate(y, preds, match, np.zeros_like(kills), kills, E6.PlayerInput(False, "no_player_ids", None, None, None),
                    s_scored=False)


# ----------------------------------------------------------------------------- synthetic rehearsal labels and the label loader (A10, B2)

def test_synthetic_labels_are_deterministic_and_in_the_sealed_format():
    keys = np.asarray([f"k{i}".encode() for i in range(5000)])
    match = np.repeat(np.arange(1000), 5).astype(np.int64)
    a, b = E6.synthetic_labels(keys, match), E6.synthetic_labels(keys, match)
    assert tuple(a) == E6.KA_COLUMNS
    for name in E6.KA_COLUMNS:
        assert np.array_equal(a[name], b[name])
    u = np.random.Generator(np.random.PCG64(20261014)).random((5000, 2))
    assert np.array_equal(~a["eligible"], u[:, 0] < 0.098) and np.array_equal(a["y_ka"] == 1.0, (u[:, 1] < 0.5) & a["eligible"])
    assert a["ka"].dtype == np.int32 and a["y_ka"].dtype == np.float64 and a["eligible"].dtype == np.bool_


def test_the_label_loader_refuses_the_sealed_file_until_the_opening_is_saved(tmp_path):
    keys = np.asarray([f"k{i}".encode() for i in range(50)])
    match = np.repeat(np.arange(10), 5).astype(np.int64)
    labels = E6.synthetic_labels(keys, match)
    paths = E6.Paths(out=tmp_path / "out", openings=tmp_path / "OPENINGS.json")
    sealed = paths.out / "KA_LABELS_E6_SEALED.npz"
    E6.write_npz_once(sealed, labels)
    rows = {"engagement_key": keys, "match_index": match}
    entry = {"experiment": "E6", "attempt": 1, "utc": "t", "lock_sha256": "a" * 64}
    with pytest.raises(E6.LabelAccessError):
        E6.load_ka_labels(sealed, rows, paths=paths, opening=entry)
    E6.append_opening(paths.openings, entry)
    got = E6.load_ka_labels(sealed, rows, paths=paths, opening=entry)
    assert np.array_equal(got["idx"], np.flatnonzero(labels["eligible"])) and got["ties"] == int((~labels["eligible"]).sum())
    synth = paths.out / "KA_LABELS_E6_SYNTH.npz"
    E6.write_npz_once(synth, labels)
    assert np.array_equal(E6.load_ka_labels(synth, rows, paths=paths)["y"], labels["y_ka"][labels["eligible"]])


# ----------------------------------------------------------------------------- lock enumeration (A12) and the opening order (B0-B2)

def _touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_lock_enumerates_worker_records_recursively_branch_files_and_hash_only_labels(tmp_path):
    paths = E6.Paths(out=tmp_path / "out", training=tmp_path / "tr")
    out = paths.out
    recs = [_touch(out / "worker_modules" / st / f"r_{i}_{i}{suf}.json") for i, st in enumerate(("audit", "history"))
            for suf in (".start", "")]
    nested = _touch(out / "worker_modules" / "bank" / "deeper" / "r_0_9.start.json")
    sealed = [_touch(out / "KA_LABELS_E6_SEALED.npz"), _touch(paths.ext("labels") / "EXTERNAL_LABELS_SEALED_kr_e6.npz"),
              _touch(paths.ext("labels") / "PRED_final.npz"), _touch(paths.ext("labels") / "PRED_final" / "part_00000.npz"),
              _touch(paths.ext("labels") / "PRED_final" / "part_00000.json")]
    for st in ("select", "build"):
        _touch(out / f"STAGE_MODULES_{st}.json")
    files_a = E6.lock_files(paths, "가")
    files_c = E6.lock_files(paths, "다")
    worker_paths = {Path(p) for k, p in files_a.items() if k.startswith("worker_record:")}
    assert worker_paths == set(recs) | {nested}
    hash_only = {Path(p) for k, p in files_a.items() if k.startswith("labels_hash_only:")}
    assert hash_only == set(sealed)
    assert {Path(p).name for k, p in files_a.items() if k.startswith("stage_modules:")} == {"STAGE_MODULES_select.json",
                                                                                          "STAGE_MODULES_build.json"}
    assert "input:PLAYERS_E6.npz" in files_a and "input:PLAYERS_E6.json" in files_a
    assert "input:PLAYERS_E6_UNAVAILABLE.json" in files_c and "input:PLAYERS_E6.npz" not in files_c
    assert {"provenance:EXTERNAL_SCORE_LOCK.json", "provenance:E2a_REPORT.json", "provenance:E1_REPORT.json",
            "provenance:parent_run.json"} <= set(files_a)
    assert sum(k.startswith("old_repo_builder:") for k in files_a) == 30
    verified = E6.verify_entries({k: p for k, p in files_a.items() if k.startswith("labels_hash_only:")})
    assert all(len(v["sha256"]) == 64 for v in verified.values())


def _opening_tree(tmp_path):
    paths = E6.Paths(out=tmp_path / "out", openings=tmp_path / "OPENINGS.json", exposure=tmp_path / "EXPOSURE.md")
    keys = np.asarray([f"k{i}".encode() for i in range(60)])
    match = np.repeat(np.arange(12), 5).astype(np.int64)
    E6.write_npz_once(paths.out / "KA_LABELS_E6_SEALED.npz", E6.synthetic_labels(keys, match))
    lock_sha = E6.write_json_once(paths.out / "E6_LOCK.json", {"prespec": {"blob_sha256": "b" * 64}, "experiment": "E6"})
    paths.exposure.write_text(f"row {lock_sha}\n", encoding="utf-8")
    return paths, keys, match, lock_sha


def _deps(paths, keys, match, events, *, problems=None, equal=True):
    def load_labels(p, inputs, opening):
        on_disk = json.loads(p.openings.read_text(encoding="utf-8"))["openings"] if p.openings.exists() else []
        events.append(("load_labels", len(on_disk)))
        return E6.load_ka_labels(p.out / "KA_LABELS_E6_SEALED.npz", {"engagement_key": keys, "match_index": match},
                                 paths=p, opening=opening)

    return E6.TestDeps(
        warm_up=lambda: events.append(("warm_up",)),
        verify_lock=lambda lock, sha, p: (events.append(("verify_lock",)), problems or {})[1],
        preflight=lambda p, lock: (events.append(("preflight",)), {"rows": len(keys)})[1],
        reproduce=lambda p, lock, inputs: (events.append(("reproduce",)), {"equal": equal})[1],
        modules_outside_lock=lambda lock: {},
        load_labels=load_labels,
        analyse=lambda p, lock, inputs, labels: (events.append(("analyse",)), {"theta_hat": 0.0, "rows": int(labels["idx"].size)})[1])


def test_opening_order_checks_then_saved_entry_then_labels_then_report_once(tmp_path):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    events = []
    assert E6.test_stage(paths, deps=_deps(paths, keys, match, events), time_limit_s=None) == 0
    names = [e[0] for e in events]
    assert names == ["warm_up", "verify_lock", "preflight", "reproduce", "load_labels", "analyse"]
    assert ("load_labels", 1) in events                                    # the OPENINGS entry was on disk before the read
    openings = json.loads(paths.openings.read_text(encoding="utf-8"))["openings"]
    assert len(openings) == 1 and openings[0]["experiment"] == "E6" and openings[0]["lock_sha256"] == lock_sha
    assert openings[0]["prespec_sha256"] == "b" * 64 and openings[0]["attempt"] == 1
    assert json.loads((paths.out / "E6_OPENING.json").read_text(encoding="utf-8"))["lock_sha256"] == lock_sha
    report = json.loads((paths.out / "E6_REPORT.json").read_text(encoding="utf-8"))
    assert report["opening"]["lock_sha256"] == lock_sha and report["result"]["rows"] > 0
    events2 = []
    assert E6.test_stage(paths, deps=_deps(paths, keys, match, events2), time_limit_s=None) == 2   # same lock: refused
    assert ("load_labels", 1) not in events2 and len(json.loads(paths.openings.read_text(encoding="utf-8"))["openings"]) == 1


@pytest.mark.parametrize("failure", ["lock", "reproduce", "exposure_row"])
def test_failures_before_the_opening_leave_no_entry_and_read_no_label(tmp_path, failure):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    if failure == "exposure_row":
        paths.exposure.write_text("no row\n", encoding="utf-8")
    events = []
    deps = _deps(paths, keys, match, events, problems={"changed": True} if failure == "lock" else None,
                 equal=failure != "reproduce")
    if failure == "exposure_row":
        deps = E6.TestDeps(**{**deps.__dict__, "verify_lock": lambda lock, sha, p: E6.exposure_problems(sha, p)})
    assert E6.test_stage(paths, deps=deps, time_limit_s=None) == 2
    assert not paths.openings.exists() and "load_labels" not in [e[0] for e in events]
    assert list(paths.out.glob("TEST_BLOCKED_*.json")) and not (paths.out / "E6_REPORT.json").exists()


# ----------------------------------------------------------------------------- part B after the OPENINGS entry (fix round 1)
# The steps after the entry is saved are the real ones here (E6_OPENING.json write, watchdog thread, failure records);
# only the label-free checks before the opening and the analysis are callbacks.

def _failure_records(paths):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(paths.out.glob("E6_FAILED_*.json"))]


def _openings_on_disk(paths):
    return json.loads(paths.openings.read_text(encoding="utf-8"))["openings"]


def _fds_are_files():
    try:
        return all(stat.S_ISREG(os.fstat(fd).st_mode) for fd in (1, 2))
    except OSError:
        return False


def _with_analyse(deps, analyse):
    return E6.TestDeps(**{**deps.__dict__, "analyse": analyse})


def test_a_write_failure_after_the_opening_is_recorded_and_the_opening_kept(tmp_path):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    E6.write_json_once(paths.out / "E6_OPENING.json", {"left": "by an earlier run"})   # the write after the entry fails
    events = []
    assert E6.test_stage(paths, deps=_deps(paths, keys, match, events), time_limit_s=None) == 3
    assert len(_openings_on_disk(paths)) == 1                                          # the opening stays
    assert "load_labels" not in [e[0] for e in events] and not (paths.out / "E6_REPORT.json").exists()
    (record,) = _failure_records(paths)
    assert record["status"] == "failed_after_exposure" and record["phase"] == "writing E6_OPENING.json"
    exp = record["exposure"]
    assert exp["opening_persisted"] == "yes" and exp["labels_read"] == "no"
    assert exp["label_derived_values_in_files"] == "no" and exp["label_derived_values_on_screen"] == "no"
    assert record["error_type"] == "FileExistsError" and "E6_OPENING.json" in record["error"]   # label-free, so kept
    assert "label_values_printed" not in record and "single invocation" in record["ledger_writer"]


def test_a_watchdog_that_cannot_start_stops_the_run_before_the_opening(tmp_path, monkeypatch):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)

    class NoThread(E6.threading.Thread):
        def start(self):
            raise RuntimeError("cannot start a new thread")

    monkeypatch.setattr(E6.threading, "Thread", NoThread)
    events = []
    assert E6.test_stage(paths, deps=_deps(paths, keys, match, events), time_limit_s=30) == 2
    assert not paths.openings.exists() and "load_labels" not in [e[0] for e in events]
    (blocked,) = [json.loads(p.read_text(encoding="utf-8")) for p in paths.out.glob("TEST_BLOCKED_*.json")]
    assert "watchdog" in blocked["reason"] and blocked["before_exposure"] is True
    assert not list(paths.out.glob("E6_FAILED_*.json"))


def test_the_time_limit_after_the_labels_are_read_records_the_exposure_and_ends_the_run(tmp_path, monkeypatch, capfd):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    exits = []
    monkeypatch.setattr(E6.os, "_exit", lambda code: exits.append(code))          # the watchdog's hard exit, recorded
    events = []

    def slow_analyse(p, lock, inputs, labels):
        events.append(("analyse",))
        deadline = time.monotonic() + 20
        while not exits and time.monotonic() < deadline:
            time.sleep(0.05)
        return {"theta_hat": 0.0, "rows": 1}

    deps = _with_analyse(_deps(paths, keys, match, events), slow_analyse)
    assert E6.test_stage(paths, deps=deps, time_limit_s=0.5) == 3
    assert exits == [3] and not (paths.out / "E6_REPORT.json").exists()
    assert len(_openings_on_disk(paths)) == 1
    (record,) = _failure_records(paths)
    assert record["reason"] == "part B time limit" and record["phase"] == "analysing"
    exp = record["exposure"]
    assert exp["opening_persisted"] == "yes" and exp["labels_read"] == "yes" and exp["report_file"] == "not written"
    assert exp["label_derived_values_in_files"] == "no"
    assert exp["label_derived_values_on_screen"] == ("no" if _fds_are_files() else "unknown")


def test_an_exception_after_the_labels_are_read_keeps_label_derived_text_off_the_record_and_the_screen(tmp_path, capfd):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    events = []

    def failing_analyse(p, lock, inputs, labels):
        print("derived 0.123")
        raise ValueError("derived 0.456")

    deps = _with_analyse(_deps(paths, keys, match, events), failing_analyse)
    assert E6.test_stage(paths, deps=deps, time_limit_s=None) == 3
    shown = capfd.readouterr()
    assert "0.123" not in shown.out + shown.err and "0.456" not in shown.out + shown.err
    (path,) = sorted(paths.out.glob("E6_FAILED_*.json"))
    text = path.read_text(encoding="utf-8")
    assert "0.123" not in text and "0.456" not in text and "label_values_printed" not in text
    record = json.loads(text)
    exp = record["exposure"]
    assert record["error_type"] == "ValueError" and exp["labels_read"] == "yes" and len(_openings_on_disk(paths)) == 1
    sealed = {Path(f["file"]).name: f["status"] for f in exp["sealed_files"]}
    assert len(sealed) == 2 and all("SEALED" in name for name in sealed) and set(sealed.values()) == {"unknown"}
    assert all((paths.out / name).exists() for name in sealed)
    with pytest.raises(E6.SealedPathError):
        E6.guard_read(paths.out / next(iter(sealed)))
    assert exp["label_derived_values_in_files"] == "unknown"
    assert exp["label_derived_values_on_screen"] == ("no" if _fds_are_files() else "unknown")
    assert not (paths.out / "E6_REPORT.json").exists()


def test_a_failure_while_saving_the_opening_after_it_reached_disk_is_recorded(tmp_path, monkeypatch):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)

    def broken_check(p, opening):
        raise OSError("the check after the replace failed")

    monkeypatch.setattr(E6, "opening_saved", broken_check)
    events = []
    assert E6.test_stage(paths, deps=_deps(paths, keys, match, events), time_limit_s=None) == 3
    assert len(_openings_on_disk(paths)) == 1 and "load_labels" not in [e[0] for e in events]
    (record,) = _failure_records(paths)
    assert record["phase"] == "saving the OPENINGS entry" and record["exposure"]["opening_persisted"] == "yes"
    assert record["exposure"]["labels_read"] == "no"


def test_a_failure_before_the_entry_reaches_disk_blocks_without_exposure(tmp_path, monkeypatch):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)

    def no_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(E6.os, "replace", no_replace)
    events = []
    assert E6.test_stage(paths, deps=_deps(paths, keys, match, events), time_limit_s=None) == 2
    monkeypatch.undo()
    assert not paths.openings.exists() and "load_labels" not in [e[0] for e in events]
    assert list(paths.out.glob("TEST_BLOCKED_*.json")) and not list(paths.out.glob("E6_FAILED_*.json"))


def test_the_failure_record_falls_back_to_a_fixed_line_when_it_cannot_be_written(tmp_path, monkeypatch, capfd):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    real = E6.write_json_once

    def no_failure_records(path, payload, list_key=None):
        if Path(path).name.startswith("E6_FAILED"):
            raise OSError("disk full")
        return real(path, payload, list_key)

    monkeypatch.setattr(E6, "write_json_once", no_failure_records)
    events = []

    def failing_analyse(p, lock, inputs, labels):
        raise ValueError("derived 0.789")

    deps = _with_analyse(_deps(paths, keys, match, events), failing_analyse)
    assert E6.test_stage(paths, deps=deps, time_limit_s=None) == 3
    err = capfd.readouterr().err
    assert "E6 TEST FAILED AFTER THE OPENING" in err and "OSError" in err and "0.789" not in err
    assert len(_openings_on_disk(paths)) == 1


# ----------------------------------------------------------------------------- lock versions (fix round 1)

def test_lock_step_writes_a_new_version_only_when_the_committed_code_differs(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    code, test = _touch(repo / E6.CODE_REL, b"code v1"), _touch(repo / E6.TEST_REL, b"test v1")
    paths = E6.Paths(out=out, repo=repo, openings=tmp_path / "OPENINGS.json")
    assert E6.lock_step(paths) == "lock"
    verify = {f"code:{E6.CODE_REL}": {"sha256": E6.sha256_file(code)}, f"code:{E6.TEST_REL}": {"sha256": E6.sha256_file(test)}}
    r0 = E6.write_json_once(out / "E6_LOCK.json", {"verify": verify})
    assert E6.lock_step(paths) is None
    code.write_bytes(b"code v2")
    assert E6.lock_step(paths) == "lock_r1"
    E6.write_json_once(out / "E6_LOCK_r1.json", {"verify": {**verify, f"code:{E6.CODE_REL}": {"sha256": E6.sha256_file(code)}}})
    assert E6.lock_step(paths) is None
    test.write_bytes(b"test v2")
    assert E6.lock_step(paths) == "lock_r2"
    E6.append_opening(paths.openings, {"experiment": "E6", "attempt": 1, "utc": "t", "lock_sha256": r0})
    with pytest.raises(E6.Blocked):                                    # Ruling 31: no new lock after an opening
        E6.lock_step(paths)


def test_earlier_lock_versions_are_listed_with_whether_an_opening_used_them(tmp_path):
    paths = E6.Paths(out=tmp_path / "out", openings=tmp_path / "OPENINGS.json")
    r0 = E6.write_json_once(paths.out / "E6_LOCK.json", {"verify": {}})
    assert E6.previous_locks_record(paths) == [{"path": str(paths.out / "E6_LOCK.json"), "sha256": r0, "opened": False,
                                                "superseded_before_any_look": True}]
    E6.append_opening(paths.openings, {"experiment": "E6", "attempt": 1, "utc": "t", "lock_sha256": r0})
    assert E6.previous_locks_record(paths)[0]["opened"] is True


def test_attempts_record_names_an_earlier_lock_run_as_an_earlier_lock_version(tmp_path):
    out = tmp_path / "out"
    _run_record(out, "lock", "L0", "completed", 0)
    _run_record(out, "lock", "L1", "blocked", 0, error="the final attempts do not reconcile")
    E6.write_json_once(out / "STAGE_history.json", {"status": "completed", "run_id": "h1"})
    _run_record(out, "history", "h1", "completed", 0)
    rec = E6.attempts_record(E6.Paths(out=out), ("history",))
    reasons = {s["run_id"]: s["reason"] for s in rec["superseded"]}
    assert "earlier lock version" in reasons["L0"] and reasons["L1"].startswith("blocked")


def test_the_lock_binds_every_census_after_version(tmp_path):
    paths = E6.Paths(out=tmp_path / "out", training=tmp_path / "tr")
    first, second = _touch(paths.out / "ENV_CENSUS_after.json"), _touch(paths.out / "ENV_CENSUS_after_r1.json")
    files = E6.lock_files(paths, "가")
    assert files["input:ENV_CENSUS_after.json"] == first and files["input:ENV_CENSUS_after_r1.json"] == second


# ----------------------------------------------------------------------------- the recorder itself (fix round 2)

def _all_record_text(paths):
    return "".join(p.read_text(encoding="utf-8") for p in sorted(paths.out.glob("*.json")))


def _waiting_analyse(exits, events, *, native=None):
    def analyse(p, lock, inputs, labels):
        events.append(("analyse",))
        if native is not None:
            os.write(1, native)                                        # output that bypasses Python (native code)
        deadline = time.monotonic() + 20
        while not exits and time.monotonic() < deadline:
            time.sleep(0.05)
        return {"theta_hat": 0.0, "rows": 1}

    return analyse


def test_native_output_after_the_labels_are_read_makes_file_exposure_unknown_on_a_time_limit(tmp_path, monkeypatch, capfd):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    exits, events = [], []
    monkeypatch.setattr(E6.os, "_exit", lambda code: exits.append(code))
    deps = _with_analyse(_deps(paths, keys, match, events), _waiting_analyse(exits, events, native=b"native 0.5\n"))
    assert E6.test_stage(paths, deps=deps, time_limit_s=0.5) == 3
    assert exits == [3]
    (record,) = _failure_records(paths)
    exp = record["exposure"]
    assert record["reason"] == "part B time limit" and exp["labels_read"] == "yes" and exp["sealed_files"] == []
    assert exp["label_derived_values_in_files"] == "unknown"                 # the redirected stdout file grew
    assert exp["label_derived_values_on_screen"] == "unknown"


@pytest.mark.parametrize("broken", ["_exposure", "next_free"])
def test_a_recorder_preparation_failure_after_an_exception_leaves_a_minimal_record_and_no_label_text(
        tmp_path, monkeypatch, capfd, broken):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("metadata access denied")

    monkeypatch.setattr(E6, broken, denied)
    events = []

    def failing_analyse(p, lock, inputs, labels):
        raise ValueError("derived 0.321")

    deps = _with_analyse(_deps(paths, keys, match, events), failing_analyse)
    assert E6.test_stage(paths, deps=deps, time_limit_s=None) == 3              # nothing escapes, no chained text
    shown = capfd.readouterr()
    assert "0.321" not in shown.out + shown.err and "0.321" not in _all_record_text(paths)
    (record,) = _failure_records(paths)
    assert record["minimal"] is True and record["recorder_error_type"] == "PermissionError"
    assert record["error_type"] == "ValueError" and record["status"] == "failed_after_exposure"
    exp = record["exposure"]
    assert exp["labels_read"] == "yes" and exp["label_derived_values_in_files"] == "unknown"
    assert exp["label_derived_values_on_screen"] == "unknown" and len(_openings_on_disk(paths)) == 1


def test_a_recorder_preparation_failure_on_the_time_limit_still_leaves_a_record(tmp_path, monkeypatch, capfd):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    exits, events = [], []
    monkeypatch.setattr(E6.os, "_exit", lambda code: exits.append(code))

    def denied(*args, **kwargs):
        raise PermissionError("metadata access denied")

    monkeypatch.setattr(E6, "_exposure", denied)
    deps = _with_analyse(_deps(paths, keys, match, events), _waiting_analyse(exits, events))
    assert E6.test_stage(paths, deps=deps, time_limit_s=0.5) == 3
    assert exits == [3]
    (record,) = _failure_records(paths)
    assert record["minimal"] is True and record["reason"] == "part B time limit"
    assert record["recorder_error_type"] == "PermissionError" and record["exposure"]["labels_read"] == "yes"


def test_when_the_minimal_record_also_fails_a_fixed_line_reaches_stderr(tmp_path, monkeypatch, capfd):
    paths, keys, match, lock_sha = _opening_tree(tmp_path)
    real = E6.write_json_once

    def denied(*args, **kwargs):
        raise PermissionError("metadata access denied")

    def no_failure_records(path, payload, list_key=None):
        if Path(path).name.startswith("E6_FAILED"):
            raise OSError("disk full")
        return real(path, payload, list_key)

    monkeypatch.setattr(E6, "_exposure", denied)
    monkeypatch.setattr(E6, "write_json_once", no_failure_records)
    events = []

    def failing_analyse(p, lock, inputs, labels):
        raise ValueError("derived 0.654")

    deps = _with_analyse(_deps(paths, keys, match, events), failing_analyse)
    assert E6.test_stage(paths, deps=deps, time_limit_s=None) == 3
    err = capfd.readouterr().err
    assert "E6 TEST FAILED AFTER THE OPENING" in err and "PermissionError" in err and "OSError" in err
    assert "0.654" not in err and not _failure_records(paths) and len(_openings_on_disk(paths)) == 1
