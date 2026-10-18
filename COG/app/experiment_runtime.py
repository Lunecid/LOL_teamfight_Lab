from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

from app.experiment_exec_helpers import (
    build_experiment_args as _build_experiment_args,
    find_unsupported_overlay_flags as _find_unsupported_overlay_flags,
    parse_latest_run_result as _parse_latest_run_result,
    pick_run_dir as _pick_run_dir,
)
from app.experiment_types import ExperimentResult
from train.speed_config import apply_speed_overlay


def apply_config_overlay(cfg_obj: Any, overlay: Dict[str, Any]) -> None:
    """Apply treatment-specific config values to cfg object."""
    for key, value in overlay.items():
        try:
            setattr(cfg_obj, key, value)
        except AttributeError:
            print(f"[WARN] Config attribute '{key}' does not exist, adding dynamically")
            setattr(cfg_obj, key, value)


def reset_config_to_baseline(cfg_obj: Any) -> None:
    """Reset all ablation flags to baseline defaults."""
    baseline_flags = {
        "USE_FOCAL_LOSS": False,
        "FOCAL_GAMMA": 2.0,
        "FOCAL_ALPHA": 0.25,
        "USE_GAME_PHASE": False,
        "GAME_PHASE_TAU": 3.0,
        "USE_ATTENTION_POOL": False,
        "ATTENTION_POOL_DIM": 64,
        "USE_MOMENTUM_FEATURES": False,
        "MOMENTUM_K_SHORT": 3,
        "USE_ROLE_AWARE_ADJ": False,
        "ROLE_ADJ_INIT": 0.0,
        "USE_MULTI_TASK": False,
        "MTL_LAMBDA_GOLD": 0.1,
        "MTL_LAMBDA_KILL": 0.05,
        "MTL_LAMBDA_OBJ": 0.05,
        "LABEL_SMOOTHING": 0.0,
        # Interpolation / detector defaults (for non-leaky ablations)
        "BIN_MS": 5000,
        "DETECT_STEP_MS": 10000,
        "INTERP_XY": True,
        "INTERP_SCALARS_METHOD": "ffill",
        "TF2_GRID_STEP_MS": 5000,
        "TF2_USE_FRAME_INTERP": True,
        "TF2_USE_KILL_TRAJECTORY_INTERP": True,
    }
    apply_config_overlay(cfg_obj, baseline_flags)


def run_single_experiment(
    treatment_overlay: Dict[str, Any],
    seed: int,
    feature_set: str = "full",
    split_mode: str = "patch_holdout",
    experiment_tag: str = "",
) -> ExperimentResult:
    """Run one experiment and parse metrics from the produced run directory."""
    from app.experiment import run as run_experiment
    from core.config import RUN_DIR, cfg
    from core.utils import set_seed

    print(f"\n    [EXEC] {experiment_tag} | seed={seed} | overlay={treatment_overlay}")
    t0 = time.time()

    reset_config_to_baseline(cfg)

    _speed_env = str(os.environ.get("LOL_SPEED_OVERLAY", "1")).strip().lower()
    _use_speed = _speed_env not in ("0", "false", "off", "no")
    _vram_gb = float(os.environ.get("LOL_VRAM_GB", "24.0"))
    _speed_profile_raw = str(os.environ.get("LOL_SPEED_PROFILE", "auto" if _use_speed else "none")).strip().lower()
    _speed_profile = "none" if _speed_profile_raw in ("", "off") else _speed_profile_raw
    if _use_speed:
        apply_speed_overlay(cfg, vram_gb=_vram_gb)
        cfg.SPEED_PROFILE = _speed_profile
        try:
            from train.speed import apply_speed_profile as _apply_runtime_speed_profile

            _applied = "none" if _speed_profile == "none" else _apply_runtime_speed_profile(cfg, profile=_speed_profile)
        except Exception:
            _applied = "none"
        print(
            "    [SPEED] enabled "
            f"(vram={_vram_gb:.1f}GB, profile_req={_speed_profile}, profile_applied={_applied}, "
            f"batch={getattr(cfg, 'BATCH_SIZE', '?')}, "
            f"amp={getattr(cfg, 'AMP', False)}, "
            f"compile={getattr(cfg, 'TORCH_COMPILE', False)}, "
            f"cache_train={getattr(cfg, 'CACHE_TRAIN_SAMPLES_IN_RAM', False)})"
        )
        if _speed_profile != "none" and _applied == "none":
            print("    [SPEED] runtime profile fallback: overlay-only (likely CUDA unavailable for auto profile)")
    else:
        cfg.SPEED_PROFILE = "none"
        print("    [SPEED] disabled")

    try:
        from train.models import reset_model_singletons

        reset_model_singletons()
    except ImportError:
        pass

    apply_config_overlay(cfg, treatment_overlay)

    unsupported = _find_unsupported_overlay_flags(treatment_overlay)
    if unsupported:
        msg = " | ".join(unsupported)
        print(f"    [ERROR] Unsupported overlay: {msg}")
        return ExperimentResult(
            treatment_id=-1,
            treatment_name=experiment_tag,
            seed=seed,
            hp_config=treatment_overlay,
            train_time_sec=time.time() - t0,
        )

    set_seed(seed)

    args = _build_experiment_args(
        feature_set=feature_set,
        seed=seed,
        split_mode=split_mode,
    )

    model_list = list(getattr(cfg, "MODEL_LIST", []))
    args.model_list = model_list

    run_dirs_before: set[str] = set()
    try:
        run_dirs_before = {d.name for d in RUN_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")}
    except Exception:
        run_dirs_before = set()

    try:
        run_experiment(args)
    except Exception as e:
        print(f"    [ERROR] Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return ExperimentResult(
            treatment_id=-1,
            treatment_name=experiment_tag,
            seed=seed,
            hp_config=treatment_overlay,
            train_time_sec=time.time() - t0,
        )

    run_dir_hint = _pick_run_dir(
        run_root=RUN_DIR,
        seed=seed,
        before_run_names=run_dirs_before,
        started_at=t0,
    )
    result = _parse_latest_run_result(
        experiment_tag=experiment_tag,
        seed=seed,
        hp_config=treatment_overlay,
        run_dir_hint=run_dir_hint,
        preferred_models=model_list,
    )
    result.train_time_sec = time.time() - t0

    cnt_msg = ""
    if result.n_train >= 0 and result.n_val >= 0 and result.n_test >= 0:
        cnt_msg = f" n(train/val/test)={result.n_train}/{result.n_val}/{result.n_test}"
    print(
        f"    [DONE] val_auc={result.val_auc:.4f} test_auc={result.test_auc:.4f} "
        f"time={result.train_time_sec:.1f}s{cnt_msg}"
    )

    # [MEM] 실험 완료 후 전체 메모리 해제 (RAM 캐시 + GC + CUDA)
    from core.memory import clear_all
    clear_all()

    return result


def run_single_experiment_isolated(
    treatment_overlay: Dict[str, Any],
    seed: int,
    feature_set: str = "full",
    split_mode: str = "patch_holdout",
    experiment_tag: str = "",
) -> ExperimentResult:
    """별도 프로세스에서 실험을 실행하여 메모리 격리를 보장한다.

    수학적 보장:
        ∀ i ≠ j: AddressSpace(M_i) ∩ AddressSpace(M_j) = ∅

    프로세스 종료 시 OS가 RAM을 100% 회수하므로 메모리 누수 원천 차단.
    """
    import pathlib

    print(f"\n    [EXEC-ISOLATED] {experiment_tag} | seed={seed} | overlay={treatment_overlay}")
    t0 = time.time()

    # 결과를 받을 임시 파일
    result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="lol_exp_")
    os.close(result_fd)

    # 환경 변수 전파 (speed overlay 등)
    env = os.environ.copy()

    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve().parent.parent / "train_model_subprocess.py"),
        "--mode", "experiment",
        "--feature_set", feature_set,
        "--seed", str(seed),
        "--split_mode", split_mode,
        "--experiment_tag", experiment_tag,
        "--treatment_overlay", json.dumps(treatment_overlay, default=str),
        "--result_file", result_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # 실시간 출력 확인
            timeout=None,         # 시간 제한 없음 (모델 훈련은 오래 걸릴 수 있음)
        )
        # 프로세스 종료 → OS가 RAM 완전 회수

        result_data = {}
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)

        if proc.returncode != 0 or not result_data.get("_subprocess_ok", False):
            error_msg = result_data.get("_error", f"exit code {proc.returncode}")
            print(f"    [ERROR-ISOLATED] {error_msg}")
            return ExperimentResult(
                treatment_id=-1,
                treatment_name=experiment_tag,
                seed=seed,
                hp_config=treatment_overlay,
                train_time_sec=time.time() - t0,
            )

        # Reconstruct ExperimentResult from dict
        result = ExperimentResult(
            treatment_id=int(result_data.get("treatment_id", -1)),
            treatment_name=str(result_data.get("treatment_name", experiment_tag)),
            seed=seed,
            hp_config=treatment_overlay,
        )
        for fld in (
            "train_auc", "val_auc", "test_auc",
            "train_ap", "val_ap", "test_ap",
            "train_f1", "val_f1", "test_f1",
            "val_brier", "val_ece", "test_brier", "test_ece",
            "best_epoch", "n_train", "n_val", "n_test", "n_fights_all",
        ):
            if fld in result_data:
                setattr(result, fld, type(getattr(result, fld))(result_data[fld]))

        result.train_time_sec = time.time() - t0

        cnt_msg = ""
        if result.n_train >= 0 and result.n_val >= 0 and result.n_test >= 0:
            cnt_msg = f" n(train/val/test)={result.n_train}/{result.n_val}/{result.n_test}"
        print(
            f"    [DONE-ISOLATED] val_auc={result.val_auc:.4f} test_auc={result.test_auc:.4f} "
            f"time={result.train_time_sec:.1f}s{cnt_msg}"
        )
        return result

    except Exception as e:
        print(f"    [ERROR-ISOLATED] Subprocess failed: {e}")
        import traceback
        traceback.print_exc()
        return ExperimentResult(
            treatment_id=-1,
            treatment_name=experiment_tag,
            seed=seed,
            hp_config=treatment_overlay,
            train_time_sec=time.time() - t0,
        )
    finally:
        # 임시 결과 파일 정리
        try:
            os.unlink(result_path)
        except OSError:
            pass
