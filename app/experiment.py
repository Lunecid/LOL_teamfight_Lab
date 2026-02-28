from __future__ import annotations

import random
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.experiment_helpers import (
    LGBM_MAX_TRAIN as _LGBM_MAX_TRAIN,
    best_variant_for as _best_variant_for,
    infer_feature_set_for_model as _infer_feature_set_for_model,
    infer_rnn_gnn_models,
    needs_lgbm_logit as _needs_lgbm_logit,
    normalize_split_mode as _normalize_split_mode,
    resolve_alias as _resolve_alias,
    subsample_refs_for_deep as _subsample_refs_for_deep,
    subsample_refs_for_lgbm as _subsample_refs_for_lgbm,
)
from app.split_reports import emit_split_reports as _emit_split_reports
from core.config import CACHE_DIR, RUN_DIR, cfg
from core.fight_types import ref_key
from core.common import parse_csv_nums, parse_csv_str
from core.utils import save_csv_rows, save_json, set_seed, write_log
from data.cache_io import build_match_pairs, prebuild_cache
from data.file_io import dump_fight_refs_csv, ensure_dir, now_tag
from data.index_split import build_fight_index, split_refs
from data.indexing import (
    check_split_leakage,
    count_patches_from_refs,
    filter_loadable_refs,
    log_patch_block,
    scan_cache_match_ids,
    scan_cache_match_patch_counts,
    split_by_match_id_kfold,
    split_refs_patch_holdout,
)
from data.labels import get_label_map
from train.baseline import densify_logit_map, run_lgbm_baseline
from data.dataset import InMemoryFightDataset, LogitInjectorDataset
from train.deep import train_deep_model
from train.fusion import (
    stack_oof_meta,
    stack_simple,
    stack_factorial,
    refit_meta_trainval_predict_test,
    calibrate_logits_by_patch,
    split_logit_map_by_refs,
)
from train.speed import apply_speed_profile, setup_torch_speed
from core.memory import clear_memory

def run(args) -> None:
    """Run the pipeline according to args/cfg.

    The pipeline enforces the time contract:
    - FightRef.t_start is a minute index.
    """

    feature_set = str(args.feature_set)
    seed = int(args.seed)
    set_seed(seed)

    # R sweep list
    r_sweep = parse_csv_nums(getattr(args, "r_core_sweep", ""), cast=float)
    if not r_sweep:
        r_sweep = [float(getattr(args, "r_core", getattr(cfg, "STANDOFF_RADIUS", 1800.0)))]

    # Model list (runner passes already-resolved list)
    model_list: List[str] = list(getattr(args, "model_list", []))
    model_list = [_resolve_alias(m) for m in model_list if str(m).strip()]

    # fusion request tokens
    fusion_requests = [m for m in model_list if (m or "").lower().startswith("fusion_")]
    fusion_auto_best = any((m or "").lower() == "fusion_auto_best" for m in fusion_requests)
    if fusion_requests:
        model_list = [m for m in model_list if not (m or "").lower().startswith("fusion_")]

    ablation_mode = str(getattr(args, "ablation_mode", getattr(cfg, "ABLATION_MODE", "baseline_plus")))
    # Policy: baseline logit is injected only in baseline_plus mode.
    use_logit_inputs = (ablation_mode == "baseline_plus")
    require_lgbm = bool(getattr(args, "require_lgbm", False) or getattr(cfg, "REQUIRE_LGBM_FOR_ABLATION", False))

    enable_factorial = not bool(getattr(args, "no_factorial_fusion", False))
    stacking_mode = str(getattr(args, "stacking_mode", getattr(cfg, "STACKING_MODE", "simple"))).lower()

    # Infer rnn/gnn names (overridable)
    rnn_name, gnn_name = infer_rnn_gnn_models(model_list)

    rnn_arg = str(getattr(args, "rnn_model", "")).strip()
    gnn_arg = str(getattr(args, "gnn_model", "")).strip()

    if rnn_arg:
        rnn_name = _resolve_alias(rnn_arg)
    if gnn_arg:
        gnn_name = _resolve_alias(gnn_arg)

    # Split settings
    split_mode = _normalize_split_mode(getattr(args, "split_mode", getattr(cfg, "SPLIT_MODE", "auto")))
    cfg.SPLIT_MODE = split_mode
    train_patches = parse_csv_str(getattr(args, "train_patches", ""))
    test_patches = parse_csv_str(getattr(args, "test_patches", ""))
    val_patches = parse_csv_str(getattr(args, "val_patches", ""))
    val_ratio = float(getattr(args, "val_ratio", getattr(cfg, "VAL_RATIO", 0.15)))

    sweep_tag = f"sweep_{now_tag()}__fs={feature_set}__seed={seed}" if len(r_sweep) > 1 else None
    sweep_root = ensure_dir(RUN_DIR / sweep_tag) if sweep_tag else RUN_DIR

    cache_pc = scan_cache_match_patch_counts(max_matches=int(cfg.MAX_MATCHES) if cfg.MAX_MATCHES else None)

    # 1) Cache build (once)
    if str(cfg.MODE) in ("build_cache", "all"):
        log_fp = (sweep_root / "build_cache.log") if sweep_tag else (RUN_DIR / f"build_cache_{now_tag()}.log")
        write_log("[STEP] build_cache", log_fp)
        pairs = build_match_pairs(cfg.DETAIL_DIR, cfg.TIMELINE_DIR)
        write_log(f"[CACHE] found pairs={len(pairs)}", log_fp)
        prebuild_cache(pairs, log_fp=log_fp)
        if str(cfg.MODE) == "build_cache":
            return

    # baseline source fixed to lgbm (no XGB)
    try:
        setattr(cfg, "BASE_LOGIT_SOURCE", "lgbm")
    except Exception:
        pass

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(cfg, "USE_CUDA", True) else "cpu")

    # Sweep loop
    sweep_rows: List[dict] = []

    for R in r_sweep:
        try:
            setattr(cfg, "STANDOFF_RADIUS", float(R))
        except Exception:
            pass

        run_tag = f"run_{now_tag()}__R={int(R)}__fs={feature_set}__seed={seed}"
        run_root = ensure_dir(sweep_root / run_tag) if sweep_tag else ensure_dir(RUN_DIR / run_tag)
        run_log = run_root / "run.log"
        meta_root = ensure_dir(run_root / "meta")
        models_root = ensure_dir(run_root / "models")

        write_log(f"[RUN] {run_tag}", run_log)
        write_log(f"[CFG] MODE={cfg.MODE} MAX_MATCHES={cfg.MAX_MATCHES}", run_log)
        write_log(f"[CFG] ABLATION_MODE={ablation_mode} REQUIRE_LGBM={require_lgbm} USE_LOGIT_INPUTS={use_logit_inputs}", run_log)
        write_log(f"[CFG] STANDOFF_RADIUS(R_core)={getattr(cfg,'STANDOFF_RADIUS',None)}", run_log)
        write_log(f"[CFG] PREDICTION_GAP_MS={int(getattr(cfg, 'PREDICTION_GAP_MS', 0))}", run_log)
        write_log(f"[CFG] SPLIT_MODE={split_mode}", run_log)
        write_log(f"[CFG] FACTORIAL_FUSION={enable_factorial} stacking_mode={stacking_mode} rnn={rnn_name} gnn={gnn_name}", run_log)
        if fusion_requests:
            write_log(f"[CFG] FUSION_REQUESTS={fusion_requests} fusion_auto_best={fusion_auto_best}", run_log)
        write_log(f"[CFG] RUN_ROOT={run_root}", run_log)
        log_patch_block("CACHE matches", cache_pc, run_log)

        speed_profile = str(getattr(cfg, "SPEED_PROFILE", "none"))
        if speed_profile.lower() not in ("none", "off", ""):
            applied = apply_speed_profile(cfg, profile=speed_profile, log_fp=run_log)
            write_log(f"[CFG] SPEED_PROFILE={speed_profile} applied={applied}", run_log)

        setup_torch_speed(cfg, log_fp=run_log)

        # 2) Build fight index (depends on R)
        write_log("[STEP] build_fight_index", run_log)

        max_matches_opt = int(cfg.MAX_MATCHES) if cfg.MAX_MATCHES else None
        cache_match_ids = scan_cache_match_ids(max_matches=max_matches_opt)

        try:
            refs = build_fight_index(cache_match_ids, max_matches=max_matches_opt, tag=run_tag)
        except TypeError:
            try:
                refs = build_fight_index(cache_match_ids, max_matches=max_matches_opt)
            except TypeError:
                refs = build_fight_index(cache_match_ids)

        if not refs:
            write_log("[FATAL] No fights found. Check cache / detection rules.", run_log)
            continue

        fight_pc_all = count_patches_from_refs(refs)
        log_patch_block("FIGHTS(all refs)", fight_pc_all, run_log)

        save_json(
            meta_root / "fight_index.json",
            {
                "n_fights": len(refs),
                "patch_counts_fights": fight_pc_all,
                "patch_counts_cache_matches": cache_pc,
                "cache_dir": str(CACHE_DIR),
                "feature_set": feature_set,
                "R_core": float(R),
                "t_start_unit": "minute_idx",
                "t_start_contract": "t_start is minute index into cache['minute_ts']",
            },
        )

        try:
            dump_fight_refs_csv(meta_root / "fight_refs_all.csv", refs, split="all")
        except Exception as e:
            write_log(f"[WARN] dump_fight_refs_csv(all) failed: {e}", run_log)

        if str(cfg.MODE) == "index":
            continue

        # 3) Split
        if split_mode == "patch_holdout":
            tr_refs, va_refs, te_refs, split_info = split_refs_patch_holdout(
                refs=refs,
                seed=seed,
                train_patches=train_patches if train_patches else None,
                test_patches=test_patches if test_patches else None,
                val_patches=val_patches if val_patches else None,
                val_ratio_from_train=val_ratio,
                log_fp=run_log,
            )
        else:
            tr_refs, va_refs, te_refs, split_info = split_refs(refs, mode=split_mode, seed=seed)

        log_patch_block("FIGHTS(train refs)", count_patches_from_refs(tr_refs), run_log)
        log_patch_block("FIGHTS(val refs)", count_patches_from_refs(va_refs), run_log)
        log_patch_block("FIGHTS(test refs)", count_patches_from_refs(te_refs), run_log)

        check_split_leakage(
            tr_refs,
            va_refs,
            te_refs,
            run_log,
            fail_on_leakage=not bool(getattr(args, "allow_split_leakage", False)),
        )
        write_log(f"[SPLIT] train={len(tr_refs)} val={len(va_refs)} test={len(te_refs)} info={split_info}", run_log)

        try:
            dump_fight_refs_csv(meta_root / "fight_refs_train.csv", tr_refs, split="train")
            dump_fight_refs_csv(meta_root / "fight_refs_val.csv", va_refs, split="val")
            dump_fight_refs_csv(meta_root / "fight_refs_test.csv", te_refs, split="test")
        except Exception as e:
            write_log(f"[WARN] dump_fight_refs_csv(split) failed: {e}", run_log)

        # Optional: filter loadable refs
        if bool(getattr(args, "filter_loadable", False)) or bool(getattr(cfg, "FILTER_LOADABLE_REFS", False)):
            write_log("[STEP] filter_loadable_refs", run_log)
            tr_refs, pc_tr_used = filter_loadable_refs(tr_refs, feature_set=feature_set, tag="train", log_fp=run_log)
            va_refs, pc_va_used = filter_loadable_refs(va_refs, feature_set=feature_set, tag="val", log_fp=run_log)
            te_refs, pc_te_used = filter_loadable_refs(te_refs, feature_set=feature_set, tag="test", log_fp=run_log)
        else:
            pc_tr_used, pc_va_used, pc_te_used = {}, {}, {}

        save_json(
            meta_root / "split.json",
            {
                "split_info": split_info,
                "n_train": len(tr_refs),
                "n_val": len(va_refs),
                "n_test": len(te_refs),
                "patch_counts_fights_refs": {
                    "all": count_patches_from_refs(refs),
                    "train": count_patches_from_refs(tr_refs),
                    "val": count_patches_from_refs(va_refs),
                    "test": count_patches_from_refs(te_refs),
                },
                "patch_counts_fights_loadable": {
                    "train": pc_tr_used,
                    "val": pc_va_used,
                    "test": pc_te_used,
                },
                "patch_counts_cache_matches": cache_pc,
                "R_core": float(R),
                "t_start_unit": "minute_idx",
            },
        )

        if str(cfg.MODE) == "report":
            write_log("[MODE] report -> wrote index/split meta. (No training.)", run_log)
            continue

        # 4) Baselines + Deep
        results_rows: List[dict] = []
        deep_reports: Dict[str, Any] = {}
        deep_pred_maps: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}

        need_baseline_logits = (
            use_logit_inputs
            or any((m or "").lower() == "lgbm" for m in model_list)
            or bool(getattr(cfg, "RUN_LGBM_BASELINE", False))
        )

        lgbm_pack = None
        lgbm_logit_map: Dict[str, float] = {}
        if need_baseline_logits:
            lgbm_dir = ensure_dir(models_root / "lgbm" / "baseline")
            lgbm_log = lgbm_dir / "run.log"
            write_log(f"[STEP] LGBM baseline (need_baseline_logits={need_baseline_logits})", run_log)

            # Ã¢â€â‚¬Ã¢â€â‚¬ Memory-safe: subsample train refs for LGBM Ã¢â€â‚¬Ã¢â€â‚¬
            lgbm_max_train = int(getattr(cfg, "LGBM_MAX_TRAIN", _LGBM_MAX_TRAIN))
            tr_refs_lgbm = _subsample_refs_for_lgbm(
                tr_refs, max_n=lgbm_max_train, seed=seed, log_fp=run_log,
            )
            write_log(
                f"[LGBM] train refs: {len(tr_refs):,} -> {len(tr_refs_lgbm):,} "
                f"(val={len(va_refs):,}, test={len(te_refs):,})",
                run_log,
            )
            lgbm_pack = run_lgbm_baseline(feature_set, tr_refs_lgbm, va_refs, te_refs, seed, lgbm_log, out_dir=lgbm_dir)
            del tr_refs_lgbm
            clear_memory(log_fp=run_log)  # [MEM] LGBM 훈련 후 메모리 해제
            if lgbm_pack and lgbm_pack.get("ok"):
                lgbm_logit_map = dict(lgbm_pack.get("logit_map", {}))
                met = lgbm_pack.get("metrics", {})
                results_rows.append(
                    {
                        "run_tag": run_tag,
                        "R_core": float(R),
                        "model": "lgbm",
                        "variant": "baseline",
                        "feature_set": feature_set,
                        "seed": seed,
                        "tr_auc": met.get("train", {}).get("auc"),
                        "va_auc": met.get("val", {}).get("auc"),
                        "te_auc": met.get("test", {}).get("auc"),
                        "tr_acc": met.get("train", {}).get("acc"),
                        "va_acc": met.get("val", {}).get("acc"),
                        "te_acc": met.get("test", {}).get("acc"),
                        "artifact_dir": str(lgbm_dir),
                        "checkpoint": lgbm_pack.get("model_path"),
                    }
                )
        else:
            write_log("[STEP] LGBM baseline skipped (not needed)", run_log)

        if require_lgbm and enable_factorial and (not lgbm_pack or not lgbm_pack.get("ok")):
            write_log("[FATAL] Fusion requires baseline but LGBM baseline not available -> stop this R run.", run_log)
            continue

        base_logit_map = lgbm_logit_map
        all_split_refs = list(tr_refs) + list(va_refs) + list(te_refs)
        base_logit_map_full = densify_logit_map(all_split_refs, base_logit_map, default_logit=0.0) if base_logit_map else {}

        # ─── Dataset sharing: build once, reuse across models ────────
        share_datasets = bool(getattr(args, "share_datasets", False))
        _shared_base_datasets = None  # (ds_tr, ds_va, ds_te) without logits

        if share_datasets:
            _deep_max = int(getattr(cfg, "DEEP_MAX_TRAIN", 200_000))
            _tr_refs_shared = _subsample_refs_for_deep(tr_refs, _deep_max, seed, log_fp=run_log)
            cache_train = bool(getattr(cfg, "CACHE_TRAIN_SAMPLES_IN_RAM", getattr(cfg, "CACHE_IN_RAM", False)))
            cache_eval = bool(getattr(cfg, "CACHE_EVAL_SAMPLES_IN_RAM", True))

            write_log(
                f"[SHARE_DS] Building shared base datasets once "
                f"(train={len(_tr_refs_shared):,}, val={len(va_refs):,}, test={len(te_refs):,})",
                run_log,
            )
            _t0_share = __import__("time").time()

            _ds_base_tr = InMemoryFightDataset(
                _tr_refs_shared,
                feature_set=feature_set,
                model_name="__shared_base__",
                lgbm_logit_map=None,
                cache_in_ram=cache_train,
                force_emit_logits=False,
                split_label="train",
            )
            _ds_base_va = InMemoryFightDataset(
                va_refs,
                feature_set=feature_set,
                model_name="__shared_base__",
                lgbm_logit_map=None,
                cache_in_ram=cache_eval,
                force_emit_logits=False,
                split_label="val",
            )
            _ds_base_te = InMemoryFightDataset(
                te_refs,
                feature_set=feature_set,
                model_name="__shared_base__",
                lgbm_logit_map=None,
                cache_in_ram=cache_eval,
                force_emit_logits=False,
                split_label="test",
            )
            _shared_base_datasets = (_ds_base_tr, _ds_base_va, _ds_base_te)

            _elapsed_share = __import__("time").time() - _t0_share
            write_log(
                f"[SHARE_DS] Done in {_elapsed_share:.1f}s — "
                f"train={len(_ds_base_tr)}, val={len(_ds_base_va)}, test={len(_ds_base_te)}",
                run_log,
            )

        # 5) Deep models
        for model_name in model_list:
            if (model_name or "").lower() == "lgbm":
                continue

            fs = _infer_feature_set_for_model(model_name, feature_set)
            model_logit_capable = _needs_lgbm_logit(model_name)

            if (not use_logit_inputs) and model_logit_capable:
                write_log(
                    f"[SKIP] {model_name} expects baseline logits, but logit inputs are disabled "
                    f"(enable with --ablation_mode baseline_plus).",
                    run_log,
                )
                continue

            if use_logit_inputs and (not base_logit_map_full):
                msg = f"[SKIP] {model_name} needs baseline-plus inputs but baseline logits are unavailable."
                if require_lgbm:
                    write_log("[FATAL] " + msg, run_log)
                    break
                write_log(msg, run_log)
                continue

            if ablation_mode == "as_is":
                need_logit = bool(use_logit_inputs and model_logit_capable)
                logit_map = base_logit_map_full if need_logit else None
                variant_tag = "default"

                model_dir = ensure_dir(models_root / model_name / variant_tag)
                model_log = model_dir / "run.log"
                write_log(f"[MODEL] {model_name}/{variant_tag} -> {model_dir}", run_log)

                # [FIX P1-1] Subsample train refs for deep model
                _deep_max = int(getattr(cfg, "DEEP_MAX_TRAIN", 200_000))
                tr_refs_deep = _subsample_refs_for_deep(tr_refs, _deep_max, seed, log_fp=run_log)

                # Dataset sharing: reuse prebuilt datasets when available
                _prebuilt = None
                if _shared_base_datasets is not None:
                    _base_tr, _base_va, _base_te = _shared_base_datasets
                    if need_logit and logit_map:
                        # Fusion model — wrap with logit injection
                        _prebuilt = (
                            LogitInjectorDataset(_base_tr, logit_map),
                            LogitInjectorDataset(_base_va, logit_map),
                            LogitInjectorDataset(_base_te, logit_map),
                        )
                        write_log(f"[SHARE_DS] {model_name}: reusing shared datasets + logit injection", run_log)
                    else:
                        # Pure model — use base directly
                        _prebuilt = (_base_tr, _base_va, _base_te)
                        write_log(f"[SHARE_DS] {model_name}: reusing shared datasets (no logit)", run_log)

                rep = train_deep_model(
                    model_name=model_name,
                    feature_set=fs,
                    variant_tag=variant_tag,
                    tr_refs=tr_refs_deep,
                    va_refs=va_refs,
                    te_refs=te_refs,
                    seed=seed,
                    device=device,
                    out_dir=model_dir,
                    log_fp=model_log,
                    lgbm_logit_map=logit_map,
                    return_pred_maps=True,
                    prebuilt_datasets=_prebuilt,
                )

                deep_reports[f"{model_name}::{variant_tag}"] = rep
                if rep.get("ok") and rep.get("_pred_maps_in_memory"):
                    deep_pred_maps[(model_name, variant_tag)] = rep["_pred_maps_in_memory"]
                    _emit_split_reports(
                        model_dir=model_dir,
                        model_name=model_name,
                        variant_tag=variant_tag,
                        feature_set=fs,
                        refs_by_split={"train": tr_refs_deep, "val": va_refs, "test": te_refs},
                        rep=rep,
                        run_log=run_log,
                    )

                met = rep.get("metrics", {})
                results_rows.append(
                    {
                        "run_tag": run_tag,
                        "R_core": float(R),
                        "model": model_name,
                        "variant": variant_tag,
                        "feature_set": fs,
                        "seed": seed,
                        "tr_auc": met.get("train", {}).get("auc"),
                        "va_auc": met.get("val", {}).get("auc"),
                        "te_auc": met.get("test", {}).get("auc"),
                        "tr_acc": met.get("train", {}).get("acc"),
                        "va_acc": met.get("val", {}).get("acc"),
                        "te_acc": met.get("test", {}).get("acc"),
                        "artifact_dir": str(model_dir),
                        "checkpoint": rep.get("checkpoint"),
                    }
                )
                # [MEM] θ_{M_i} + B_{M_i} 해제 → 다음 모델 안전 할당
                clear_memory(log_fp=run_log)
                continue

            # ablation grid: deep_only / plus_baseline
            variants_to_run: List[Tuple[str, Optional[Dict[str, float]]]] = []

            if use_logit_inputs:
                variants_to_run = [("plus_baseline", base_logit_map_full)]
            else:
                variants_to_run = [("deep_only", None)]

            for variant_tag, logit_map in variants_to_run:
                model_dir = ensure_dir(models_root / model_name / variant_tag)
                model_log = model_dir / "run.log"
                write_log(f"[MODEL] {model_name}/{variant_tag} -> {model_dir}", run_log)

                # [FIX P1-1] Subsample train refs for deep model
                _deep_max = int(getattr(cfg, "DEEP_MAX_TRAIN", 200_000))
                tr_refs_deep = _subsample_refs_for_deep(tr_refs, _deep_max, seed, log_fp=run_log)

                # Dataset sharing: reuse prebuilt datasets when available
                _prebuilt = None
                if _shared_base_datasets is not None:
                    _base_tr, _base_va, _base_te = _shared_base_datasets
                    if logit_map:
                        _prebuilt = (
                            LogitInjectorDataset(_base_tr, logit_map),
                            LogitInjectorDataset(_base_va, logit_map),
                            LogitInjectorDataset(_base_te, logit_map),
                        )
                        write_log(f"[SHARE_DS] {model_name}/{variant_tag}: shared + logit injection", run_log)
                    else:
                        _prebuilt = (_base_tr, _base_va, _base_te)
                        write_log(f"[SHARE_DS] {model_name}/{variant_tag}: shared (no logit)", run_log)

                rep = train_deep_model(
                    model_name=model_name,
                    feature_set=fs,
                    variant_tag=variant_tag,
                    tr_refs=tr_refs_deep,
                    va_refs=va_refs,
                    te_refs=te_refs,
                    seed=seed,
                    device=device,
                    out_dir=model_dir,
                    log_fp=model_log,
                    lgbm_logit_map=logit_map,
                    return_pred_maps=True,
                    prebuilt_datasets=_prebuilt,
                )
                deep_reports[f"{model_name}::{variant_tag}"] = rep
                if rep.get("ok") and rep.get("_pred_maps_in_memory"):
                    deep_pred_maps[(model_name, variant_tag)] = rep["_pred_maps_in_memory"]
                    _emit_split_reports(
                        model_dir=model_dir,
                        model_name=model_name,
                        variant_tag=variant_tag,
                        feature_set=fs,
                        refs_by_split={"train": tr_refs_deep, "val": va_refs, "test": te_refs},
                        rep=rep,
                        run_log=run_log,
                    )

                met = rep.get("metrics", {})
                results_rows.append(
                    {
                        "run_tag": run_tag,
                        "R_core": float(R),
                        "model": model_name,
                        "variant": variant_tag,
                        "feature_set": fs,
                        "seed": seed,
                        "tr_auc": met.get("train", {}).get("auc"),
                        "va_auc": met.get("val", {}).get("auc"),
                        "te_auc": met.get("test", {}).get("auc"),
                        "tr_acc": met.get("train", {}).get("acc"),
                        "va_acc": met.get("val", {}).get("acc"),
                        "te_acc": met.get("test", {}).get("acc"),
                        "artifact_dir": str(model_dir),
                        "checkpoint": rep.get("checkpoint"),
                    }
                )
                # [MEM] θ_{M_i} + B_{M_i} 해제 → 다음 variant/model 안전 할당
                clear_memory(log_fp=run_log)

        # persist reports
        save_json(run_root / "deep_reports.json", deep_reports)
        if results_rows:
            save_csv_rows(run_root / "ablation_summary.csv", fieldnames=list(results_rows[0].keys()), rows=results_rows)

        # Optional: post-hoc temperature scaling for fusion bases
        # Calibrate T on TRAIN (per patch), apply same T to TRAIN/VAL/TEST logits.
        if bool(getattr(cfg, "TEMP_SCALING_ENABLED", False)) and base_logit_map_full:
            try:
                write_log("[TEMP_SCALE] start (fit on TRAIN, apply to all splits)", run_log)
                y_tr_map = get_label_map(tr_refs, feature_set, log_fp=run_log, log_every=50000)
                patch_by_key: Dict[str, str] = {
                    ref_key(r): str(getattr(r, "patch", "unknown"))
                    for r in (tr_refs + va_refs + te_refs)
                }

                def _apply_temp_by_patch(
                    logit_map: Dict[str, float],
                    t_by_patch: Dict[str, float],
                ) -> Dict[str, float]:
                    out: Dict[str, float] = {}
                    for k, z in (logit_map or {}).items():
                        p = patch_by_key.get(k, None)
                        t = float(t_by_patch.get(p, 1.0)) if p is not None else 1.0
                        if not np.isfinite(t) or t <= 0.0:
                            t = 1.0
                        out[k] = float(z) / float(t)
                    return out

                # baseline map
                _, t_base = calibrate_logits_by_patch(
                    base_logit_map_full,
                    tr_refs,
                    y_tr_map,
                    log_fp=run_log,
                )
                base_logit_map_full = _apply_temp_by_patch(base_logit_map_full, t_base)

                # deep prediction maps used for fusion
                for mk, pm in list(deep_pred_maps.items()):
                    tr_map = dict(pm.get("train", {}))
                    if not tr_map:
                        continue
                    _, t_model = calibrate_logits_by_patch(
                        tr_map,
                        tr_refs,
                        y_tr_map,
                        log_fp=run_log,
                    )
                    for sp in ("train", "val", "test"):
                        pm[sp] = _apply_temp_by_patch(dict(pm.get(sp, {})), t_model)
                    deep_pred_maps[mk] = pm

                write_log("[TEMP_SCALE] done", run_log)
            except Exception as e:
                write_log(f"[TEMP_SCALE] skipped due to error: {e}", run_log)

        # 6) Fusion / stacking
        if enable_factorial or fusion_requests:
            fusion_root = ensure_dir(models_root / "fusion")
            meta_method = str(getattr(args, "oof_meta", "logreg")).strip().lower()
            if meta_method != "logreg":
                write_log(f"[FUSION] unsupported oof_meta={meta_method!r}; fallback to 'logreg'", run_log)
                meta_method = "logreg"

            factorial_adopted = False

            # ------------------------------------------------------------
            # (A) Factorial stacking + automatic best adoption
            # ------------------------------------------------------------
            if enable_factorial:
                def _merge_pred_map(pm: Dict[str, Dict[str, float]]) -> Dict[str, float]:
                    mm: Dict[str, float] = {}
                    for sp in ("train", "val", "test"):
                        mm.update(pm.get(sp, {}))
                    return mm

                def _count_present(refs, mp: Dict[str, float]) -> int:
                    c = 0
                    for r in refs:
                        k = ref_key(r)
                        if k in mp:
                            c += 1
                    return c

                cand_names: List[str] = []
                cand_maps: List[Dict[str, float]] = []
                cand_map_by_name: Dict[str, Dict[str, float]] = {}

                # lgbm is optional candidate (not forced in combos).
                if base_logit_map_full:
                    cand_names.append("lgbm")
                    cand_maps.append(base_logit_map_full)
                    cand_map_by_name["lgbm"] = base_logit_map_full
                else:
                    write_log("[FACTORIAL] lgbm baseline unavailable -> proceed without lgbm candidate", run_log)

                # add ALL available deep variants that have reasonable coverage on tr/va/te
                for (mn, vt), pm in deep_pred_maps.items():
                    name = f"{mn}:{vt}"
                    merged = _merge_pred_map(pm)

                    cov_tr = _count_present(tr_refs, merged)
                    cov_va = _count_present(va_refs, merged)
                    cov_te = _count_present(te_refs, merged)

                    if cov_tr >= 50 and cov_va >= 50 and cov_te >= 50:
                        cand_names.append(name)
                        cand_maps.append(merged)
                        cand_map_by_name[name] = merged
                    else:
                        write_log(
                            f"[FACTORIAL] drop candidate={name} (coverage tr={cov_tr}, val={cov_va}, test={cov_te})",
                            run_log,
                        )

                if len(cand_names) >= 2:
                    factorial_out = ensure_dir(fusion_root / "factorial_search")

                    min_k = int(getattr(args, "factorial_min_k", 2))
                    max_k = int(getattr(args, "factorial_max_k", 3))
                    max_combos = int(getattr(args, "factorial_max_combos", 300))
                    min_k = max(2, min_k)
                    max_k = max(min_k, max_k)

                    write_log(
                        f"[FACTORIAL] start: candidates={len(cand_names)} "
                        f"min_k={min_k} max_k={max_k} max_combos={max_combos}",
                        run_log,
                    )

                    factorial_summary = stack_factorial(
                        tr_refs=tr_refs,
                        va_refs=va_refs,
                        te_refs=te_refs,
                        feature_set=feature_set,
                        cand_names=cand_names,
                        cand_maps=cand_maps,
                        out_dir=factorial_out,
                        log_fp=run_log,
                        seed=seed,
                        meta_method=meta_method,
                        min_k=min_k,
                        max_k=max_k,
                        anchor_name=None,
                        anchor_must_include=False,
                        max_combos=max_combos,
                    )

                    # Auto-adopt best factorial combo:
                    # refit meta on TRAIN+VAL, then evaluate TEST.
                    try:
                        best = factorial_summary.get("best", {}) if isinstance(factorial_summary, dict) else {}
                        best_tag = best.get("tag")
                        best_row = (
                            factorial_summary.get("results", {}).get(best_tag, {})
                            if isinstance(factorial_summary, dict) and best_tag
                            else {}
                        )
                        best_names = list(best_row.get("base_names", [])) if isinstance(best_row, dict) else []

                        if len(best_names) >= 2:
                            best_maps = [cand_map_by_name[nm] for nm in best_names if nm in cand_map_by_name]
                            if len(best_maps) == len(best_names):
                                adopt_dir = ensure_dir(fusion_root / "factorial_best_refit")
                                rep_adopt = refit_meta_trainval_predict_test(
                                    tr_refs=tr_refs,
                                    va_refs=va_refs,
                                    te_refs=te_refs,
                                    feature_set=feature_set,
                                    base_names=best_names,
                                    base_maps=best_maps,
                                    out_dir=adopt_dir,
                                    log_fp=run_log,
                                    seed=seed,
                                    meta_method=meta_method,
                                )
                                adopt_info = {
                                    "source": "factorial_search",
                                    "best_tag": best_tag,
                                    "best_val_auc": best.get("val_auc"),
                                    "best_test_auc_during_search": best.get("test_auc"),
                                    "adopt_base_names": best_names,
                                    "adopt_ok": bool(getattr(rep_adopt, "ok", False)),
                                    "adopt_out_dir": str(adopt_dir),
                                }
                                save_json(adopt_dir / "adopted_from_factorial.json", adopt_info)
                                if rep_adopt.ok:
                                    factorial_adopted = True
                                    write_log(
                                        f"[FACTORIAL] adopted best combo={best_names} with train+val refit",
                                        run_log,
                                    )
                                else:
                                    write_log("[FACTORIAL] best combo refit failed -> fallback to selected fusion", run_log)
                            else:
                                write_log(
                                    "[FACTORIAL] cannot adopt best combo: missing maps for some selected bases",
                                    run_log,
                                )
                        else:
                            write_log("[FACTORIAL] no valid best combo to adopt", run_log)
                    except Exception as e:
                        write_log(f"[FACTORIAL] auto-adopt failed (ignored): {e}", run_log)
                else:
                    write_log("[FACTORIAL] skipped: not enough candidates after filtering", run_log)

            # ------------------------------------------------------------
            # (B) Existing selected stacking (fallback/legacy)
            # ------------------------------------------------------------
            if factorial_adopted:
                write_log("[FUSION] selected-stacking skipped (factorial best already adopted)", run_log)
            else:
                rnn_variant = _best_variant_for(rnn_name, deep_reports) if (fusion_auto_best and rnn_name) else "deep_only"
                gnn_variant = _best_variant_for(gnn_name, deep_reports) if (fusion_auto_best and gnn_name) else "deep_only"

                if rnn_name and (rnn_name, rnn_variant) not in deep_pred_maps:
                    rnn_variant = "plus_baseline" if (rnn_name, "plus_baseline") in deep_pred_maps else rnn_variant
                if gnn_name and (gnn_name, gnn_variant) not in deep_pred_maps:
                    gnn_variant = "plus_baseline" if (gnn_name, "plus_baseline") in deep_pred_maps else gnn_variant

                base_names: List[str] = []
                maps_train: List[Dict[str, float]] = []
                maps_val: List[Dict[str, float]] = []
                maps_test: List[Dict[str, float]] = []

                if base_logit_map_full:
                    base_names.append("lgbm")
                    maps_train.append(split_logit_map_by_refs(tr_refs, base_logit_map_full))
                    maps_val.append(split_logit_map_by_refs(va_refs, base_logit_map_full))
                    maps_test.append(split_logit_map_by_refs(te_refs, base_logit_map_full))

                if rnn_name and (rnn_name, rnn_variant) in deep_pred_maps:
                    pm = deep_pred_maps[(rnn_name, rnn_variant)]
                    base_names.append(f"{rnn_name}:{rnn_variant}")
                    maps_train.append(pm.get("train", {}))
                    maps_val.append(pm.get("val", {}))
                    maps_test.append(pm.get("test", {}))
                if gnn_name and (gnn_name, gnn_variant) in deep_pred_maps:
                    pm = deep_pred_maps[(gnn_name, gnn_variant)]
                    base_names.append(f"{gnn_name}:{gnn_variant}")
                    maps_train.append(pm.get("train", {}))
                    maps_val.append(pm.get("val", {}))
                    maps_test.append(pm.get("test", {}))

                if len(base_names) < 2:
                    write_log("[FUSION] skipped: not enough base models (need >=2)", run_log)
                else:
                    if stacking_mode == "oof":
                        n_splits = int(getattr(args, "oof_folds", getattr(cfg, "OOF_FOLDS", 5)))
                        folds = split_by_match_id_kfold(tr_refs, n_splits=n_splits, seed=seed)
                        out_dir = ensure_dir(fusion_root / f"oof_meta_{n_splits}folds")
                        stack_oof_meta(
                            tr_refs=tr_refs,
                            va_refs=va_refs,
                            te_refs=te_refs,
                            feature_set=feature_set,
                            base_names=base_names,
                            base_maps_tr=maps_train,
                            base_maps_va=maps_val,
                            base_maps_te=maps_test,
                            folds=folds,
                            out_dir=out_dir,
                            log_fp=run_log,
                            seed=seed,
                            meta_method=meta_method,
                        )
                    else:
                        combined_maps = []
                        for m_tr, m_va, m_te in zip(maps_train, maps_val, maps_test):
                            mm = {}
                            mm.update(m_tr)
                            mm.update(m_va)
                            mm.update(m_te)
                            combined_maps.append(mm)

                        out_dir = ensure_dir(fusion_root / "simple_train_fit")
                        stack_simple(
                            tr_refs=tr_refs,
                            va_refs=va_refs,
                            te_refs=te_refs,
                            feature_set=feature_set,
                            base_names=base_names,
                            base_maps=combined_maps,
                            out_dir=out_dir,
                            log_fp=run_log,
                            seed=seed,
                            meta_method=meta_method,
                            fit_on="train",
                        )

        write_log("[DONE] finished this R_core run.", run_log)

        sweep_rows.append(
            {
                "run_tag": run_tag,
                "R_core": float(R),
                "n_fights": len(refs),
                "n_train": len(tr_refs),
                "n_val": len(va_refs),
                "n_test": len(te_refs),
            }
        )

    # sweep summary
    if sweep_rows:
        try:
            out_fp = sweep_root / "sweep_summary.csv"
            save_csv_rows(out_fp, fieldnames=list(sweep_rows[0].keys()), rows=sweep_rows)
            save_json(sweep_root / "sweep_summary.json", {"rows": sweep_rows})
        except Exception:
            pass
