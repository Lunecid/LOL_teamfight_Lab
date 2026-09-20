#!/usr/bin/env python3
"""Match-fold OOF MLP Expanded evaluators → TRAIN SVI labels for q.

Contract: docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md

For each fold k:
  - Fit ProfileBundle + embedding MLP on bucket V rows of matches NOT in fold k
    (early-stop holdout carved inside remaining matches only; seed=7, 15%)
  - Fit PosSlopeSigmoid g^(-k) on V_CAL
  - Score fold-k T∩valid_h90 engagements with the SAME V^(-k) at pre and post
  - Emit p_pre, p_post, delta_V, Y_SVI, B40 from that V only

Does NOT copy fit85 weights. Does NOT use frozen bundle for TRAIN labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT_T = REPO / "outputs" / "q_newv_fit85_20260920"
OUT_S_ROOT = REPO / "outputs" / "q_newv_fit85_20260920_S"
OOF_DIR_T = OUT_T / "oof_evaluators"  # always reuse T fold evaluators for S
ROLE_T = "EXPLORATORY_Q_NEWV_OOF_PRIOR_TEST_EXPOSURE"
ROLE_S = "EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE"

# Contract §2 expected S TRAIN h90 row counts per fold (warn-only on mismatch)
EXPECTED_ENG_S = {0: 24030, 1: 25005, 2: 24520, 3: 24690, 4: 24804}

from v_redesign_feature_adapters import (  # noqa: E402
    FeatureSchema,
    ProfileBundle,
    mean_one_match_weights,
    match_holdout_mask,
)
from rr20260920_q_build_newv_labels import cohort_keys  # noqa: E402


def _load_wave4():
    path = REPO / "scripts" / "rr20260919_v_redesign_fit_wave4_corrected.py"
    spec = importlib.util.spec_from_file_location("wave4_fit", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def labels_dir(cohort: str) -> Path:
    if cohort == "T":
        return OUT_T / "labels"
    if cohort == "S":
        return OUT_S_ROOT / "labels"
    raise SystemExit(f"unsupported cohort {cohort!r}")


def integrity_report(pack: Dict[str, np.ndarray], fold_meta: List[Dict], expected_n: int) -> Dict[str, Any]:
    y = pack["Y_SVI"]
    dV = pack["delta_V"]
    p0, p1 = pack["p_pre"], pack["p_post"]
    miss = pack["missing_score"].astype(bool)
    ok = ~miss
    keys = list(zip(pack["match"].astype(str).tolist(), pack["s"].astype(np.int64).tolist()))
    n_unique = len(set(keys))
    cons_y = bool(np.all(y[ok] == (dV[ok] > 0).astype(np.int8)))
    cons_d = bool(np.allclose(dV[ok], p1[ok] - p0[ok], atol=1e-7, equal_nan=False))
    b40 = pack["B40"].astype(bool)
    cons_b40 = bool(np.all(b40[ok] == ((p0[ok] >= 0.40) & (p0[ok] <= 0.60))))
    checks = dict(
        n_rows=int(len(y)),
        expected_n=int(expected_n),
        n_unique_keys=int(n_unique),
        no_duplicate_keys=n_unique == len(keys),
        row_count_matches_expected=(len(y) == expected_n),
        y_matches_delta_sign=cons_y,
        delta_matches_p_post_minus_p_pre=cons_d,
        B40_from_same_p_pre=cons_b40,
        n_missing=int(miss.sum()),
        n_exact_zero=int((pack["exact_zero"] == 1).sum()),
        folds_recorded=len(fold_meta),
    )
    checks["PASS"] = all(
        [
            checks["no_duplicate_keys"],
            checks["row_count_matches_expected"],
            checks["y_matches_delta_sign"],
            checks["delta_matches_p_post_minus_p_pre"],
            checks["B40_from_same_p_pre"],
            checks["n_missing"] == 0,
            checks["folds_recorded"] == 5,
        ]
    )
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4", help="comma folds to run")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--cohort", choices=["T", "S"], default="T")
    ap.add_argument(
        "--reuse-evaluators",
        action="store_true",
        default=False,
        help="score cohort engagements with saved T fold V^(-k); skip fit (required for S)",
    )
    args = ap.parse_args(argv)

    data_root = _data_root()
    _setup(data_root)
    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    W4 = _load_wave4()
    fit_mlp_emb = W4.fit_mlp_emb
    predict_mlp_emb = W4.predict_mlp_emb
    PosSlopeSigmoid = W4.PosSlopeSigmoid

    LAB = labels_dir(args.cohort)
    OOF_DIR = OOF_DIR_T  # evaluators always from T lineage
    ROLE = ROLE_S if args.cohort == "S" else ROLE_T
    LAB.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    L = D.Layout(False)
    train_keys = cohort_keys(data_root, "MAIN_TRAIN", cohort=args.cohort)

    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    all_roles = [f"fold{k}" for k in range(C.N_FOLDS)]

    fold_matches: Dict[int, set] = {}
    expected_by_fold: Dict[int, int] = {}
    census_warnings: List[str] = []

    if not args.reuse_evaluators:
        print("load all TRAIN fold bucket V rows…", flush=True)
        TR = D.load_v_rows(L, "MAIN", all_roles, bucket_only=True)
        names = list(TR["names"])
        schema = FeatureSchema.from_names(names)
        W_tr = D.load_outcomes(L, "MAIN", all_roles, purpose="q OOF V")
        yTR = np.asarray([W_tr[m][0] for m in TR["match"].tolist()], dtype=np.float64)
        gTR = TR["match"].astype(str)
        print(f"  TR n={len(yTR)} matches={len(np.unique(gTR))}", flush=True)

        print("load V_CAL for fold-wise g…", flush=True)
        CA = D.load_v_rows(L, "MAIN", ["V_CAL"], bucket_only=True)
        W_cal = D.load_outcomes(L, "MAIN", ["V_CAL"], purpose="q OOF calib")
        yCA = np.asarray([W_cal[m][0] for m in CA["match"].astype(str).tolist()], dtype=np.float64)
        gCA = CA["match"].astype(str)

        for k in range(C.N_FOLDS):
            fold_matches[k] = set(gTR[TR["sub_role"].astype(str) == f"fold{k}"].tolist())
            print(f"  fold{k}: V-bucket matches={len(fold_matches[k])}", flush=True)
    else:
        TR = yTR = gTR = CA = yCA = gCA = schema = None  # unused when reusing
        # Match sets from engagement fold roles (same leave-match guarantee)
        for k in range(C.N_FOLDS):
            E0 = D.load_engagements(L, "MAIN", [f"fold{k}"], states=False, counts=False)
            fold_matches[k] = set(E0["match"].astype(str).tolist())
            print(f"  fold{k}: eng-role matches={len(fold_matches[k])} (reuse mode)", flush=True)

    expected_eng = 0
    for k in range(C.N_FOLDS):
        E = D.load_engagements(L, "MAIN", [f"fold{k}"], states=False, counts=False)
        em, es = E["match"].astype(str), E["s"].astype(np.int64)
        keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
        in_coh = np.array([(a, int(b)) in train_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
        keep &= in_coh
        n_k = int(keep.sum())
        expected_by_fold[k] = n_k
        expected_eng += n_k
        if args.cohort == "S":
            exp = EXPECTED_ENG_S[k]
            if n_k != exp:
                msg = f"fold{k}: eng_labels={n_k} expected_census={exp} (warn only)"
                census_warnings.append(msg)
                print(f"  WARN {msg}", flush=True)
        stray = [m for m in set(em[keep].tolist()) if m not in fold_matches[k]]
        if stray:
            raise SystemExit(f"fold{k}: engagement match not in V fold match set e.g. {stray[0]}")
        print(f"  fold{k}: eng_labels_expected={n_k}", flush=True)

    # For S integrity: use actual totals so finite-score drift does not FAIL the pack
    integrity_expected = expected_eng

    parts: List[Dict[str, np.ndarray]] = []
    fold_meta: List[Dict[str, Any]] = []
    evaluator_sha: Dict[str, str] = {}

    for k in fold_ids:
        t_fold0 = time.time()
        held = fold_matches[k]
        print(f"\n=== OOF fold{k} cohort={args.cohort}: hold {len(held)} matches ===", flush=True)

        if args.reuse_evaluators:
            fold_path = OOF_DIR / f"V_oof_fold{k}_mlp_expanded.joblib"
            bun_path = OOF_DIR / f"bundle_oof_fold{k}.joblib"
            missing = [str(p) for p in (fold_path, bun_path) if not p.is_file()]
            if missing:
                raise SystemExit(
                    "reuse-evaluators missing files:\n  " + "\n  ".join(missing)
                )
            fold_obj = joblib.load(fold_path)
            bun_obj = joblib.load(bun_path)
            bun = bun_obj["bundle"]
            pack = fold_obj["mlp"]
            calib = fold_obj["calib"]
            gcal = PosSlopeSigmoid()
            gcal.coef_ = float(calib["coef"])
            gcal.intercept_ = float(calib["intercept"])
            gcal.ok = bool(calib["ok"])
            evaluator_sha[f"V_oof_fold{k}"] = sha256_file(fold_path)[:16]
            evaluator_sha[f"bundle_oof_fold{k}"] = sha256_file(bun_path)[:16]

            def predict_cal(X, _bun=bun, _pack=pack, _gcal=gcal):
                num = _bun.standardize_numeric(_bun.numeric_raw(X))
                ids = _bun.embedding_ids(X)
                return _gcal.transform(predict_mlp_emb(_pack, num, ids))

            if isinstance(pack, dict) and "best_epoch" in pack:
                best_epoch = int(pack["best_epoch"])
                best_val_brier = float(pack.get("best_val_brier", float("nan")))
            else:
                best_epoch = -1
                best_val_brier = float("nan")
            fit_m_sum = int(fold_obj.get("n_fit_rows", -1))
            stop_sum = int(fold_obj.get("n_stop_rows", -1))
        else:
            assert TR is not None and gTR is not None
            in_held = np.array([m in held for m in gTR.tolist()], dtype=bool)
            train_rows = ~in_held
            assert not np.any(in_held & train_rows)
            stop = match_holdout_mask(gTR, args.holdout_frac, seed=args.seed) & train_rows
            fit_m = train_rows & ~stop
            print(
                f"  fit_rows={int(fit_m.sum())} stop_rows={int(stop.sum())} "
                f"held_query_rows={int(in_held.sum())}",
                flush=True,
            )
            if int(np.any(np.array([m in held for m in gTR[fit_m].tolist()], dtype=bool))):
                raise SystemExit("LEAK: held match in fit set")
            if int(np.any(np.array([m in held for m in gTR[stop].tolist()], dtype=bool))):
                raise SystemExit("LEAK: held match in stop set")

            bun = ProfileBundle(schema, "expanded").fit(TR["X"][fit_m])
            num_all = bun.standardize_numeric(bun.numeric_raw(TR["X"]))
            id_all = bun.embedding_ids(TR["X"])
            print("  fit MLP…", flush=True)
            pack = fit_mlp_emb(
                num_all[fit_m],
                id_all[fit_m],
                yTR[fit_m],
                mean_one_match_weights(gTR[fit_m]),
                num_all[stop],
                id_all[stop],
                yTR[stop],
                mean_one_match_weights(gTR[stop]),
                n_vocab=bun.vocab.n_vocab,
                max_epochs=args.max_epochs,
                seed=args.seed,
            )
            print(
                f"  MLP best_epoch={pack['best_epoch']} best_val_brier={pack['best_val_brier']:.5f} "
                f"({time.time()-t_fold0:.0f}s so far)",
                flush=True,
            )

            num_ca = bun.standardize_numeric(bun.numeric_raw(CA["X"]))
            id_ca = bun.embedding_ids(CA["X"])
            raw_ca = predict_mlp_emb(pack, num_ca, id_ca)
            gcal = PosSlopeSigmoid().fit(raw_ca, yCA, mean_one_match_weights(gCA))
            print(f"  g^(-{k}) ok={gcal.ok} coef={gcal.coef_:.4f} intercept={gcal.intercept_:.4f}", flush=True)

            def predict_cal(X, _bun=bun, _pack=pack, _gcal=gcal):
                num = _bun.standardize_numeric(_bun.numeric_raw(X))
                ids = _bun.embedding_ids(X)
                return _gcal.transform(predict_mlp_emb(_pack, num, ids))

            fold_path = OOF_DIR / f"V_oof_fold{k}_mlp_expanded.joblib"
            joblib.dump(
                dict(
                    kind="mlp_embedding_oof",
                    fold=k,
                    profile="expanded",
                    mlp=pack,
                    calib=dict(coef=gcal.coef_, intercept=gcal.intercept_, ok=gcal.ok),
                    fit_scope=f"leave_fold{k}_match_holdout_frac{args.holdout_frac}_seed{args.seed}",
                    n_fit_rows=int(fit_m.sum()),
                    n_stop_rows=int(stop.sum()),
                    n_held_matches=len(held),
                    n_eng_labeled=-1,
                ),
                fold_path,
            )
            try:
                joblib.dump({"profile": "expanded", "bundle": bun}, OOF_DIR / f"bundle_oof_fold{k}.joblib")
            except Exception as e:
                print(f"  warn: could not dump ProfileBundle ({e})", flush=True)
            fit_m_sum = int(fit_m.sum())
            stop_sum = int(stop.sum())
            best_epoch = int(pack["best_epoch"])
            best_val_brier = float(pack["best_val_brier"])
            evaluator_sha[f"V_oof_fold{k}"] = sha256_file(fold_path)[:16]

        # score engagements in fold k
        E = D.load_engagements(L, "MAIN", [f"fold{k}"], states=True, counts=False)
        em, es = E["match"].astype(str), E["s"].astype(np.int64)
        keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
        in_coh = np.array([(a, int(b)) in train_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
        keep &= in_coh
        bad = [m for m in em[keep].tolist() if m not in held]
        if bad:
            raise SystemExit(f"fold{k}: engagement match not in fold match set example={bad[0]}")

        p_pre = predict_cal(E["X_pre"][keep])
        p_post = predict_cal(E["X_post_h90"][keep])
        dV = p_post - p_pre
        miss = ~(np.isfinite(p_pre) & np.isfinite(p_post))
        y = np.full(len(dV), -1, dtype=np.int8)
        y[~miss] = (dV[~miss] > 0).astype(np.int8)
        n_k = int(keep.sum())
        print(f"  labeled eng={n_k} P(SVI=1)={float(np.mean(y[~miss]==1)):.3f}", flush=True)

        if not args.reuse_evaluators:
            # update n_eng_labeled on dump already written — rewrite lightly
            fold_path = OOF_DIR / f"V_oof_fold{k}_mlp_expanded.joblib"
            obj = joblib.load(fold_path)
            obj["n_eng_labeled"] = n_k
            joblib.dump(obj, fold_path)

        parts.append(
            dict(
                match=em[keep],
                s=es[keep],
                sub_role=E["sub_role"][keep].astype(str),
                fold_id=np.full(n_k, k, dtype=np.int8),
                p_pre=p_pre.astype(np.float64),
                p_post=p_post.astype(np.float64),
                delta_V=dV.astype(np.float64),
                Y_SVI=y,
                exact_zero=(dV == 0).astype(np.int8),
                missing_score=miss.astype(np.int8),
                B40=((p_pre >= 0.40) & (p_pre <= 0.60) & (~miss)).astype(np.int8),
                valid_h90=np.ones(n_k, dtype=np.int8),
                label_kind=np.array(["oof"] * n_k),
                v_lineage=np.array([f"A_MLP_expanded_oof_fold{k}"] * n_k),
            )
        )
        fold_meta.append(
            dict(
                fold=k,
                n_held_matches=len(held),
                n_eng=n_k,
                n_fit_rows=fit_m_sum,
                n_stop_rows=stop_sum,
                best_epoch=best_epoch,
                best_val_brier=best_val_brier,
                calib_ok=bool(gcal.ok),
                calib_coef=float(gcal.coef_),
                calib_intercept=float(gcal.intercept_),
                wall_s=float(time.time() - t_fold0),
                evaluator=str((OOF_DIR / f"V_oof_fold{k}_mlp_expanded.joblib").relative_to(REPO)).replace("\\", "/"),
                evaluator_reused=bool(args.reuse_evaluators),
            )
        )

    if set(fold_ids) != set(range(5)):
        status = dict(status="PARTIAL", folds=fold_meta, note="Re-run with all folds 0-4 to finalize TRAIN_oof_h90.npz")
        (LAB / "TRAIN_oof_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print("partial fold run — not writing final TRAIN_oof_h90.npz", flush=True)
        return 0

    keys = parts[0].keys()
    pack = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    checks = integrity_report(pack, fold_meta, integrity_expected)
    out_npz = LAB / "TRAIN_oof_h90.npz"
    np.savez_compressed(out_npz, **pack)
    meta = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        role_tag=ROLE,
        cohort=args.cohort,
        evaluator_reused=bool(args.reuse_evaluators),
        evaluator_sha16=evaluator_sha,
        census_warnings=census_warnings,
        expected_eng_by_fold=expected_by_fold,
        contract="docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md",
        scale_split_contract="docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md",
        architecture="Expanded embedding MLP (wave-4 recipe); NOT fit85 weight clone",
        folds=fold_meta,
        integrity=checks,
        path=str(out_npz.relative_to(REPO)).replace("\\", "/"),
        P_SVI=float(np.mean(pack["Y_SVI"] == 1)),
        B40_n=int(pack["B40"].sum()),
    )
    (LAB / "TRAIN_oof_h90_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (LAB / "TRAIN_oof_STATUS.json").write_text(
        json.dumps(dict(status="COMPLETE" if checks["PASS"] else "FAIL_INTEGRITY", integrity=checks), indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("wrote", out_npz, "PASS" if checks["PASS"] else "FAIL", checks, flush=True)
    return 0 if checks["PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
