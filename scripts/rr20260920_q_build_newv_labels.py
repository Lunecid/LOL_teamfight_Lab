#!/usr/bin/env python3
"""Build q-lineage SVI labels under fit85 MLP V (design contract).

Roles:
  - TEST / Q_CAL / Q_SELECT: frozen evaluator bundle (safe — not in V TRAIN)
  - TRAIN: OOF leave-fold only (--train-oof); never frozen bundle as train labels

Writes: outputs/q_newv_fit85_20260920/labels/
Contract: docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
OUT = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"

from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated  # noqa: E402


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def cohort_keys(data_root: Path, set_name: str) -> set:
    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / f"{set_name}_labels.npz",
        allow_pickle=False,
    )
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / f"{set_name}_cohort.npz",
        allow_pickle=False,
    )
    m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    return set(
        zip(lab["match"][m].astype(str).tolist(), lab["s"][m].astype(np.int64).tolist())
    )


def score_role(
    D,
    L,
    roles: List[str],
    lab_keys: set,
    predict_fn,
    label_kind: str,
    v_sha: str,
    v_lineage: str,
) -> Dict[str, np.ndarray]:
    E = D.load_engagements(L, "MAIN", roles, states=True, counts=False)
    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    keep = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    keep &= (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    Xpre = E["X_pre"][keep]
    Xpost = E["X_post_h90"][keep]
    p_pre = predict_fn(Xpre)
    p_post = predict_fn(Xpost)
    dV = p_post - p_pre
    miss = ~(np.isfinite(p_pre) & np.isfinite(p_post))
    y = np.full(len(dV), -1, dtype=np.int8)
    y[~miss] = (dV[~miss] > 0).astype(np.int8)
    return dict(
        match=em[keep],
        s=es[keep],
        sub_role=E["sub_role"][keep].astype(str),
        p_pre=p_pre.astype(np.float64),
        p_post=p_post.astype(np.float64),
        delta_V=dV.astype(np.float64),
        Y_SVI=y,
        exact_zero=(dV == 0).astype(np.int8),
        missing_score=miss.astype(np.int8),
        B40=((p_pre >= 0.40) & (p_pre <= 0.60) & (~miss)).astype(np.int8),
        valid_h90=np.ones(int(keep.sum()), dtype=np.int8),
        label_kind=np.array([label_kind] * int(keep.sum())),
        v_lineage=np.array([v_lineage] * int(keep.sum())),
        v_sha16=np.array([v_sha[:16]] * int(keep.sum())),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=["TEST", "Q_CAL", "Q_SELECT"],
                    help="frozen-eval label roles (default: TEST Q_CAL Q_SELECT)")
    ap.add_argument("--train-oof", action="store_true",
                    help="build TRAIN OOF labels (5-fold MLP; heavy — separate step)")
    args = ap.parse_args(argv)

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D
    import fc20260915_common as C

    OUT.mkdir(parents=True, exist_ok=True)
    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")
    v_sha = sha256_file(BUNDLE)
    ev = load_evaluator(BUNDLE)
    L = D.Layout(False)

    def predict_frozen(X):
        return predict_calibrated(ev, X)

    role_to_set = {
        "TEST": "MAIN_TEST",
        "Q_CAL": "MAIN_VALIDATION",
        "Q_SELECT": "MAIN_VALIDATION",
    }
    # Q_CAL / Q_SELECT share MAIN_VALIDATION cohort file; filter by sub_role after load
    for role in args.roles:
        print(f"label {role} (frozen eval)…", flush=True)
        set_name = role_to_set.get(role)
        if set_name is None:
            raise SystemExit(f"unsupported frozen role {role}")
        keys = cohort_keys(data_root, set_name)
        pack = score_role(D, L, [role], keys, predict_frozen, "frozen_eval", v_sha, "A_MLP_expanded_fit85")
        # For VAL roles, cohort file is pooled — already filtered by load_engagements role
        path = OUT / f"{role}_h90.npz"
        np.savez_compressed(path, **pack)
        meta = dict(
            role=role,
            n=int(len(pack["Y_SVI"])),
            P_SVI=float(np.mean(pack["Y_SVI"][pack["Y_SVI"] >= 0] == 1)),
            B40_n=int(pack["B40"].sum()),
            label_kind="frozen_eval",
            v_sha256=v_sha,
            generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        )
        (OUT / f"{role}_h90_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {path} n={meta['n']} P(SVI=1)={meta['P_SVI']:.3f} B40={meta['B40_n']}", flush=True)

    if args.train_oof:
        print("TRAIN OOF: launching fold loop (heavy)…", flush=True)
        return _train_oof(data_root, D, L, C, v_sha)
    else:
        note = {
            "train_oof": "NOT BUILT — run with --train-oof after confirming GPU/time budget",
            "contract": "docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md",
            "forbidden": "Do not fit q on TRAIN using frozen_eval labels",
        }
        (OUT / "TRAIN_oof_STATUS.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
        print("TRAIN OOF deferred (see TRAIN_oof_STATUS.json)", flush=True)
    return 0


def _train_oof(data_root, D, L, C, frozen_sha_note: str) -> int:
    """Leave-fold MLP Expanded labels for TRAIN engagements.

    Reuses wave-4 architecture via v_redesign_feature_adapters + same training recipe
    as rr20260919_v_redesign_fit_wave4_corrected (fit85 holdout inside each fold-train).
    """
    # Heavy path: import training helpers from wave4 module when available.
    # For safety, this first cut writes a runnable checklist if import/train is too coupled.
    try:
        from v_redesign_feature_adapters import FeatureSchema, ProfileBundle, match_holdout_mask
    except Exception as e:
        (OUT / "TRAIN_oof_STATUS.json").write_text(
            json.dumps({"error": str(e), "status": "adapters_missing"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1

    # Placeholder implementation note — full fold training is a dedicated job.
    # Emit fold→match maps and engagement indices so a worker can fill scores.
    folds = [f"fold{k}" for k in range(C.N_FOLDS)]
    index = {}
    for k, role in enumerate(folds):
        E = D.load_engagements(L, "MAIN", [role], states=False, counts=False)
        keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
        # Restrict to T cohort via MAIN_TRAIN labels
        keys = cohort_keys(data_root, "MAIN_TRAIN")
        em, es = E["match"].astype(str), E["s"].astype(np.int64)
        in_t = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
        keep &= in_t
        index[role] = dict(
            n=int(keep.sum()),
            n_matches=int(len(np.unique(em[keep]))),
            train_folds=[f for f in folds if f != role],
        )
        print(f"  {role}: T∩valid engagements={index[role]['n']}", flush=True)

    manifest = dict(
        status="INDEX_READY_TRAINING_PENDING",
        design="leave_fold_MLP_expanded_fit85_recipe",
        folds=index,
        note=(
            "Worker must fit MLP on train_folds (bucket V rows), calibrate on V_CAL, "
            "score this fold's X_pre/X_post_h90 with the same g∘f∘T, write TRAIN_oof_h90.npz."
        ),
        frozen_eval_sha_not_for_train=frozen_sha_note[:16],
        contract="docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md",
    )
    (OUT / "TRAIN_oof_STATUS.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote TRAIN_oof_STATUS.json (index ready; fold MLP training is next worker)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
