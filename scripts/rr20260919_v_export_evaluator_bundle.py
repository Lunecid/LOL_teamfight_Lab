#!/usr/bin/env python3
"""Build frozen evaluator_bundle for wave-4 MLP (+ peers) and verify reload.

Uses the same TRAIN match-holdout seed as wave-4 to rebuild ProfileBundle, then
attaches saved model weights + V_CAL calibration from results.json.

Writes:
  outputs/v_redesign_wave4_corrected_20260919/evaluators/*.joblib
  docs/V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
OUT_EVAL = WAVE4 / "evaluators"

from v_redesign_feature_adapters import (  # noqa: E402
    FeatureSchema,
    ProfileBundle,
    match_holdout_mask,
)
from v_redesign_evaluator_bundle import (  # noqa: E402
    PosSlopeSigmoid,
    build_mlp_evaluator_payload,
    load_evaluator,
    predict_calibrated,
    predict_raw_mlp,
    save_evaluator,
)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-check", type=int, default=2048)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_common as C
    import fc20260915_data as D

    results = json.loads((WAVE4 / "results.json").read_text(encoding="utf-8"))
    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load TRAIN / V_SELECT / TEST…", flush=True)
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    SE = D.load_v_rows(L, "MAIN", ["V_SELECT"], bucket_only=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    schema = FeatureSchema.from_names(TR["names"])
    stop = match_holdout_mask(TR["match"], 0.15, seed=7)
    fit_m = ~stop
    bun = ProfileBundle(schema, "expanded").fit(TR["X"][fit_m])

    mlp_path = WAVE4 / "models" / "A_MLP_expanded.joblib"
    if not mlp_path.is_file():
        raise SystemExit(f"missing {mlp_path}")
    model_pack = joblib.load(mlp_path)
    cal = results["selection"]["A_MLP_expanded"]["calib"]
    fit_scope = "train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit"

    sv2 = data_root / "worktrees" / "engagement-state-value" / "gameplay" / "state_value_v2.py"
    src_hash = hashlib.sha256(sv2.read_bytes()).hexdigest()[:16] if sv2.is_file() else None

    payload = build_mlp_evaluator_payload(
        name="A_MLP_expanded",
        bun=bun,
        model_pack=model_pack,
        calibration=cal,
        fit_scope=fit_scope,
        meta=dict(
            wave4_generated=results.get("generated"),
            L_time=results["selection"]["A_MLP_expanded"]["L_time"],
            state_value_v2_sha256_16=src_hash,
            census=results.get("census"),
            note=(
                "Provisional freeze-prep evaluator. Fit on TRAIN 85% match-holdout; "
                "not full-TRAIN refit. Continuity and labels must use THIS bundle."
            ),
        ),
    )
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    ev_path = OUT_EVAL / "A_MLP_expanded_evaluator.joblib"
    save_evaluator(ev_path, payload)
    print("wrote", ev_path, flush=True)

    # also dump JSON sidecar (no tensors) for audit
    side = dict(
        evaluator_id=payload["evaluator_id"],
        kind=payload["kind"],
        fit_scope=payload["fit_scope"],
        calibration=payload["calibration"],
        preproc_dims=payload["preproc"]["dims"],
        vocab_n=payload["preproc"]["vocab_n"],
        meta=payload["meta"],
        path=str(ev_path.relative_to(REPO)).replace("\\", "/"),
    )
    (OUT_EVAL / "A_MLP_expanded_evaluator_meta.json").write_text(
        json.dumps(side, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Reload reproducibility: score fixed SE/TEST heads twice
    ev1 = load_evaluator(ev_path)
    rng = np.random.default_rng(0)
    # single + batch consistency on SELECT
    n = min(args.n_check, len(SE["match"]))
    idx = rng.choice(len(SE["match"]), size=n, replace=False)
    X = SE["X"][idx]
    p_batch = predict_calibrated(ev1, X)
    p_single = np.array([predict_calibrated(ev1, X[i : i + 1])[0] for i in range(min(64, n))])
    single_ok = bool(np.allclose(p_batch[: len(p_single)], p_single, atol=args.atol, rtol=0))

    # second load
    ev2 = load_evaluator(ev_path)
    p2 = predict_calibrated(ev2, X)
    reload_ok = bool(np.allclose(p_batch, p2, atol=args.atol, rtol=0))
    max_abs = float(np.max(np.abs(p_batch - p2)))

    # raw vs calib finite
    raw = predict_raw_mlp(ev1, X[:128])
    finite_ok = bool(np.isfinite(raw).all() and np.isfinite(p_batch).all())

    check = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        evaluator=str(ev_path),
        n_check=int(n),
        atol=args.atol,
        reload_ok=reload_ok,
        single_vs_batch_ok=single_ok,
        finite_ok=finite_ok,
        max_abs_reload_diff=max_abs,
        fit_scope=fit_scope,
        pass_all=reload_ok and single_ok and finite_ok,
    )
    (OUT_EVAL / "reload_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    md = REPO / "docs" / "V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md"
    lines = [
        "# Evaluator bundle reload check — A_MLP_expanded",
        "",
        f"Generated: {check['generated']}",
        "",
        "**Fit scope (LOCKED for continuity):** "
        "`train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`",
        "",
        "Collaborator freeze-prep: preproc + model + calibration must travel together. "
        "This check loads the joblib in-process twice and compares calibrated probs.",
        "",
        "## Bundle path",
        "",
        f"- `{ev_path.relative_to(REPO).as_posix()}`",
        f"- Meta: `{ (OUT_EVAL / 'A_MLP_expanded_evaluator_meta.json').relative_to(REPO).as_posix() }`",
        "",
        "## Checks",
        "",
        f"| Check | Result |",
        f"|---|---|",
        f"| Reload bit-match (atol={args.atol}) | **{reload_ok}** (max abs diff={max_abs:.3e}) |",
        f"| Single vs batch (n≤64) | **{single_ok}** |",
        f"| Finite raw/calib | **{finite_ok}** |",
        f"| **PASS** | **{check['pass_all']}** |",
        "",
        "## Contract notes",
        "",
        "- Continuity / SVI labels for this lineage must use **this** evaluator, not a later full-TRAIN refit, unless a new bundle version is cut.",
        "- Full-TRAIN refit remains optional `corrected_v2` and requires recorded `best_epoch` before re-calib + continuity.",
        "- History density caveat for GRU/H5 unchanged — see wave-4 ledger.",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(check, indent=2))
    print("wrote", md)
    return 0 if check["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
