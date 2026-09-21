#!/usr/bin/env python3
"""Finish freeze-prep reproducibility: subprocess reload + wave-4 path parity.

1) Same-process reload (already done) — restated
2) Fresh OS process: only bundle + fixed X (no TRAIN / no ProfileBundle.fit)
3) Parity vs wave-4 live path: rebuild bun on TRAIN fit85 + A_MLP_expanded.joblib + calib

Writes docs + outputs under wave4/evaluators/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
WORKER = REPO / "scripts" / "rr20260919_v_score_bundle_worker.py"

from v_redesign_feature_adapters import (  # noqa: E402
    FeatureSchema,
    ProfileBundle,
    match_holdout_mask,
)
from v_redesign_evaluator_bundle import (  # noqa: E402
    PosSlopeSigmoid,
    load_evaluator,
    predict_calibrated,
    predict_raw_mlp,
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


def wave4_live_predict(X_raw, bun, model_pack, calib) -> np.ndarray:
    """Original wave-4 scoring path: bun transforms + model pack + PosSlopeSigmoid."""
    import torch
    import torch.nn as nn

    num = bun.standardize_numeric(bun.numeric_raw(X_raw))
    ids = bun.embedding_ids(X_raw)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(model_pack["n_vocab"], model_pack["emb_dim"], padding_idx=0)
            d = model_pack["d_num"] + model_pack["n_slots"] * model_pack["emb_dim"]
            layers = []
            for h in model_pack["hidden"]:
                layers += [
                    nn.Linear(d, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(model_pack["dropout"]),
                ]
                d = h
            layers += [nn.Linear(d, 1)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, num_t, ids_t):
            e = self.emb(ids_t).reshape(ids_t.size(0), -1)
            return self.mlp(torch.cat([num_t, e], dim=-1)).squeeze(-1)

    net = Net()
    net.load_state_dict(model_pack["state_dict"])
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(num), 65536):
            xb = torch.from_numpy(num[i : i + 65536].astype(np.float32))
            ib = torch.from_numpy(ids[i : i + 65536].astype(np.int64))
            outs.append(torch.sigmoid(net(xb, ib)).numpy())
    raw = np.concatenate(outs)
    g = PosSlopeSigmoid.from_dict(calib)
    return g.transform(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-check", type=int, default=4096)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    if not BUNDLE.is_file():
        raise SystemExit(f"missing bundle {BUNDLE}; run rr20260919_v_export_evaluator_bundle.py first")

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_common as C
    import fc20260915_data as D

    results = json.loads((WAVE4 / "results.json").read_text(encoding="utf-8"))
    calib = results["selection"]["A_MLP_expanded"]["calib"]
    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load splits…", flush=True)
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    SE = D.load_v_rows(L, "MAIN", ["V_SELECT"], bucket_only=True)
    schema = FeatureSchema.from_names(TR["names"])
    stop = match_holdout_mask(TR["match"], 0.15, seed=7)
    bun = ProfileBundle(schema, "expanded").fit(TR["X"][~stop])
    model_pack = joblib.load(WAVE4 / "models" / "A_MLP_expanded.joblib")

    rng = np.random.default_rng(0)
    n = min(args.n_check, len(SE["match"]))
    idx = np.sort(rng.choice(len(SE["match"]), size=n, replace=False))
    X = SE["X"][idx]

    # Reference: live wave-4 path
    print("score wave-4 live path…", flush=True)
    p_live = wave4_live_predict(X, bun, model_pack, calib)

    # Same-process bundle
    print("score same-process bundle…", flush=True)
    ev = load_evaluator(BUNDLE)
    p_same = predict_calibrated(ev, X)

    # Subprocess bundle-only
    print("score subprocess bundle-only…", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x_path = td / "X.npy"
        out_path = td / "p.npy"
        np.save(x_path, X)
        cmd = [
            sys.executable,
            str(WORKER),
            "--bundle",
            str(BUNDLE),
            "--X",
            str(x_path),
            "--out",
            str(out_path),
            "--calibrated",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit(f"subprocess failed: {r.returncode}")
        p_sub = np.load(out_path)

    def ok(a, b):
        return bool(np.allclose(a, b, atol=args.atol, rtol=0))

    def mad(a, b):
        return float(np.max(np.abs(a - b)))

    check: Dict[str, Any] = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        n_check=int(n),
        atol=args.atol,
        same_process_reload_vs_live=dict(ok=ok(p_same, p_live), max_abs=mad(p_same, p_live)),
        subprocess_bundle_vs_live=dict(ok=ok(p_sub, p_live), max_abs=mad(p_sub, p_live)),
        subprocess_vs_same_process=dict(ok=ok(p_sub, p_same), max_abs=mad(p_sub, p_same)),
        note=(
            "subprocess worker loads only evaluator_bundle + X; does not read TRAIN or refit ProfileBundle"
        ),
    )
    check["pass_all"] = (
        check["same_process_reload_vs_live"]["ok"]
        and check["subprocess_bundle_vs_live"]["ok"]
        and check["subprocess_vs_same_process"]["ok"]
    )

    out_dir = WAVE4 / "evaluators"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repro_subprocess_parity.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    (REPO / "docs" / "V_EVALUATOR_REPRO_SUBPROCESS_PARITY_20260920.json").write_text(
        json.dumps(check, indent=2) + "\n", encoding="utf-8"
    )

    md = REPO / "docs" / "V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md"
    prev = md.read_text(encoding="utf-8") if md.is_file() else ""
    # rewrite with accurate scope
    lines = [
        "# Evaluator bundle reproducibility — A_MLP_expanded",
        "",
        f"Updated: {check['generated']}",
        "",
        "**Fit scope (LOCKED):** `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`",
        "",
        "$$\\widehat{V}_{\\mathrm{fit85}}=g_{\\mathrm{V\\_CAL}}\\circ f_{\\mathrm{MLP,fit85}}\\circ T_{\\mathrm{fit85}}$$",
        "",
        "## Scope of checks",
        "",
        "| Check | Meaning | Status |",
        "|---|---|---|",
        "| Same-process reload | Load bundle twice in one process | done earlier (diff=0) |",
        f"| **Bundle vs wave-4 live path** | Rebuild TRAIN fit85 `ProfileBundle` + `A_MLP_expanded.joblib` + calib vs bundle | "
        f"**{check['same_process_reload_vs_live']['ok']}** (max abs={check['same_process_reload_vs_live']['max_abs']:.3e}) |",
        f"| **Fresh OS subprocess** | Worker gets **only** bundle + fixed `X.npy` (no TRAIN, no refit) | "
        f"**{check['subprocess_bundle_vs_live']['ok']}** (max abs={check['subprocess_bundle_vs_live']['max_abs']:.3e}) |",
        "",
        f"**PASS all:** **{check['pass_all']}** (n={n}, atol={args.atol})",
        "",
        "## Artifacts",
        "",
        f"- Bundle: `outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib`",
        f"- JSON: `docs/V_EVALUATOR_REPRO_SUBPROCESS_PARITY_20260920.json`",
        f"- Worker: `scripts/rr20260919_v_score_bundle_worker.py`",
        "",
        "## Reading",
        "",
        "- Earlier ‘reload PASS’ was **same-process only**; this update closes **subprocess + live-path parity**.",
        "- Continuity / TEST band tables must score through this bundle (or an identical hash).",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(check, indent=2))
    print("wrote", md)
    return 0 if check["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
