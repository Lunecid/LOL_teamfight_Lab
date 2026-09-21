#!/usr/bin/env python3
"""Rescore engagement labels with frozen redesign V (shared_lgbm + calib).

Writes NEW lineage labels under outputs/v_redesign_20260919/labels/ — does not
overwrite full_corpus_training_20260915/labels (old V).

Groups:
  - test_external: MAIN TEST (+ EXT if present) with final frozen V
  - validation: VAL role slices with final frozen V
  - train_diagnostic: TRAIN with final V (diagnostic only; OOF adapters deferred)

Contract: freeze_manifest.json; docs/V_REDESIGN_CONTRACT_20260919.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
HS = (60, 90, 120)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(data_root / "worktrees" / "engagement-state-value"))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


class PosSlopeSigmoid:
    def __init__(self, coef: float, intercept: float, ok: bool = True):
        self.coef_ = coef
        self.intercept_ = intercept
        self.ok = ok

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        z = self.coef_ * logit + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))


def expanded_X(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    return X[:, keep].astype(np.float64, copy=False), cols, champ_ix


def predict_lgbm(clf, X, cols, champ_ix):
    import pandas as pd
    df = pd.DataFrame(X, columns=list(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in champ_ix:
            df[cols[j]] = df[cols[j]].astype("category")
        return clf.predict_proba(df)[:, 1]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_block(X, names, pack, calib) -> np.ndarray:
    Xe, cols, champ_ix = expanded_X(X, names)
    assert cols == pack["cols"], "feature column mismatch vs frozen pack"
    raw = predict_lgbm(pack["model"], Xe, cols, champ_ix)
    return calib.transform(raw)


def label_set(
    *,
    L,
    D,
    C,
    set_id: str,
    sub_roles: List[str],
    out_name: str,
    pack: dict,
    calib: PosSlopeSigmoid,
    names: List[str],
    out_dir: Path,
    adapter_tag: str,
    adapter_sha: str,
) -> Dict[str, Any]:
    print(f"load engagements {out_name}…", flush=True)
    E = D.load_engagements(L, set_id, sub_roles, states=True, counts=True)
    n = len(E["match"])
    if n == 0:
        return dict(rows=0)

    p_pre = np.full(n, np.nan)
    p_post = {h: np.full(n, np.nan) for h in HS}
    pre_ok = E["pre_ok"] == 1
    if pre_ok.any():
        p_pre[pre_ok] = score_block(E["X_pre"][pre_ok], names, pack, calib)
    for h in HS:
        v = E[f"valid_h{h}"] == 1
        if v.any():
            p_post[h][v] = score_block(E[f"X_post_h{h}"][v], names, pack, calib)

    arrays: Dict[str, Any] = dict(
        match=E["match"],
        sub_role=E["sub_role"],
        adapter_id=np.full(n, adapter_tag, dtype="U32"),
        adapter_sha256=np.full(n, adapter_sha, dtype="U64"),
        p_pre=p_pre,
        model_version=np.asarray("v_redesign_shared_lgbm_20260919"),
        state_version=np.asarray(C.STATE_VERSION),
        v_lineage=np.asarray("shared_lgbm_freeze_provisional"),
    )
    for k in D.E_SCALAR + ("pre_reason",):
        arrays[k] = E[k]

    checks = {}
    for h in HS:
        v = E[f"valid_h{h}"] == 1
        delta = np.where(v, p_post[h] - p_pre, np.nan)
        Y = np.where(v, (delta > 0).astype(np.int8), -1).astype(np.int8)
        checks[f"h{h}"] = dict(
            valid_rows=int(v.sum()),
            finite_valid=bool(np.isfinite(delta[v]).all()) if v.any() else True,
            positive=int(np.sum(Y[v] == 1)) if v.any() else 0,
            exact_zero_delta=int(np.sum(delta[v] == 0)) if v.any() else 0,
            mean_abs_delta=float(np.mean(np.abs(delta[v]))) if v.any() else None,
        )
        arrays[f"endpoint_h{h}"] = E[f"endpoint_h{h}"]
        arrays[f"valid_h{h}"] = E[f"valid_h{h}"]
        arrays[f"p_post_h{h}"] = p_post[h]
        arrays[f"delta_h{h}"] = delta
        arrays[f"Y_h{h}"] = Y
        arrays[f"after_counts_h{h}"] = E[f"after_h{h}"]

    arrays["during_counts"] = E["during"]
    arrays["count_keys"] = np.asarray(E["count_keys"])

    path = out_dir / f"{out_name}_labels.npz"
    np.savez_compressed(path, **arrays)
    sha = file_sha256(path)
    print(f"  wrote {path.name} n={n} h90_pos={checks['h90']['positive']}", flush=True)
    return dict(rows=n, path=str(path), sha256=sha, checks=checks)


def compare_old(new_path: Path, old_path: Path, cohort_path: Optional[Path]) -> Dict[str, Any]:
    if not new_path.is_file() or not old_path.is_file():
        return dict(skipped=True)
    N = np.load(new_path, allow_pickle=False)
    O = np.load(old_path, allow_pickle=False)
    # align by (match,s)
    n_key = list(zip(N["match"].astype(str).tolist(), N["s"].astype(np.int64).tolist()))
    o_map = {k: i for i, k in enumerate(zip(O["match"].astype(str).tolist(), O["s"].astype(np.int64).tolist()))}
    idx_n, idx_o = [], []
    for i, k in enumerate(n_key):
        j = o_map.get(k)
        if j is not None:
            idx_n.append(i)
            idx_o.append(j)
    idx_n = np.asarray(idx_n, int)
    idx_o = np.asarray(idx_o, int)
    m = (N["valid_h90"][idx_n] == 1) & (O["valid_h90"][idx_o] == 1)
    yn = N["Y_h90"][idx_n][m]
    yo = O["Y_h90"][idx_o][m]
    out = dict(
        aligned=int(len(idx_n)),
        valid_h90=int(m.sum()),
        svi_agree=float(np.mean(yn == yo)) if m.any() else None,
        mean_abs_delta_diff=float(np.mean(np.abs(N["delta_h90"][idx_n][m] - O["delta_h90"][idx_o][m]))) if m.any() else None,
        new_pos_rate=float(np.mean(yn == 1)) if m.any() else None,
        old_pos_rate=float(np.mean(yo == 1)) if m.any() else None,
    )
    if cohort_path and cohort_path.is_file():
        coh = np.load(cohort_path, allow_pickle=False)
        # cohort is parallel to OLD labels
        c = coh["cohort"][idx_o][m] == 1  # T
        if c.any():
            out["T_svi_agree"] = float(np.mean(yn[c] == yo[c]))
            out["T_n"] = int(c.sum())
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--v-dir", type=Path, default=REPO / "outputs" / "v_redesign_20260919")
    ap.add_argument(
        "--group",
        choices=("test", "validation", "train_diagnostic", "all"),
        default="test",
    )
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    v_dir = args.v_dir
    out_dir = v_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    freeze = json.loads((v_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    pack = joblib.load(v_dir / "models" / "shared_lgbm.joblib")
    model_path = v_dir / "models" / "shared_lgbm.joblib"
    adapter_sha = file_sha256(model_path)
    calib = PosSlopeSigmoid(
        coef=float(freeze["calibration"]["coef"]),
        intercept=float(freeze["calibration"]["intercept"]),
        ok=bool(freeze["calibration"]["ok"]),
    )

    L = D.Layout(False)
    # names from a small V load
    TE0 = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TE0["names"])
    del TE0

    summaries = {}
    groups = []
    if args.group in ("test", "all"):
        groups.append(("MAIN", ["TEST"], "MAIN_TEST"))
    if args.group in ("validation", "all"):
        groups.append(("MAIN", ["V_CAL", "V_SELECT", "Q_CAL", "Q_SELECT"], "MAIN_VAL"))
    if args.group in ("train_diagnostic", "all"):
        groups.append(("MAIN", ["TRAIN"], "MAIN_TRAIN_DIAG_FINAL_V"))

    for set_id, roles, out_name in groups:
        summaries[out_name] = label_set(
            L=L, D=D, C=C, set_id=set_id, sub_roles=roles, out_name=out_name,
            pack=pack, calib=calib, names=names, out_dir=out_dir,
            adapter_tag="shared_lgbm_final", adapter_sha=adapter_sha,
        )

    # Compare MAIN_TEST vs old V labels
    old_lab = data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz"
    coh = data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz"
    cmp = compare_old(out_dir / "MAIN_TEST_labels.npz", old_lab, coh if coh.is_file() else None)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        freeze=freeze["model_id"],
        model_sha256=adapter_sha,
        calibration=freeze["calibration"],
        groups=summaries,
        vs_old_v_MAIN_TEST=cmp,
        note="TRAIN_DIAG uses final V — not for q targets; OOF adapters still deferred",
    )
    (out_dir / "RELABEL_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# New-V engagement labels (shared_lgbm freeze)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        f"**Model sha256:** `{adapter_sha[:16]}…`",
        "",
        "## Groups",
        "",
    ]
    for name, blk in summaries.items():
        if not blk.get("rows"):
            lines.append(f"- `{name}`: empty")
            continue
        c = blk["checks"]["h90"]
        lines.append(
            f"- `{name}`: n={blk['rows']}, h90 valid={c['valid_rows']}, "
            f"pos={c['positive']}, mean|ΔV|={c['mean_abs_delta']:.4f}"
        )
    lines += ["", "## vs old V (MAIN_TEST h90)", ""]
    if cmp.get("skipped"):
        lines.append("skipped (missing files)")
    else:
        lines.append(f"- aligned valid h90: {cmp.get('valid_h90')}")
        lines.append(f"- SVI sign agree: {cmp.get('svi_agree')}")
        if "T_svi_agree" in cmp:
            lines.append(f"- T only agree: {cmp.get('T_svi_agree')} (n={cmp.get('T_n')})")
        lines.append(f"- new/old pos rate: {cmp.get('new_pos_rate')} / {cmp.get('old_pos_rate')}")
        lines.append(f"- mean |ΔV_new − ΔV_old|: {cmp.get('mean_abs_delta_diff')}")
    lines += [
        "",
        "## Next",
        "",
        "- Fit TRAIN OOF adapters before q / TRAIN SVI targets.",
        "- Rebuild sealed primary / B40 tables under `outputs/svi_newv_*` using these labels.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_dir / "REPORT.md")
    print("vs old SVI agree:", cmp.get("svi_agree"), "T:", cmp.get("T_svi_agree"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
