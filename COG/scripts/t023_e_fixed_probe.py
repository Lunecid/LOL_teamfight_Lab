#!/usr/bin/env python3
"""E4 §7.3 — e_fixed stopping-rule probe (CACHE_MAIN).

e_fixed = min(L + 30_000 ms, match_end − 1). No next-kill early stop.
Frozen fit85 V; frozen q/PT label-transfer. Task: .ai/tasks/T023.md
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WAVE4 = REPO / "outputs/v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E4_E_FIXED_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E4_E_FIXED_20260921.md"
HOLD_MS = 30_000


def data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    for x in Path.home().iterdir():
        cand = x / "LOL_Teamfight"
        if (cand / "outputs" / "full_corpus_training_20260915").is_dir():
            return cand
    raise SystemExit("data root not found")


def setup(root: Path) -> None:
    wt = root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ["LOL_OUTPUT_ROOT"] = str(root / "outputs" / "full_corpus_training_20260915" / "runtime")


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def brier(y, p, g) -> float:
    w = match_weights(g)
    return float(np.sum(w * (np.asarray(p, float) - np.asarray(y, float)) ** 2) / np.sum(w))


def game_end_ms(pack: dict) -> Optional[int]:
    info = pack.get("info") or {}
    if "gameDuration" in info and info["gameDuration"] is not None:
        v = int(info["gameDuration"])
        return v * 1000 if v < 10_000 else v
    for e in pack.get("events") or []:
        if e.get("type") == "GAME_END":
            return int(e.get("timestamp", 0) or 0)
    mts = pack.get("minute_ts")
    return int(mts[-1]) if mts is not None and len(mts) else None


def main() -> int:
    root = data_root()
    setup(root)
    import fc20260915_common as C
    import fc20260915_data as D
    import data.cache_io as cio
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value_v2 import StateBuilder, state_matrix
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    print(f"CACHE_DIR={cio.CACHE_DIR}", flush=True)
    ev = load_evaluator(BUNDLE)
    Lyt = D.Layout(False)
    E = D.load_engagements(Lyt, "MAIN", ["TEST"], states=True, counts=False)
    names = list(E["names"])
    pred = np.load(PRED_T, allow_pickle=True)
    pred_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(
            zip(pred["match"].astype(str).tolist(), pred["s"].astype(np.int64).tolist())
        )
    }
    coh = np.load(
        root / "outputs/cohort_role_training_20260915/cohorts/MAIN_TEST_cohort.npz",
        allow_pickle=False,
    )
    keys_T = set(
        zip(coh["match"][coh["cohort"] == 1].astype(str).tolist(), coh["s"][coh["cohort"] == 1].astype(np.int64).tolist())
    )

    em, es = E["match"].astype(str), E["s"].astype(np.int64)
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    in_t = np.array([(a, int(b)) in keys_T for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    keep &= in_t
    idx = np.where(keep)[0]
    rows, pidx = [], []
    for i in idx.tolist():
        j = pred_key.get((em[i], int(es[i])))
        if j is not None:
            rows.append(i)
            pidx.append(j)
    idx, pidx = np.asarray(rows, int), np.asarray(pidx, int)

    by_match: Dict[str, List[int]] = defaultdict(list)
    for local, i in enumerate(idx.tolist()):
        by_match[em[i]].append(local)

    y_h90, y_fix, q, pt, match = [], [], [], [], []
    len_fixed, len_h90, same_ep = [], [], []
    n_miss = n_fail = n_invalid = n_ok = 0
    t0 = time.time()
    mids = sorted(by_match.keys())
    for mi, mid in enumerate(mids):
        if mi % 500 == 0:
            print(f"  e_fixed {mi}/{len(mids)} ok={n_ok}", flush=True)
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_miss += 1
            continue
        gend = game_end_ms(pack)
        if gend is None:
            n_fail += 1
            continue
        try:
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
            last_frame = int(np.asarray(pack["minute_ts"], dtype=np.int64)[-1])
        except Exception:
            n_fail += 1
            continue
        for local in by_match[mid]:
            i = idx[local]
            j = pidx[local]
            q_pre = int(E["q_pre"][i])
            L = int(E["L"][i])
            ep90 = int(E["endpoint_h90"][i])
            e_fix = min(L + HOLD_MS, int(gend) - 1)
            if not (e_fix > q_pre and e_fix <= last_frame and e_fix >= L):
                n_invalid += 1
                continue
            try:
                s0 = builder.at(q_pre)
                s_f = builder.at(e_fix)
                X = state_matrix([s0, s_f], names)
                p = predict_calibrated(ev, X)
                a, pf = map(float, p)
            except Exception:
                n_fail += 1
                continue
            n_ok += 1
            dV = pf - a
            y_fix.append(1.0 if dV > 0 else 0.0)
            y_h90.append(float(pred["y"][j]))
            q.append(float(pred["p_q_base"][j]))
            pt.append(float(pred["p_PT_flex"][j]))
            match.append(str(pred["match"][j]))
            len_fixed.append((e_fix - q_pre) / 1000.0)
            len_h90.append((ep90 - q_pre) / 1000.0)
            same_ep.append(int(e_fix == ep90))

    y_h90 = np.asarray(y_h90)
    y_fix = np.asarray(y_fix)
    q = np.asarray(q)
    pt = np.asarray(pt)
    match = np.asarray(match)
    flip = float(np.mean(y_h90 != y_fix)) if n_ok else float("nan")

    doc = dict(
        schema="SUPPLEMENTARY_E4_E_FIXED_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        task=".ai/tasks/T023.md",
        bundle_sha16=sha16(BUNDLE),
        cache_dir=str(cio.CACHE_DIR),
        definition="e_fixed=min(L+30000, match_end-1); no next-kill early stop",
        n_ok=n_ok,
        n_cache_miss=n_miss,
        n_fail=n_fail,
        n_invalid=n_invalid,
        wall_s=float(time.time() - t0),
        share_same_endpoint_as_h90=float(np.mean(same_ep)) if n_ok else None,
        mean_followup_s_fixed=float(np.mean(len_fixed)) if n_ok else None,
        mean_followup_s_h90=float(np.mean(len_h90)) if n_ok else None,
        svi_flip_vs_h90=flip,
        frozen_q_pt=dict(
            delta_brier_q_minus_pt_on_Y_h90=brier(y_h90, q, match) - brier(y_h90, pt, match),
            delta_brier_q_minus_pt_on_Y_fixed=brier(y_fix, q, match) - brier(y_fix, pt, match),
            label_flip=flip,
        ),
        reading=(
            "Alternative stopping rule diagnostic. Extra engagements may enter the window; "
            "attribution is broader than h90. Not a claim that e_fixed is better."
        ),
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Supplementary E4 — e_fixed stopping rule (§7.3)",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**bundle:** `{doc['bundle_sha16']}`  ",
        "",
        f"n_ok={n_ok}; miss={n_miss}; fail={n_fail}; invalid={n_invalid}; wall_s={doc['wall_s']:.1f}.",
        "",
        f"- same endpoint as h90: {doc['share_same_endpoint_as_h90']}",
        f"- mean followup s (fixed / h90): {doc['mean_followup_s_fixed']:.1f} / {doc['mean_followup_s_h90']:.1f}",
        f"- SVI flip vs h90: {flip:.4f}",
        f"- ΔBrier q−PT on Y_h90: {doc['frozen_q_pt']['delta_brier_q_minus_pt_on_Y_h90']:.5f}",
        f"- ΔBrier q−PT on Y_fixed: {doc['frozen_q_pt']['delta_brier_q_minus_pt_on_Y_fixed']:.5f}",
        "",
        doc["reading"],
        "",
    ]
    DOCS_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {DOCS_MD}", flush=True)
    print(f"flip={flip:.4f} same_ep={doc['share_same_endpoint_as_h90']}", flush=True)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
