#!/usr/bin/env python3
"""E4 §7 — horizon strata + peer LR label-transfer (frozen q/PT).

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §7
Task: .ai/tasks/T020.md

Does NOT: refit q/PT on peer labels; retune V; rebuild e_fixed/OAT without match packs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WAVE4 = REPO / "outputs/v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
LR_PATH = WAVE4 / "models" / "A_LR_expanded.joblib"
PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
PRED_S = REPO / "outputs/review_response_rr12_20260920_S_qS_id/prediction_table.npz"
DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E4_SENSITIVITY_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E4_SENSITIVITY_20260921.md"
HS = (60, 90, 120)
SMALL_DV = 0.05
AGE_CUT_S = 30.0


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
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = match_weights(g)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def cohort_keys(data_root: Path, set_name: str, cohort: str) -> set:
    coh = np.load(
        data_root
        / "outputs"
        / "cohort_role_training_20260915"
        / "cohorts"
        / f"{set_name}_cohort.npz",
        allow_pickle=False,
    )
    if cohort == "T":
        m = coh["cohort"] == 1
    else:
        m = (coh["cohort"] == 0) & (coh["fine"] == 1)
    return set(
        zip(coh["match"][m].astype(str).tolist(), coh["s"][m].astype(np.int64).tolist())
    )


def load_pred(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return dict(
        match=z["match"].astype(str),
        s=z["s"].astype(np.int64),
        y=z["y"].astype(np.float64),
        q=z["p_q_base"].astype(np.float64),
        pt=z["p_PT_flex"].astype(np.float64),
        p_pre=z["p_pre"].astype(np.float64),
        delta_V=z["delta_V"].astype(np.float64),
    )


def join_pred(
    match: np.ndarray, s: np.ndarray, pred: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    key = {(a, int(b)): i for i, (a, b) in enumerate(zip(pred["match"].tolist(), pred["s"].tolist()))}
    rows, idxs = [], []
    for i, (a, b) in enumerate(zip(match.tolist(), s.tolist())):
        j = key.get((a, int(b)))
        if j is not None:
            rows.append(i)
            idxs.append(j)
    return np.asarray(rows, int), np.asarray(idxs, int)


def scope_masks(
    p_pre: np.ndarray,
    dV: np.ndarray,
    same: np.ndarray,
    tmin: np.ndarray,
) -> Dict[str, np.ndarray]:
    b40 = (p_pre >= 0.40) & (p_pre <= 0.60)
    return {
        "all": np.ones(len(p_pre), dtype=bool),
        "B40": b40,
        "small_abs_dV": np.abs(dV) < SMALL_DV,
        "same_frame": same.astype(bool),
        "new_frame": ~same.astype(bool),
        "t_2_10": (tmin >= 2.0) & (tmin < 10.0),
        "t_10_20": (tmin >= 10.0) & (tmin < 20.0),
        "t_20_30": (tmin >= 20.0) & (tmin < 30.0),
        "t_30_inf": tmin >= 30.0,
    }


def pairwise_horizon(
    Ys: Dict[int, np.ndarray],
    dVs: Dict[int, np.ndarray],
    endpoints: Dict[int, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    if not mask.any():
        return dict(n=0)
    out: Dict[str, Any] = dict(n=int(mask.sum()))
    flips = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        ya, yb = Ys[a][mask], Ys[b][mask]
        da, db = dVs[a][mask], dVs[b][mask]
        mnz = (np.sign(da) != 0) & (np.sign(db) != 0)
        flips[f"h{a}_vs_h{b}"] = dict(
            svi_flip_rate=float(np.mean(ya != yb)),
            mean_abs_delta_diff=float(np.mean(np.abs(da - db))),
            sign_agree_nonzero=float(np.mean(np.sign(da)[mnz] == np.sign(db)[mnz])) if mnz.any() else None,
        )
    out["pairwise_svi"] = flips
    ep = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        ea, eb = endpoints[a][mask], endpoints[b][mask]
        same_ep = ea == eb
        ya, yb = Ys[a][mask], Ys[b][mask]
        ep[f"h{a}_vs_h{b}"] = dict(
            share_same_endpoint=float(same_ep.mean()),
            among_same_endpoint_svi_flip=float(np.mean(ya[same_ep] != yb[same_ep])) if same_ep.any() else None,
            among_diff_endpoint_svi_flip=float(np.mean(ya[~same_ep] != yb[~same_ep])) if (~same_ep).any() else None,
        )
    out["endpoint_identity"] = ep
    return out


def label_transfer_block(
    match: np.ndarray,
    y_main: np.ndarray,
    y_alt: np.ndarray,
    q: np.ndarray,
    pt: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    if not mask.any():
        return dict(n=0)
    m = match[mask]
    ym, ya = y_main[mask], y_alt[mask]
    qq, ppt = q[mask], pt[mask]
    flip = float(np.mean(ym != ya))
    return dict(
        n=int(mask.sum()),
        n_matches=int(len(np.unique(m))),
        label_flip_vs_main=flip,
        brier_q_on_main=brier(ym, qq, m),
        brier_q_on_alt=brier(ya, qq, m),
        brier_pt_on_main=brier(ym, ppt, m),
        brier_pt_on_alt=brier(ya, ppt, m),
        delta_brier_q_minus_pt_on_main=brier(ym, qq, m) - brier(ym, ppt, m),
        delta_brier_q_minus_pt_on_alt=brier(ya, qq, m) - brier(ya, ppt, m),
        P_SVI_main=float(np.mean(ym > 0)),
        P_SVI_alt=float(np.mean(ya > 0)),
    )


def horizon_cohort(
    cohort: str,
    E: Dict[str, Any],
    keys: set,
    pred: Dict[str, np.ndarray],
    ev,
) -> Dict[str, Any]:
    from v_redesign_evaluator_bundle import predict_calibrated

    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    q_pre = E["q_pre"].astype(np.int64)
    keep = (E["pre_ok"] == 1)
    for h in HS:
        keep &= E[f"valid_h{h}"] == 1
    in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    if in_coh[keep].sum() == 0:
        in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), q_pre.tolist())], dtype=bool)
        key_s = q_pre
    else:
        key_s = es
    keep &= in_coh
    idx_e = np.where(keep)[0]
    if len(idx_e) == 0:
        return dict(ok=False, reason="empty_common_valid")

    rows, idxs = join_pred(em[idx_e], key_s[idx_e], pred)
    if len(idxs) == 0:
        return dict(ok=False, reason="no_pred_join")
    idx_e = idx_e[rows]

    Xpre = E["X_pre"][idx_e]
    p0 = predict_calibrated(ev, Xpre)
    Ys, dVs, endpoints = {}, {}, {}
    for h in HS:
        p1 = predict_calibrated(ev, E[f"X_post_h{h}"][idx_e])
        dV = p1 - p0
        dVs[h] = dV
        Ys[h] = (dV > 0).astype(np.float64)
        endpoints[h] = E[f"endpoint_h{h}"][idx_e].astype(np.int64)

    match = pred["match"][idxs]
    y_main = pred["y"][idxs]
    q = pred["q"][idxs]
    pt = pred["pt"][idxs]
    p_pre = pred["p_pre"][idxs]
    dV_main = pred["delta_V"][idxs]
    same = E["pre_snapshot"][idx_e] == E["post_snapshot_h90"][idx_e]
    tmin = key_s[idx_e].astype(np.float64) / 60000.0

    # consistency: main Y vs recomputed h90 from frozen MLP
    y_h90 = Ys[90]
    agree_main = float(np.mean(y_main == y_h90))

    scopes = scope_masks(p_pre, dV_main, same, tmin)
    strata = {}
    transfer = {}
    for name, mask in scopes.items():
        strata[name] = pairwise_horizon(Ys, dVs, endpoints, mask)
        # label transfer: frozen q/PT vs Y at each horizon
        transfer[name] = {
            f"Y_h{h}": label_transfer_block(match, y_main, Ys[h], q, pt, mask) for h in HS
        }

    return dict(
        ok=True,
        cohort=cohort,
        n_common=int(len(idx_e)),
        n_matches=int(len(np.unique(match))),
        join_rate=float(len(idxs) / max(int(keep.sum()), 1)),
        y_main_vs_Y_h90_agree=agree_main,
        small_abs_dV_cut=SMALL_DV,
        strata=strata,
        frozen_pred_label_transfer=transfer,
        reading=(
            "Horizon flips and frozen-score label transfer are diagnostics; "
            "not claims that changing the horizon redesigns the method. "
            "Separate same-endpoint vs diff-endpoint flip rates (§7.3)."
        ),
    )


def predict_lr_expanded(X: np.ndarray, bun, lr, cal: Dict[str, float]) -> np.ndarray:
    from v_redesign_evaluator_bundle import PosSlopeSigmoid

    raw = lr.predict_proba(bun.matrix_onehot(X))[:, 1]
    return PosSlopeSigmoid.from_dict(cal).transform(raw)


def peer_label_transfer(
    data_root: Path,
    E: Dict[str, Any],
    keys: set,
    pred: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    import joblib
    import fc20260915_common as C
    import fc20260915_data as D
    from v_redesign_feature_adapters import FeatureSchema, ProfileBundle, match_holdout_mask

    if not LR_PATH.is_file():
        return dict(ok=False, reason="missing_A_LR_expanded")

    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    q_pre = E["q_pre"].astype(np.int64)
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    if in_coh[keep].sum() == 0:
        in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), q_pre.tolist())], dtype=bool)
        key_s = q_pre
    else:
        key_s = es
    keep &= in_coh
    idx_e = np.where(keep)[0]
    rows, idxs = join_pred(em[idx_e], key_s[idx_e], pred)
    if len(idxs) == 0:
        return dict(ok=False, reason="no_pred_join")
    idx_e = idx_e[rows]

    print("  fit ProfileBundle on TRAIN fit85 for peer LR…", flush=True)
    t0 = time.time()
    L = D.Layout(False)
    TR = D.load_v_rows(L, "MAIN", [f"fold{k}" for k in range(C.N_FOLDS)], bucket_only=True)
    schema = FeatureSchema.from_names(TR["names"])
    stop = match_holdout_mask(TR["match"], 0.15, seed=7)
    bun = ProfileBundle(schema, "expanded").fit(TR["X"][~stop])
    lr = joblib.load(LR_PATH)["model"]
    w4 = json.loads((WAVE4 / "results.json").read_text(encoding="utf-8"))
    cal = w4["selection"]["A_LR_expanded"]["calib"]
    print(f"  bundle fit wall_s={time.time()-t0:.1f}", flush=True)

    print("  score peer LR on TEST engagements…", flush=True)
    p_pre_p = predict_lr_expanded(E["X_pre"][idx_e], bun, lr, cal)
    p_post_p = predict_lr_expanded(E["X_post_h90"][idx_e], bun, lr, cal)
    dV_p = p_post_p - p_pre_p
    y_peer = (dV_p > 0).astype(np.float64)

    match = pred["match"][idxs]
    y_main = pred["y"][idxs]
    q = pred["q"][idxs]
    pt = pred["pt"][idxs]
    p_pre = pred["p_pre"][idxs]
    dV_main = pred["delta_V"][idxs]
    same = E["pre_snapshot"][idx_e] == E["post_snapshot_h90"][idx_e]
    tmin = key_s[idx_e].astype(np.float64) / 60000.0

    # sign agree among nonzero
    s_m, s_p = np.sign(dV_main), np.sign(dV_p)
    mnz = (s_m != 0) & (s_p != 0) & np.isfinite(s_m) & np.isfinite(s_p)

    scopes = scope_masks(p_pre, dV_main, same, tmin)
    by_scope = {name: label_transfer_block(match, y_main, y_peer, q, pt, mask) for name, mask in scopes.items()}

    return dict(
        ok=True,
        peer="A_LR_expanded_fit85",
        peer_path=str(LR_PATH.relative_to(REPO).as_posix()),
        n=int(len(idxs)),
        n_matches=int(len(np.unique(match))),
        sign_agree_nonzero=float(np.mean(s_m[mnz] == s_p[mnz])) if mnz.any() else None,
        n_sign_compared=int(mnz.sum()),
        label_flip_vs_main_all=float(np.mean(y_main != y_peer)),
        by_scope=by_scope,
        note=(
            "Frozen q/PT predictions scored against Y_peer; p_pre in q inputs was NOT replaced by peer. "
            "This is label-transfer diagnostics, not a peer-trained pipeline."
        ),
        wall_s=float(time.time() - t0),
    )


def incomplete_blocks(data_root: Path) -> Dict[str, Any]:
    cache = data_root / "outputs/full_corpus_training_20260915/runtime/cache"
    n_files = 0
    if cache.is_dir():
        n_files = sum(1 for _ in cache.glob("*") if _.is_file())
    reason = (
        "Match packs required for e_fixed StateBuilder rebuild and engagement-definition OAT "
        f"are unavailable (CACHE_DIR files={n_files}). Leave INCOMPLETE rather than invent endpoints."
    )
    return dict(
        e_fixed=dict(status="INCOMPLETE", reason=reason),
        definition_oat=dict(status="INCOMPLETE", reason=reason),
        cache_dir=str(cache),
        cache_n_files=n_files,
    )


def _md_table_horizon(strata: Dict[str, Any]) -> List[str]:
    lines = [
        "| Scope | n | flip 60–90 | flip 90–120 | same-ep 60–90 | flip|same-ep 60–90 | flip|diff-ep 60–90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in strata.items():
        if block.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        f = block["pairwise_svi"]
        ep = block["endpoint_identity"]["h60_vs_h90"]
        lines.append(
            f"| {name} | {block['n']} | {f['h60_vs_h90']['svi_flip_rate']:.4f} | "
            f"{f['h90_vs_h120']['svi_flip_rate']:.4f} | {ep['share_same_endpoint']:.3f} | "
            f"{ep['among_same_endpoint_svi_flip'] if ep['among_same_endpoint_svi_flip'] is not None else float('nan'):.4f} | "
            f"{ep['among_diff_endpoint_svi_flip'] if ep['among_diff_endpoint_svi_flip'] is not None else float('nan'):.4f} |"
        )
    return lines


def _md_table_transfer(transfer: Dict[str, Any], peer: bool = False) -> List[str]:
    lines = [
        "| Scope | n | flip vs Y_main | ΔBrier q−PT on main | ΔBrier q−PT on alt |",
        "|---|---:|---:|---:|---:|",
    ]
    if peer:
        for name, b in transfer.items():
            if b.get("n", 0) == 0:
                lines.append(f"| {name} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {name} | {b['n']} | {b['label_flip_vs_main']:.4f} | "
                f"{b['delta_brier_q_minus_pt_on_main']:.5f} | "
                f"{b['delta_brier_q_minus_pt_on_alt']:.5f} |"
            )
    else:
        # use Y_h60 as alt representative in compact table; detail in JSON
        for name, by_h in transfer.items():
            b = by_h["Y_h60"]
            if b.get("n", 0) == 0:
                lines.append(f"| {name} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {name} (alt=Y_h60) | {b['n']} | {b['label_flip_vs_main']:.4f} | "
                f"{b['delta_brier_q_minus_pt_on_main']:.5f} | "
                f"{b['delta_brier_q_minus_pt_on_alt']:.5f} |"
            )
    return lines


def to_md(doc: Dict[str, Any]) -> str:
    lines = [
        "# Supplementary E4 — Horizon strata + peer label-transfer",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**task:** {doc['task']}  ",
        f"**bundle_sha16:** `{doc['bundle_sha16']}`  ",
        "",
        "Frozen V/q/PT. Diagnostics only — not new journal headlines.",
        "",
        "## Incomplete (§7.3 e_fixed / §7.4 definition OAT)",
        "",
        f"- **e_fixed:** `{doc['incomplete']['e_fixed']['status']}`",
        f"- **definition OAT:** `{doc['incomplete']['definition_oat']['status']}`",
        "",
        doc["incomplete"]["e_fixed"]["reason"],
        "",
    ]
    for cohort in ("T", "S"):
        block = doc["horizon"][cohort]
        lines.append(f"## Horizon strata — cohort {cohort} (§7.1 / §7.3)")
        lines.append("")
        if not block.get("ok"):
            lines.append(f"FAIL: {block.get('reason')}")
            lines.append("")
            continue
        lines.append(
            f"n_common={block['n_common']} / {block['n_matches']} matches; "
            f"Y_main vs recomputed Y_h90 agree={block['y_main_vs_Y_h90_agree']:.6f}; "
            f"|ΔV| cut={block['small_abs_dV_cut']}."
        )
        lines.append("")
        lines.extend(_md_table_horizon(block["strata"]))
        lines.append("")
        lines.append("### Frozen q/PT label-transfer vs Y_h60 (compact)")
        lines.append("")
        lines.extend(_md_table_transfer(block["frozen_pred_label_transfer"], peer=False))
        lines.append("")
        lines.append(block["reading"])
        lines.append("")

    peer = doc["peer_label_transfer"]
    lines.append("## Peer LR label-transfer — T (§7.2)")
    lines.append("")
    if not peer.get("ok"):
        lines.append(f"FAIL: {peer.get('reason')}")
    else:
        lines.append(
            f"peer=`{peer['peer']}`; n={peer['n']}; "
            f"sign_agree_nonzero={peer['sign_agree_nonzero']}; "
            f"label_flip_all={peer['label_flip_vs_main_all']:.4f}."
        )
        lines.append("")
        lines.append(peer["note"])
        lines.append("")
        lines.extend(_md_table_transfer(peer["by_scope"], peer=True))
    lines.append("")
    lines.append("## Forbidden")
    lines.append("")
    for f in doc["forbidden"]:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D
    from v_redesign_evaluator_bundle import load_evaluator

    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")
    if not PRED_T.is_file():
        raise SystemExit(f"missing {PRED_T}")

    print("load evaluator + TEST engagements…", flush=True)
    ev = load_evaluator(BUNDLE)
    L = D.Layout(False)
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
    keys_T = cohort_keys(data_root, "MAIN_TEST", "T")
    keys_S = cohort_keys(data_root, "MAIN_TEST", "S")
    pred_T = load_pred(PRED_T)
    pred_S = load_pred(PRED_S) if PRED_S.is_file() else pred_T

    print("horizon strata T…", flush=True)
    hor_T = horizon_cohort("T", E, keys_T, pred_T, ev)
    print("horizon strata S…", flush=True)
    hor_S = horizon_cohort("S", E, keys_S, pred_S, ev)

    print("peer LR label-transfer T…", flush=True)
    peer = peer_label_transfer(data_root, E, keys_T, pred_T)

    incomplete = incomplete_blocks(data_root)

    doc = dict(
        schema="SUPPLEMENTARY_E4_SENSITIVITY_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#7",
        task=".ai/tasks/T020.md",
        bundle_sha16=sha16(BUNDLE),
        horizon=dict(T=hor_T, S=hor_S),
        peer_label_transfer=peer,
        incomplete=incomplete,
        forbidden=[
            "refit q/PT on peer labels in this task",
            "replace q input p_pre with peer V",
            "retune V for peer agreement",
            "fabricate e_fixed/OAT without match packs",
            "new journal headlines",
        ],
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    DOCS_MD.write_text(to_md(doc), encoding="utf-8")
    print(f"wrote {DOCS_JSON}", flush=True)
    print(f"wrote {DOCS_MD}", flush=True)
    if hor_T.get("ok"):
        print(
            f"T n_common={hor_T['n_common']} flip60-90(all)="
            f"{hor_T['strata']['all']['pairwise_svi']['h60_vs_h90']['svi_flip_rate']:.4f}",
            flush=True,
        )
    if peer.get("ok"):
        print(
            f"peer flip={peer['label_flip_vs_main_all']:.4f} "
            f"ΔBrier_alt(all)={peer['by_scope']['all']['delta_brier_q_minus_pt_on_alt']:.5f}",
            flush=True,
        )
    print(f"incomplete e_fixed/OAT status={incomplete['e_fixed']['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
