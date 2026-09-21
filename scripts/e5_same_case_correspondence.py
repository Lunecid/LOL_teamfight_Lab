#!/usr/bin/env python3
"""E5 §8.1 same-case SVI↔material correspondence + §8.2 nextobj association.

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §8
Task: .ai/tasks/T021.md

Reuses RR5 material axes and frozen next_objective_labels.npz.
Does NOT: claim SVI is a better value metric; refit CoG market_event without packs;
train R0/R1 without TRAIN nextobj.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
PRED_S = REPO / "outputs/review_response_rr12_20260920_S_qS_id/prediction_table.npz"
NEXTOBJ = REPO / "outputs/review_response_rr5_rr6b_20260920/next_objective_labels.npz"
DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.md"
SAMPLES_CSV = REPO / "docs/SUPPLEMENTARY_E5_DISAGREE_SAMPLES_20260921.csv"
N_SAMPLES_PER_CELL = 8


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


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


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
        y=z["y"].astype(np.int64),
        q=z["p_q_base"].astype(np.float64),
        pt=z["p_PT_flex"].astype(np.float64),
        p_pre=z["p_pre"].astype(np.float64),
        delta_V=z["delta_V"].astype(np.float64),
    )


def join_pred(match, s, pred):
    key = {(a, int(b)): i for i, (a, b) in enumerate(zip(pred["match"].tolist(), pred["s"].tolist()))}
    rows, idxs = [], []
    for i, (a, b) in enumerate(zip(match.tolist(), s.tolist())):
        j = key.get((a, int(b)))
        if j is not None:
            rows.append(i)
            idxs.append(j)
    return np.asarray(rows, int), np.asarray(idxs, int)


def slot_diff(X, names, field):
    ix = {n: i for i, n in enumerate(names)}
    blue = [ix[f"participant_slot{s}_{field}"] for s in range(0, 5)]
    red = [ix[f"participant_slot{s}_{field}"] for s in range(5, 10)]
    return X[:, blue].sum(axis=1) - X[:, red].sum(axis=1)


def count_net(during, after, keys, blue, red):
    ki = {str(k): i for i, k in enumerate(keys)}
    C = during + after
    return C[:, ki[blue]].astype(float) - C[:, ki[red]].astype(float)


def material_axes(E, idx_e, names):
    keys = [str(k) for k in E["count_keys"]]
    during = E["during"][idx_e]
    after = E["after_h90"][idx_e]

    def net(b, r):
        return count_net(during, after, keys, b, r)

    epic = (
        net("baron_blue", "baron_red")
        + net("dragon_blue", "dragon_red")
        + net("elder_blue", "elder_red")
        + net("herald_blue", "herald_red")
        + net("horde_blue", "horde_red")
        + net("atakhan_blue", "atakhan_red")
        + net("soul_owned_blue", "soul_owned_red")
    )
    struct = net("tower_blue", "tower_red") + net("inhibitor_blue", "inhibitor_red")
    Xpre, Xpost = E["X_pre"][idx_e], E["X_post_h90"][idx_e]
    kill_diff = slot_diff(Xpost, names, "kills") - slot_diff(Xpre, names, "kills")
    alive_post = slot_diff(Xpost, names, "alive")
    same_frame = E["post_snapshot_h90"][idx_e] == E["pre_snapshot"][idx_e]
    return dict(
        epic_net=epic,
        structure_net=struct,
        objective_net=epic + struct,
        kill_diff=kill_diff,
        alive_diff_post=alive_post,
        same_frame=same_frame.astype(bool),
    )


def crosstab(y_svi, material, g) -> Dict[str, Any]:
    y = np.asarray(y_svi, int)
    s = np.sign(np.asarray(material, float)).astype(int)
    g = np.asarray(g).astype(str)
    cells = {}
    for ys in (0, 1):
        for ms in (-1, 0, 1):
            cells[f"SVI{ys}_mat{ms}"] = int(((y == ys) & (s == ms)).sum())
    decided = s != 0
    y_pm = np.where(y == 1, 1, -1)
    if decided.any():
        disagree_uw = float(np.mean(y_pm[decided] != s[decided]))
        agree_w = float(np.average(y_pm[decided] == s[decided], weights=match_weights(g[decided])))
    else:
        disagree_uw = float("nan")
        agree_w = float("nan")
    return dict(
        n=int(len(y)),
        n_decided=int(decided.sum()),
        n_tie=int((~decided).sum()),
        tie_share=float((~decided).mean()) if len(y) else float("nan"),
        cells=cells,
        agreement_rate_match_weighted_decided=agree_w,
        disagreement_rate_unweighted_decided=disagree_uw,
    )


def kill_disagree_with_obj(y, kill, obj, g) -> Dict[str, Any]:
    y = np.asarray(y, int)
    sk = np.sign(np.asarray(kill, float)).astype(int)
    so = np.sign(np.asarray(obj, float)).astype(int)
    y_pm = np.where(y == 1, 1, -1)
    decided_k = sk != 0
    disagree = decided_k & (y_pm != sk)
    n_d = int(disagree.sum())
    return dict(
        n_kill_decided=int(decided_k.sum()),
        n_kill_svi_disagree=n_d,
        among_disagree_obj_agrees_SVI=int((disagree & (so != 0) & (so == y_pm)).sum()),
        among_disagree_obj_agrees_kill=int((disagree & (so != 0) & (so == sk)).sum()),
        among_disagree_obj_tie=int((disagree & (so == 0)).sum()),
        SVI_blue_kill_red=int((disagree & (y == 1) & (sk < 0)).sum()),
        SVI_red_kill_blue=int((disagree & (y == 0) & (sk > 0)).sum()),
        note="Denominator = kill-decided SVI≠kill cases; not causal attribution.",
    )


def hash_order(match: np.ndarray, s: np.ndarray) -> np.ndarray:
    keys = [f"{a}|{int(b)}".encode() for a, b in zip(match.tolist(), s.tolist())]
    h = np.array([hashlib.sha256(k).hexdigest() for k in keys], dtype=object)
    return np.argsort(h)


def sample_cells(
    match, s, y, kill, obj, alive, p_pre, dV, q, cells_spec: List[Tuple[str, np.ndarray]], n_each: int
) -> List[Dict[str, Any]]:
    rows = []
    order = hash_order(match, s)
    for name, mask in cells_spec:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # stable hash order among cell
        idx_set = set(idx.tolist())
        ranked = [i for i in order.tolist() if i in idx_set]
        for i in ranked[:n_each]:
            rows.append(
                dict(
                    cell=name,
                    match=str(match[i]),
                    s=int(s[i]),
                    Y_SVI=int(y[i]),
                    kill_diff=float(kill[i]),
                    objective_net=float(obj[i]),
                    alive_diff_post=float(alive[i]),
                    p_pre=float(p_pre[i]),
                    delta_V=float(dV[i]),
                    q=float(q[i]),
                    trace_note=(
                        "query→V/q from frozen tables; event-prefix / frame dump INCOMPLETE (no match packs)."
                    ),
                )
            )
    return rows


def nextobj_association(
    match, s, y, kill, b40, nextobj_path: Path
) -> Dict[str, Any]:
    if not nextobj_path.is_file():
        return dict(ok=False, reason="missing_nextobj")
    z = np.load(nextobj_path, allow_pickle=True)
    key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(zip(z["match"].astype(str).tolist(), z["s"].astype(np.int64).tolist()))
    }
    lab = np.empty(len(match), dtype=object)
    ok = np.zeros(len(match), dtype=bool)
    for i, (a, b) in enumerate(zip(match.tolist(), s.tolist())):
        j = key.get((a, int(b)))
        if j is not None:
            lab[i] = str(z["label"][j])
            ok[i] = True
    if not ok.any():
        return dict(ok=False, reason="no_join")

    def side_from_svi(yy):
        return np.where(yy == 1, "Blue", "Red")

    def side_from_kill(kk):
        out = np.array(["tie"] * len(kk), dtype=object)
        out[kk > 0] = "Blue"
        out[kk < 0] = "Red"
        return out

    def block(mask, scope):
        m = mask & ok
        labs = lab[m]
        counts = {k: int((labs == k).sum()) for k in sorted(set(labs.tolist()))}
        decided = np.isin(labs, ["Blue", "Red"])
        n_dec = int(decided.sum())
        if n_dec == 0:
            return dict(scope=scope, n=int(m.sum()), counts=counts, n_decided_BlueRed=0)
        ys = side_from_svi(y[m][decided])
        ks = side_from_kill(kill[m][decided])
        ls = labs[decided]
        kill_dec = ks != "tie"
        return dict(
            scope=scope,
            n=int(m.sum()),
            counts=counts,
            n_decided_BlueRed=n_dec,
            selected_denominator="Blue|Red only; none/end/tie excluded",
            SVI_agree_rate=float(np.mean(ys == ls)),
            kill_agree_rate_among_kill_decided=float(np.mean(ks[kill_dec] == ls[kill_dec]))
            if kill_dec.any()
            else None,
            n_kill_decided_in_selected=int(kill_dec.sum()),
            SVI_Blue_rate_among_selected=float(np.mean(ls == "Blue")),
        )

    return dict(
        ok=True,
        source=str(nextobj_path.relative_to(REPO).as_posix()),
        n_joined=int(ok.sum()),
        all_T=block(np.ones(len(match), bool), "all_T"),
        B40=block(b40.astype(bool), "B40"),
        SVI_pos=block(y == 1, "SVI_pos"),
        SVI_neg=block(y == 0, "SVI_neg"),
        reading=(
            "Association of frozen SVI/kill-net with first elite in (endpoint,+180s]. "
            "Not q accuracy; not causal ATT. Reuses RR5b labels — no new cache scan."
        ),
    )


def analyze_cohort(cohort, E, keys, pred) -> Dict[str, Any]:
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

    names = list(E["names"])
    axes = material_axes(E, idx_e, names)
    match = pred["match"][idxs]
    s = pred["s"][idxs]
    y = pred["y"][idxs]
    q = pred["q"][idxs]
    p_pre = pred["p_pre"][idxs]
    dV = pred["delta_V"][idxs]
    b40 = (p_pre >= 0.40) & (p_pre <= 0.60)

    axes_report = {}
    for name in ("kill_diff", "epic_net", "structure_net", "objective_net", "alive_diff_post"):
        axes_report[name] = dict(
            all_T=crosstab(y, axes[name], match),
            B40=crosstab(y[b40], axes[name][b40], match[b40]) if b40.any() else dict(n=0),
        )

    kill = axes["kill_diff"]
    obj = axes["objective_net"]
    alive = axes["alive_diff_post"]
    sk = np.sign(kill).astype(int)
    y_pm = np.where(y == 1, 1, -1)
    kill_dec = sk != 0
    disagree = kill_dec & (y_pm != sk)
    kill0 = sk == 0

    cells_spec = [
        ("kill_plus_SVI_minus", (sk > 0) & (y == 0)),
        ("kill_minus_SVI_plus", (sk < 0) & (y == 1)),
        ("kill_tie", kill0),
        ("kill_svi_disagree_any", disagree),
    ]
    samples = sample_cells(match, s, y, kill, obj, alive, p_pre, dV, q, cells_spec, N_SAMPLES_PER_CELL)

    return dict(
        ok=True,
        cohort=cohort,
        n=int(len(y)),
        n_matches=int(len(np.unique(match))),
        same_frame_share=float(axes["same_frame"].mean()),
        material_axes=axes_report,
        kill_disagree_obj=kill_disagree_with_obj(y, kill, obj, match),
        hash_samples=samples,
        nextobj=nextobj_association(match, s, y, kill, b40, NEXTOBJ) if cohort == "T" else dict(ok=False, reason="T_only"),
        label_functions=dict(
            SVI="1[delta_V>0] from frozen fit85 A_MLP_expanded; delta_V==0 → Y=0; missing excluded upstream",
            kill_diff="sum(kills)_blue−sum(kills)_red at post_h90 minus same at pre (state slots)",
            alive_diff_post="sum(alive)_blue−sum(alive)_red at post_h90",
            objective_net="during+after_h90 elite+structure count nets (blue−red)",
            CoG_market_event="INCOMPLETE — requires match event packs; not approximated",
        ),
        reading=(
            "Same-engagement correspondence only. Material features overlap V inputs. "
            "Do not call this independent fight-winner accuracy or causal effect."
        ),
    )


def incomplete_blocks(data_root: Path) -> Dict[str, Any]:
    cache = data_root / "outputs/full_corpus_training_20260915/runtime/cache"
    n = sum(1 for _ in cache.glob("*") if _.is_file()) if cache.is_dir() else 0
    return dict(
        CoG_market_event=dict(
            status="INCOMPLETE",
            reason=f"market_event needs event packs (CACHE_DIR files={n}).",
        ),
        R0_R1_outcome_model=dict(
            status="INCOMPLETE",
            reason=(
                "Multinomial R0/R1 requires TRAIN nextobj labels with OOF ΔV; "
                "only TEST nextobj npz exists. Association tables use TEST labels only."
            ),
        ),
        event_prefix_trace=dict(
            status="INCOMPLETE",
            reason="Match packs unavailable for query→event-prefix dumps in hash samples.",
        ),
    )


def to_md(doc: Dict[str, Any]) -> str:
    lines = [
        "# Supplementary E5 — Same-case correspondence (§8.1) + nextobj association",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**task:** {doc['task']}  ",
        "",
        "Diagnostics only. Not a claim that SVI is a better value metric than CoG labels.",
        "",
        "## Incomplete",
        "",
    ]
    for k, v in doc["incomplete"].items():
        lines.append(f"- **{k}:** `{v['status']}` — {v['reason']}")
    lines.append("")

    for cohort in ("T", "S"):
        b = doc["cohorts"][cohort]
        lines.append(f"## Cohort {cohort}")
        lines.append("")
        if not b.get("ok"):
            lines.append(f"FAIL: {b.get('reason')}")
            lines.append("")
            continue
        lines.append(f"n={b['n']} / {b['n_matches']} matches; same_frame={b['same_frame_share']:.3f}.")
        lines.append("")
        lines.append("### Label functions")
        lines.append("")
        for k, v in b["label_functions"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")
        lines.append("### Material crosstabs (match-weighted agree on decided)")
        lines.append("")
        lines.append("| Axis | scope | n_decided | tie_share | agree_w | disagree_uw |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for axis, scopes in b["material_axes"].items():
            for scope, ct in scopes.items():
                if ct.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {axis} | {scope} | {ct['n_decided']} | {ct['tie_share']:.3f} | "
                    f"{ct['agreement_rate_match_weighted_decided']:.3f} | "
                    f"{ct['disagreement_rate_unweighted_decided']:.3f} |"
                )
        lines.append("")
        kd = b["kill_disagree_obj"]
        lines.append("### Kill–SVI disagreement × objective")
        lines.append("")
        lines.append(
            f"n_disagree={kd['n_kill_svi_disagree']} / kill_decided={kd['n_kill_decided']}; "
            f"obj agrees SVI={kd['among_disagree_obj_agrees_SVI']}, "
            f"obj agrees kill={kd['among_disagree_obj_agrees_kill']}, "
            f"obj tie={kd['among_disagree_obj_tie']}."
        )
        lines.append("")
        lines.append(b["reading"])
        lines.append("")

        if b.get("nextobj", {}).get("ok"):
            nx = b["nextobj"]
            lines.append("### Next elite association (TEST labels, selected Blue|Red)")
            lines.append("")
            lines.append(nx["reading"])
            lines.append("")
            lines.append("| Scope | n | n_BlueRed | SVI_agree | kill_agree (kill-decided) |")
            lines.append("|---|---:|---:|---:|---:|")
            for key in ("all_T", "B40", "SVI_pos", "SVI_neg"):
                x = nx[key]
                ka = x.get("kill_agree_rate_among_kill_decided")
                ka_s = f"{ka:.3f}" if ka is not None else "—"
                lines.append(
                    f"| {key} | {x['n']} | {x['n_decided_BlueRed']} | "
                    f"{x.get('SVI_agree_rate', float('nan')):.3f} | {ka_s} |"
                )
            lines.append("")

    lines.append("## Hash samples")
    lines.append("")
    lines.append(f"Pre-registered cells; sha256(match|s) order; ≤{N_SAMPLES_PER_CELL}/cell. See `{SAMPLES_CSV.name}`.")
    lines.append("")
    lines.append("## Forbidden")
    lines.append("")
    for f in doc["forbidden"]:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_samples_csv(samples: List[Dict[str, Any]], path: Path) -> None:
    if not samples:
        path.write_text("cell,match,s\n", encoding="utf-8")
        return
    keys = list(samples[0].keys())
    lines = [",".join(keys)]
    for r in samples:
        lines.append(",".join(str(r[k]).replace(",", ";") for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    print("load TEST engagements (states+counts)…", flush=True)
    L = D.Layout(False)
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=True)
    keys_T = cohort_keys(data_root, "MAIN_TEST", "T")
    keys_S = cohort_keys(data_root, "MAIN_TEST", "S")
    pred_T = load_pred(PRED_T)
    pred_S = load_pred(PRED_S) if PRED_S.is_file() else pred_T

    print("analyze T…", flush=True)
    block_T = analyze_cohort("T", E, keys_T, pred_T)
    print("analyze S…", flush=True)
    block_S = analyze_cohort("S", E, keys_S, pred_S)

    samples = []
    if block_T.get("ok"):
        samples.extend(block_T["hash_samples"])
    write_samples_csv(samples, SAMPLES_CSV)

    # strip samples from JSON body size — keep path reference
    if block_T.get("ok"):
        block_T = dict(block_T)
        block_T["hash_samples_n"] = len(block_T.pop("hash_samples"))
        block_T["hash_samples_path"] = str(SAMPLES_CSV.relative_to(REPO).as_posix())
    if block_S.get("ok"):
        block_S = dict(block_S)
        block_S["hash_samples_n"] = len(block_S.pop("hash_samples"))

    doc = dict(
        schema="SUPPLEMENTARY_E5_CORRESPONDENCE_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#8",
        task=".ai/tasks/T021.md",
        cohorts=dict(T=block_T, S=block_S),
        incomplete=incomplete_blocks(data_root),
        forbidden=[
            "SVI is a better value metric than CoG",
            "independent fight-winner accuracy",
            "causal ATT from nextobj association",
            "fabricate market_event without packs",
            "train R0/R1 on TEST-only as confirmatory",
            "new journal headlines",
        ],
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    DOCS_MD.write_text(to_md(doc), encoding="utf-8")
    print(f"wrote {DOCS_JSON}", flush=True)
    print(f"wrote {DOCS_MD}", flush=True)
    print(f"wrote {SAMPLES_CSV}", flush=True)
    if block_T.get("ok"):
        k = block_T["material_axes"]["kill_diff"]["all_T"]
        print(
            f"T kill agree_w={k['agreement_rate_match_weighted_decided']:.3f} "
            f"n_dec={k['n_decided']}",
            flush=True,
        )
        if block_T["nextobj"].get("ok"):
            print(
                f"nextobj SVI_agree={block_T['nextobj']['all_T']['SVI_agree_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
