#!/usr/bin/env python3
"""Assemble docs/SCALE_SPLIT_TvsS_RESULTS_20260920.{md,json} from frozen JSON fields."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def fmt(x, nd=5):
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def fmt4(x):
    return fmt(x, 4)


def find_model(d, name):
    if isinstance(d, dict):
        if name in d and isinstance(d[name], dict) and "brier" in d[name]:
            return d[name]
        for v in d.values():
            r = find_model(v, name)
            if r:
                return r
    return None


def main() -> int:
    pc = json.loads(
        (REPO / "outputs/scale_split_TvsS_20260920/paired_contrasts.json").read_text(encoding="utf-8")
    )
    rrx = json.loads(
        (
            REPO / "outputs/review_response_rrx_external_20260920_S/rrx_external_results.json"
        ).read_text(encoding="utf-8")
    )
    prim = json.loads(
        (REPO / "outputs/q_newv_fit85_20260920_S/primary_table.json").read_text(encoding="utf-8")
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    t1 = dict(pc["table1_S_TEST"])
    lgbm = find_model(prim, "lgbm_state")
    if lgbm:
        t1["lgbm_state_S"] = {
            k: lgbm[k] for k in ("n", "n_matches", "brier", "logloss", "auc") if k in lgbm
        }

    cmap = {c["name"]: c for c in pc["contrasts"]}
    results = dict(
        generated=now,
        epistemic="EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE",
        role_tag="EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE",
        contract="docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md",
        source_commit=commit,
        note_T_numbers=(
            "T numerical headlines cited from frozen RR12/RRX packs; "
            "not recomputed here except q_TS-on-T secondary."
        ),
        table1_S_TEST=t1,
        table2_contrasts={
            k: {
                "primary": c["primary"],
                "n": c["n"],
                "n_matches": c["n_matches"],
                "seed": c["seed"],
                "boot_reps": c["boot_reps"],
                "estimate": c["delta_brier"]["estimate"],
                "ci95": c["delta_brier"]["ci95"],
                "p_gt0": c["delta_brier"]["p_gt0"],
                "note": c["note"],
            }
            for k, c in cmap.items()
        },
        table3_CORP=pc["CORP"],
        table4_EXT_S={
            sid: {
                "label": b.get("label"),
                "ok": b.get("ok"),
                "n": b.get("n"),
                "V_pre_brier": (b.get("V_to_W") or {}).get("pre", {}).get("brier")
                if b.get("ok")
                else None,
                "delta_brier_q_minus_PT": (b.get("q_to_SVI") or {}).get(
                    "delta_brier_q_minus_PT_flex"
                )
                if b.get("ok")
                else None,
                "delta_MCB": (b.get("q_to_SVI") or {}).get("delta_MCB_q_minus_PT")
                if b.get("ok")
                else None,
                "delta_DSC": (b.get("q_to_SVI") or {}).get("delta_DSC_q_minus_PT")
                if b.get("ok")
                else None,
                "pilot": b.get("pilot"),
            }
            for sid, b in rrx["cohorts"].items()
        },
        sources=dict(
            paired_contrasts="outputs/scale_split_TvsS_20260920/paired_contrasts.json",
            rrx_S="outputs/review_response_rrx_external_20260920_S/rrx_external_results.json",
            primary_S="outputs/q_newv_fit85_20260920_S/primary_table.json",
            rr12_S_qS="outputs/review_response_rr12_20260920_S_qS/paired_ci.json",
        ),
    )

    lines = [
        "# Scale-split results — teamfight T vs skirmish S under frozen fit85 V",
        "",
        "**role_tag:** `EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE`  ",
        "**contract:** [SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md](SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md)  ",
        f"**source_commit:** `{commit}`  ",
        f"**generated:** {now}  ",
        "",
        "T headline numbers are **cited from frozen** RR12/RRX artifacts. "
        "This pack evaluates S arms and the predeclared §5 contrasts only.",
        "",
        "## Table 1 — S TEST point metrics",
        "",
        "| Model | n | matches | Brier | logloss | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = [
        "q_S",
        "q_T_to_S",
        "q_TS",
        "PT_flex_S",
        "PT_linear_S",
        "b_spline_S",
        "b_linear_S",
        "lgbm_state_S",
    ]
    labels = {
        "q_S": "q_S",
        "q_T_to_S": "q_T→S",
        "q_TS": "q_TS",
        "PT_flex_S": "PT_flex_S",
        "PT_linear_S": "PT_linear_S",
        "b_spline_S": "b(p)_spline_S",
        "b_linear_S": "b(p)_linear_S",
        "lgbm_state_S": "lgbm_state_S (diagnostic)",
    }
    for k in order:
        if k not in t1:
            continue
        sc = t1[k]
        lines.append(
            f"| {labels.get(k, k)} | {sc['n']} | {sc['n_matches']} | "
            f"{fmt4(sc['brier'])} | {fmt4(sc['logloss'])} | {fmt4(sc['auc'])} |"
        )
    lines += [
        "",
        "Source: `outputs/scale_split_TvsS_20260920/paired_contrasts.json` → `table1_S_TEST` "
        "(lgbm diagnostic row from S fit outputs under `outputs/q_newv_fit85_20260920_S/`).",
        "",
        "## Table 2 — Contrasts (ΔBrier = A − B; negative ⇒ A better)",
        "",
        "| Contrast | role | estimate | CI95 | p_gt0 | n | matches | seed |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    pretty = {
        "q_S_minus_PT_flex_S": "q_S − PT_flex_S (S TEST)",
        "q_S_minus_q_T_to_S": "q_S − q_T→S (S TEST)",
        "q_S_minus_q_TS": "q_S − q_TS (S TEST)",
        "q_TS_minus_q_T": "q_TS − q_T (T TEST)",
        "q_S_minus_PT_flex_S_B40": "q_S − PT_flex_S (S∩B40)",
    }
    for name in [
        "q_S_minus_PT_flex_S",
        "q_S_minus_q_T_to_S",
        "q_S_minus_q_TS",
        "q_TS_minus_q_T",
        "q_S_minus_PT_flex_S_B40",
    ]:
        c = cmap[name]
        role = "**primary**" if c["primary"] else "secondary"
        d = c["delta_brier"]
        lines.append(
            f"| {pretty[name]} | {role} | {fmt(d['estimate'])} | "
            f"[{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}] | {fmt4(d['p_gt0'])} | "
            f"{c['n']} | {c['n_matches']} | {c['seed']} |"
        )
    lines += [
        "",
        "Source: `paired_contrasts.json` → `contrasts[*].delta_brier` "
        "(2000 match-cluster draws, seed 7, w=1/n_m).",
        "",
        "B40 cell is a **within-S** secondary contrast only "
        "(S TEST B40 n=31675 / 101205 ≈ 31.3%; not placed beside T B40).",
        "",
        "## Table 3 — CORP point values (S TEST; no intervals)",
        "",
        "| Model | n | MCB | DSC | UNC |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab in ["q_S", "PT_flex_S"]:
        c = pc["CORP"][lab]
        corp = c["CORP"]
        lines.append(
            f"| {lab} | {c['n']} | {fmt(corp['MCB'], 4)} | {fmt(corp['DSC'], 4)} | {fmt(corp['UNC'], 4)} |"
        )
    lines += [
        "",
        "Source: `paired_contrasts.json` → `CORP` via `forecast_diagnostics.metrics_bundle` "
        "(score-gap MCB/DSC).",
        "",
        "## Table 4 — EXT S (score-only; no CI; small cohorts reported, not pooled)",
        "",
        "| Cohort | n | V_pre Brier | ΔBrier(q_S − PT_flex_S) | ΔMCB | ΔDSC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sid in ["KR_16.13", "NA1_16.13", "KR_16.15", "KR_16.14_pilot"]:
        b = rrx["cohorts"][sid]
        if not b.get("ok"):
            lines.append(f"| {b.get('label', sid)} | FAIL | — | — | — | — |")
            continue
        vp = b["V_to_W"]["pre"]["brier"]
        qsv = b["q_to_SVI"]
        lines.append(
            f"| {b['label']} | {b['n']} | {fmt4(vp)} | "
            f"{fmt(qsv['delta_brier_q_minus_PT_flex'], 4)} | "
            f"{fmt(qsv['delta_MCB_q_minus_PT'], 4)} | "
            f"{fmt(qsv['delta_DSC_q_minus_PT'], 4)} |"
        )
    lines += [
        "",
        "Source: `outputs/review_response_rrx_external_20260920_S/rrx_external_results.json` → `cohorts`.",
        "Census vs contract §2: KR 15641 / NA1 16100 / KR16.15 1307 / pilot 285 "
        "(exact match on common-valid rows).",
        "",
        "## Allowed / Forbidden readings",
        "",
        "Copied from contract §6 (Forbidden readings):",
        "",
        "- Comparing S and T absolute Brier/AUC as an \"improvement\", a \"scale gradient\", "
        "or \"which engagements are more predictable\".",
        "- Any mechanism story for a T/S difference (claim ledger X-31 stays withdrawn).",
        "- Promoting a secondary contrast, a λ filter, a bin or a learner because of its TEST value.",
        "- Substituting all-N or pick results for S, or pooling the small external cohorts into a success claim.",
        "- Touching the frozen T artifacts, the journal manuscript, or any lock document before the results report is reviewed.",
        "",
        "### Forbidden (quoted phrases kept only in this section)",
        "",
        "> scale gradient; more predictable; routing; deploy",
        "",
        "## Suggestions (not executed)",
        "",
        "- Whether a shared PT_flex object should be frozen once on S Q_SELECT and reused for all S arms "
        "(current runs re-select independently; grids identical).",
        "- Whether EXT S ΔBrier sign vs T EXT should be discussed only after a predeclared transfer protocol "
        "(not in this contract).",
        "",
    ]

    (REPO / "docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "docs/SCALE_SPLIT_TvsS_RESULTS_20260920.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("wrote results; primary", cmap["q_S_minus_PT_flex_S"]["delta_brier"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
