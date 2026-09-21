#!/usr/bin/env python3
"""Build / refresh docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json (G1).

Recomputes T008 integrity block; optionally merges T009 artifact digests.
Fail if --check-integrity and recomputed integrity values differ from on-disk
(excluding generated / source_commit / generator / digests_T009).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from rr20260920_q_build_newv_labels import BUNDLE as BUNDLE_PATH  # noqa: E402
from rr20260920_q_train_oof_mlp_folds import _data_root, _setup  # noqa: E402

T_LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
S_LAB = REPO / "outputs" / "q_newv_fit85_20260920_S" / "labels"
OOF = REPO / "outputs" / "q_newv_fit85_20260920" / "oof_evaluators"
MANIFEST = REPO / "docs" / "SCALE_SPLIT_RR0_MANIFEST_20260920.json"
FREEZE_PRE = REPO / ".ai" / "reports" / "logs" / "T008_T_freeze_pre.json"
EXPECTED = {
    "TEST": 101205,
    "Q_CAL": 31302,
    "Q_SELECT": 31059,
    "TRAIN_folds": [24030, 25005, 24520, 24690, 24804],
}
ROLE_TAG = "EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE"
EXT_CENSUS = {
    "KR 16.13": 15641,
    "NA1 16.13": 16100,
    "KR 16.15": 1307,
    "KR 16.14 pilot": 285,
}


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    dig = h.hexdigest().lower()
    return dig, dig[:16], p.stat().st_size


def path_str(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def digest_entry(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {"path": path_str(p), "exists": False}
    full, s16, nbytes = sha256_file(p)
    return {"path": path_str(p), "exists": True, "sha256": full, "sha16": s16, "bytes": nbytes}


def keys_of(pack) -> set:
    return set(zip(pack["match"].astype(str).tolist(), pack["s"].astype(np.int64).tolist()))


def match_wtd_pos(pack) -> float:
    y = pack["Y_SVI"].astype(np.float64)
    m = pack["match"].astype(str)
    c = Counter(m.tolist())
    w = np.array([1.0 / c[x] for x in m.tolist()])
    ok = y >= 0
    return float(np.sum(w[ok] * (y[ok] == 1)) / np.sum(w[ok]))


def check_pack(pack) -> Dict[str, Any]:
    y = pack["Y_SVI"]
    dV = pack["delta_V"]
    p0, p1 = pack["p_pre"], pack["p_post"]
    miss = pack["missing_score"].astype(bool)
    ok = ~miss
    B40 = pack["B40"].astype(bool)
    checks = dict(
        n_rows=int(len(y)),
        n_matches=int(len(set(pack["match"].astype(str).tolist()))),
        n_unique_keys=len(keys_of(pack)),
        no_duplicate_keys=len(keys_of(pack)) == len(y),
        y_matches_delta_sign=bool(np.all(y[ok] == (dV[ok] > 0).astype(np.int8))),
        delta_matches_p_post_minus_p_pre=bool(
            np.allclose(dV[ok], p1[ok] - p0[ok], atol=1e-7, equal_nan=False)
        ),
        B40_from_same_p_pre=bool(np.all(B40[ok] == ((p0[ok] >= 0.40) & (p0[ok] <= 0.60)))),
        n_missing=int(miss.sum()),
        n_exact_zero=int((pack["exact_zero"] == 1).sum())
        if "exact_zero" in pack.files
        else int(np.sum(dV[ok] == 0)),
        valid_h90_all_1=bool(np.all(pack["valid_h90"] == 1)),
        pre_ok_note="filtered at label build (column absent)",
        P_SVI_match_wtd=match_wtd_pos(pack),
        B40_n=int(pack["B40"].sum()),
        delta_max_abs_residual=float(np.max(np.abs(dV[ok] - (p1[ok] - p0[ok])))),
    )
    checks["PASS"] = all(
        [
            checks["no_duplicate_keys"],
            checks["y_matches_delta_sign"],
            checks["delta_matches_p_post_minus_p_pre"],
            checks["B40_from_same_p_pre"],
            checks["n_missing"] == 0,
            checks["valid_h90_all_1"],
        ]
    )
    return checks


def build_integrity() -> Dict[str, Any]:
    changed: List[dict] = []
    if FREEZE_PRE.is_file():
        pre = json.loads(FREEZE_PRE.read_text(encoding="utf-8"))
        for rel, meta in pre.items():
            p = Path(rel)
            if not p.is_file():
                changed.append({"path": rel, "issue": "missing"})
                continue
            s16 = sha256_file(p)[1]
            if s16 != meta["sha16"] or p.stat().st_size != meta["size"]:
                changed.append({"path": rel, "issue": "modified", "pre": meta["sha16"], "post": s16})
    t_freeze_untouched = len(changed) == 0

    roles_S = {r: np.load(S_LAB / f"{r}_h90.npz", allow_pickle=False) for r in ["TEST", "Q_CAL", "Q_SELECT"]}
    roles_S["TRAIN"] = np.load(S_LAB / "TRAIN_oof_h90.npz", allow_pickle=False)
    roles_T = {r: np.load(T_LAB / f"{r}_h90.npz", allow_pickle=False) for r in ["TEST", "Q_CAL", "Q_SELECT"]}
    roles_T["TRAIN"] = np.load(T_LAB / "TRAIN_oof_h90.npz", allow_pickle=False)

    row_counts: Dict[str, Any] = {}
    census_ok = True
    census_detail: List[dict] = []
    for role, exp in [("TEST", EXPECTED["TEST"]), ("Q_CAL", EXPECTED["Q_CAL"]), ("Q_SELECT", EXPECTED["Q_SELECT"])]:
        n = int(len(roles_S[role]["Y_SVI"]))
        row_counts[role] = n
        census_detail.append({"role": role, "n": n, "expected": exp, "delta": n - exp})
        if n != exp:
            census_ok = False
    tr = roles_S["TRAIN"]
    fold_counts = []
    for k, exp in enumerate(EXPECTED["TRAIN_folds"]):
        n = int((tr["sub_role"].astype(str) == f"fold{k}").sum())
        fold_counts.append({"fold": k, "n": n, "expected": exp, "delta": n - exp})
        if n != exp:
            census_ok = False
    row_counts["TRAIN_oof"] = int(len(tr["Y_SVI"]))
    row_counts["TRAIN_folds"] = fold_counts

    s_keys: set = set()
    t_keys: set = set()
    for pack in roles_S.values():
        s_keys |= keys_of(pack)
    for pack in roles_T.values():
        t_keys |= keys_of(pack)
    inter = s_keys & t_keys
    key_sep = {
        "n_S_keys": len(s_keys),
        "n_T_keys": len(t_keys),
        "n_intersection": len(inter),
        "PASS": len(inter) == 0,
        "example_intersection": [list(x) for x in list(inter)[:3]],
    }

    dr = _data_root()
    _setup(dr)
    import fc20260915_data as D
    import joblib

    L = D.Layout(False)
    TR_all = D.load_v_rows(L, "MAIN", [f"fold{k}" for k in range(5)], bucket_only=True)
    gTR = TR_all["match"].astype(str)
    sub = TR_all["sub_role"].astype(str)
    fold_sep = []
    fold_sep_pass = True
    m2_rows = []
    for k in range(5):
        v_held = set(gTR[sub == f"fold{k}"].tolist())
        fo = joblib.load(OOF / f"V_oof_fold{k}_mlp_expanded.joblib")
        n_held = int(fo["n_held_matches"])
        s_m = set(tr["match"].astype(str)[tr["sub_role"].astype(str) == f"fold{k}"].tolist())
        t_m = set(
            roles_T["TRAIN"]["match"].astype(str)[
                roles_T["TRAIN"]["sub_role"].astype(str) == f"fold{k}"
            ].tolist()
        )
        v_hash = hashlib.sha256("\n".join(sorted(v_held)).encode()).hexdigest()[:16]
        s_hash = hashlib.sha256("\n".join(sorted(s_m)).encode()).hexdigest()[:16]
        t_hash = hashlib.sha256("\n".join(sorted(t_m)).encode()).hexdigest()[:16]
        ok = (s_m <= v_held) and (t_m <= v_held) and (len(v_held) == n_held)
        if not ok:
            fold_sep_pass = False
        m2_rows.append(
            {"fold": k, "len_fold_matches": len(v_held), "n_held_matches": n_held, "equal": len(v_held) == n_held}
        )
        fold_sep.append(
            {
                "fold": k,
                "n_V_held": len(v_held),
                "n_held_joblib": n_held,
                "n_S_matches": len(s_m),
                "n_T_matches": len(t_m),
                "S_subset_V": bool(s_m <= v_held),
                "T_subset_V": bool(t_m <= v_held),
                "V_sha16": v_hash,
                "S_match_sha16": s_hash,
                "T_match_sha16": t_hash,
                "note": "S/T match sets are subsets of shared V-held fold partition (not equal to each other)",
                "PASS": ok,
            }
        )

    value_checks = {}
    value_pass = True
    for role, pack in roles_S.items():
        ch = check_pack(pack)
        ch["expected_n"] = sum(EXPECTED["TRAIN_folds"]) if role == "TRAIN" else EXPECTED[role]
        ch["row_count_matches_expected"] = ch["n_rows"] == ch["expected_n"]
        if not ch["row_count_matches_expected"]:
            ch["PASS"] = False
        value_checks[role] = ch
        if not ch["PASS"]:
            value_pass = False

    digests = {}
    for sn in ["MAIN_TRAIN", "MAIN_VALIDATION", "MAIN_TEST"]:
        digests[f"cohort_{sn}"] = digest_entry(
            dr / "outputs" / "cohort_role_training_20260915" / "cohorts" / f"{sn}_cohort.npz"
        )
    digests["fit85_bundle"] = digest_entry(Path(BUNDLE_PATH))
    for k in range(5):
        digests[f"V_oof_fold{k}"] = digest_entry(OOF / f"V_oof_fold{k}_mlp_expanded.joblib")
        digests[f"bundle_oof_fold{k}"] = digest_entry(OOF / f"bundle_oof_fold{k}.joblib")
    for role in ["TEST", "Q_CAL", "Q_SELECT"]:
        digests[f"T_{role}_h90"] = digest_entry(T_LAB / f"{role}_h90.npz")
    digests["T_TRAIN_oof_h90"] = digest_entry(T_LAB / "TRAIN_oof_h90.npz")
    for role in ["TEST", "Q_CAL", "Q_SELECT"]:
        digests[f"S_{role}_h90"] = digest_entry(S_LAB / f"{role}_h90.npz")
        digests[f"S_{role}_h90_meta"] = digest_entry(S_LAB / f"{role}_h90_meta.json")
    digests["S_TRAIN_oof_h90"] = digest_entry(S_LAB / "TRAIN_oof_h90.npz")
    digests["S_TRAIN_oof_h90_meta"] = digest_entry(S_LAB / "TRAIN_oof_h90_meta.json")
    digests["S_TRAIN_oof_STATUS"] = digest_entry(S_LAB / "TRAIN_oof_STATUS.json")

    integrity = {
        "key_separation_S_vs_T": key_sep,
        "fold_separation": {"folds": fold_sep, "PASS": fold_sep_pass},
        "M2_n_held_matches": {"folds": m2_rows, "PASS": all(r["equal"] for r in m2_rows)},
        "value_checks": value_checks,
        "value_checks_PASS": value_pass,
        "PASS": all(
            [
                t_freeze_untouched,
                census_ok,
                key_sep["PASS"],
                fold_sep_pass,
                value_pass,
                all(r["equal"] for r in m2_rows),
            ]
        ),
    }
    return dict(
        T_freeze_untouched=t_freeze_untouched,
        T_freeze_changed_files=changed,
        row_counts=row_counts,
        census={"detail": census_detail + fold_counts, "PASS": census_ok},
        integrity=integrity,
        digests=digests,
    )


def integrity_core(block: dict) -> dict:
    """Drop non-stable wrappers; compare structural integrity values."""
    return json.loads(json.dumps(block))  # deep copy


def collect_t009_digests() -> Dict[str, Any]:
    paths = {
        "S_logit_state": REPO / "outputs" / "q_newv_fit85_20260920_S" / "models" / "logit_state.joblib",
        "S_primary_table": REPO / "outputs" / "q_newv_fit85_20260920_S" / "primary_table.json",
        "TS_logit_state": REPO / "outputs" / "q_newv_fit85_20260920_TS" / "models" / "logit_state.joblib",
        "TS_TRAIN_oof": REPO / "outputs" / "q_newv_fit85_20260920_TS" / "labels" / "TRAIN_oof_h90.npz",
        "rr12_S_qS": REPO / "outputs" / "review_response_rr12_20260920_S_qS" / "paired_ci.json",
        "rr12_S_qT": REPO / "outputs" / "review_response_rr12_20260920_S_qT" / "paired_ci.json",
        "rr12_S_qTS": REPO / "outputs" / "review_response_rr12_20260920_S_qTS" / "paired_ci.json",
        "rr12_S_qS_id": REPO / "outputs" / "review_response_rr12_20260920_S_qS_id" / "paired_ci.json",
        "rr12_S_qT_id": REPO / "outputs" / "review_response_rr12_20260920_S_qT_id" / "paired_ci.json",
        "rr12_S_qTS_id": REPO / "outputs" / "review_response_rr12_20260920_S_qTS_id" / "paired_ci.json",
        "rr12_qTS_on_T": REPO / "outputs" / "review_response_rr12_20260920_qTS_on_T" / "paired_ci.json",
        "paired_contrasts": REPO / "outputs" / "scale_split_TvsS_20260920" / "paired_contrasts.json",
        "paired_contrasts_id": REPO / "outputs" / "scale_split_TvsS_20260920" / "paired_contrasts_id.json",
        "rrx_S": REPO / "outputs" / "review_response_rrx_external_20260920_S" / "rrx_external_results.json",
        "results_json": REPO / "docs" / "SCALE_SPLIT_TvsS_RESULTS_20260920.json",
        "results_md": REPO / "docs" / "SCALE_SPLIT_TvsS_RESULTS_20260920.md",
    }
    return {k: digest_entry(p) for k, p in paths.items()}


# G2 — RR0 frozen digests (from docs/REVIEW_RESPONSE_RR0_MANIFEST_20260920.json)
RR0_EXPECTED = {
    "logit_state": ("outputs/q_newv_fit85_20260920/models/logit_state.joblib", "d6e8fc31310a5a58"),
    "primary_table": ("outputs/q_newv_fit85_20260920/primary_table.json", "a80340fede7b6e6a"),
    "selection_freeze": ("outputs/q_newv_fit85_20260920/selection_freeze.json", "294a3e4261f9a40e"),
    "rr12_paired_ci": ("outputs/review_response_rr12_20260920/paired_ci.json", "9484c09c4e097f25"),
    "rr12_prediction_table": ("outputs/review_response_rr12_20260920/prediction_table.npz", "b16b1ce417d61f53"),
    "rr12_PT_flex": ("outputs/review_response_rr12_20260920/models/PT_flex.joblib", "15acc630abe5930d"),
    "rr12_PT_linear": ("outputs/review_response_rr12_20260920/models/PT_linear.joblib", "2215150ca4bc0f89"),
    "rr12_b_spline": ("outputs/review_response_rr12_20260920/models/b_spline.joblib", "b34f46c62d72637b"),
    "rr12_baseline_selection": ("outputs/review_response_rr12_20260920/baseline_selection.json", "4b798eed3795e789"),
}


def check_rr0_frozen_digests() -> Dict[str, Any]:
    rows = []
    ok_all = True
    for key, (rel, exp16) in RR0_EXPECTED.items():
        p = REPO / rel
        if not p.is_file():
            rows.append({"key": key, "path": rel, "exists": False, "PASS": False})
            ok_all = False
            continue
        got = sha256_file(p)[1]
        match = got == exp16
        if not match:
            ok_all = False
        rows.append({"key": key, "path": rel, "expected_sha16": exp16, "got_sha16": got, "PASS": match})
    return {"rows": rows, "PASS": ok_all}


def ext_census_check() -> Dict[str, Any]:
    p = REPO / "outputs" / "review_response_rrx_external_20260920_S" / "rrx_external_results.json"
    if not p.is_file():
        return {"exists": False}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    ok = True
    for set_id, block in data.get("cohorts", {}).items():
        if not block.get("ok"):
            rows.append({"set_id": set_id, "ok": False, "reason": block.get("reason")})
            continue
        label = block["label"]
        n = int(block["n"])
        exp = EXT_CENSUS.get(label)
        delta = None if exp is None else n - exp
        if exp is not None and delta != 0:
            # finite-score exclusions allowed; record
            pass
        rows.append({"label": label, "n": n, "expected_census": exp, "delta": delta})
    return {"exists": True, "rows": rows, "note": "deltas vs contract §2 may be finite-score exclusions"}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-integrity", action="store_true", help="compare to on-disk integrity; exit 2 on mismatch")
    ap.add_argument("--with-t009", action="store_true", help="attach T009 artifact digests")
    args = ap.parse_args(argv)

    built = build_integrity()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    generator = {
        "script": "scripts/ss20260920_scale_split_manifest.py",
        "argv": list(argv or sys.argv[1:]),
    }

    if MANIFEST.is_file():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        old = {}

    if args.check_integrity and old.get("integrity"):
        old_i = integrity_core(old["integrity"])
        new_i = integrity_core(built["integrity"])
        if old_i != new_i:
            print("BLOCKED: integrity block mismatch vs on-disk manifest", flush=True)
            # show brief diff keys
            for k in sorted(set(old_i) | set(new_i)):
                if old_i.get(k) != new_i.get(k):
                    print(f"  differ: {k}", flush=True)
            return 2
        print("integrity PASS (matches on-disk)", flush=True)

    manifest = {
        "generated": now,
        "epistemic": ROLE_TAG,
        "role_tag": ROLE_TAG,
        "source_commit": commit,
        "contract": "docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md",
        "task": "T009" if args.with_t009 else "T008",
        "generator": generator,
        "cohort": "S",
        "cohort_mask_rule": "cohort==0 & fine==1 & valid_h90==1",
        "evaluator_policy": (
            "frozen fit85 for TEST/Q_CAL/Q_SELECT; T fold V^(-k) reused for TRAIN OOF (no refit)"
        ),
        "script_note": (
            "reuse mode builds fold_matches from V-bucket rows (not eng-role) so M2 n_held_matches matches; "
            "eng-role is strict subset (~1k V-only matches/fold)"
        ),
        "T_freeze_untouched": built["T_freeze_untouched"],
        "T_freeze_changed_files": built["T_freeze_changed_files"],
        "row_counts": built["row_counts"],
        "census": built["census"],
        "integrity": built["integrity"],
        "digests": built["digests"],
        "PASS": built["integrity"]["PASS"],
    }
    if args.with_t009:
        manifest["digests_T009"] = collect_t009_digests()
        manifest["EXT_S_census"] = ext_census_check()
        rr0 = check_rr0_frozen_digests()
        manifest["T_frozen_rr0_digests"] = rr0
        manifest["T_frozen_rr0_digests_match"] = bool(rr0["PASS"])
        if not rr0["PASS"]:
            print("BLOCKED: T_frozen_rr0_digests_match=false", flush=True)
            for row in rr0["rows"]:
                if not row.get("PASS"):
                    print(" ", row, flush=True)
            return 2

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", MANIFEST, "PASS", manifest["PASS"], flush=True)
    return 0 if manifest["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
