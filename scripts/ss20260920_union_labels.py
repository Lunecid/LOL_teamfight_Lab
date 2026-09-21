#!/usr/bin/env python3
"""Concat T and S label NPZs into TS labels for pooled q_TS fit.

Contract: docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md §4
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
T_LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
S_LAB = REPO / "outputs" / "q_newv_fit85_20260920_S" / "labels"
OUT = REPO / "outputs" / "q_newv_fit85_20260920_TS" / "labels"
ROLE = "EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE"
ROLES = ("TRAIN_oof", "Q_CAL", "Q_SELECT", "TEST")


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def keys_of(pack) -> set:
    return set(zip(pack["match"].astype(str).tolist(), pack["s"].astype(np.int64).tolist()))


def load_role(lab: Path, role: str):
    name = "TRAIN_oof_h90.npz" if role == "TRAIN_oof" else f"{role}_h90.npz"
    return np.load(lab / name, allow_pickle=False), name


def concat_packs(a, b) -> Dict[str, np.ndarray]:
    keys_a, keys_b = keys_of(a), keys_of(b)
    inter = keys_a & keys_b
    if inter:
        raise SystemExit(f"BLOCKED: key overlap n={len(inter)} e.g. {next(iter(inter))}")
    out: Dict[str, np.ndarray] = {}
    # union of array fields present in both
    common = [k for k in a.files if k in b.files]
    for k in common:
        va, vb = a[k], b[k]
        if va.dtype.kind in ("U", "S", "O") or vb.dtype.kind in ("U", "S", "O"):
            out[k] = np.concatenate([va.astype(str), vb.astype(str)])
        else:
            out[k] = np.concatenate([va, vb])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_roles: List[dict] = []
    for role in ROLES:
        ta, tname = load_role(T_LAB, role)
        sa, sname = load_role(S_LAB, role)
        pack = concat_packs(ta, sa)
        out_name = tname  # same basename
        out_path = OUT / out_name
        np.savez_compressed(out_path, **pack)
        n = int(len(pack["Y_SVI"]))
        n_m = int(len(set(pack["match"].astype(str).tolist())))
        info = dict(
            role=role,
            n=n,
            n_matches=n_m,
            n_T=int(len(ta["Y_SVI"])),
            n_S=int(len(sa["Y_SVI"])),
            sha16=sha16(out_path),
            source_T=sha16(T_LAB / tname),
            source_S=sha16(S_LAB / sname),
        )
        meta_roles.append(info)
        print(f"  {role}: n={n} (= {info['n_T']}+{info['n_S']}) matches={n_m} sha16={info['sha16']}", flush=True)
        (OUT / out_name.replace(".npz", "_meta.json")).write_text(
            json.dumps(
                dict(
                    role=role,
                    cohort="TS",
                    cohort_mask_rule="union of T (cohort==1) and S (cohort==0 & fine==1)",
                    n=n,
                    n_matches=n_m,
                    generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                    epistemic=ROLE,
                    sources=info,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    status = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        roles=meta_roles,
        key_overlap=0,
        PASS=True,
        note="Match weights must be recomputed inside the union at fit time.",
    )
    (OUT / "UNION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    # TRAIN_oof_STATUS expected by some loaders
    (OUT / "TRAIN_oof_STATUS.json").write_text(
        json.dumps(dict(train_oof="UNION_OF_T_AND_S", **status), indent=2) + "\n",
        encoding="utf-8",
    )
    print("wrote", OUT, "PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
