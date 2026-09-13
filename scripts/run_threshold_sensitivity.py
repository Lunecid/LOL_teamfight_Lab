"""Do the engagement-detection thresholds drive the results?

CoG 2026 review R1 flagged the two detection constants as unexplained:
kills within TF2_KILL_CLUSTER_GAP_MS (18 s) form one temporal cluster, and
clusters wider than CLUSTER_MAX_DIAMETER (4,000 game units) are split
spatially.  This sweep re-detects engagements under one perturbed constant
at a time and reports what actually matters downstream:

    how many engagements exist
    how many of the default setting's engagements survive (+-5 s match)
    the label's positive rate
    nested-CV AUC of the same tabular representation

Each invocation runs ONE setting so no state leaks between settings:
the fight-index disk cache is disabled and detection runs serially, so
the runtime config override provably reaches the detector.  Drive the
sweep with repeated invocations (see docs/TOG_EXTENSION_PLAN.md):

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_threshold_sensitivity.py ^
        --gap-ms 24000 --n-matches 553 --seed 7 ^
        --output <...>/threshold_gap24000.json

Superseded for corpus v3.3.  Without ``--preset`` or LOL_CFG_PRESET this runs the CFG defaults (the CoG 2026 constants:
G 18 s, D 4,000 u, R 1,800 u, B 10 s, Eq.3 label, match-length time_norm and match-wide anchors), which is
how features/thresholds/*.json were produced, and its AUC is a random-split nested CV on 553 matches.  The
v3.3 G x D sensitivity (clean features, market_event with draws dropped, patch holdout, match-clustered
intervals, one-to-one overlap) is scripts/run_gd_sweep_v33.py.  ``--preset NAME`` selects a core/presets.py
preset before core.config is imported; whenever a preset is in effect (flag or inherited LOL_CFG_PRESET) the output
records it and the effective constants.  Without a preset the output schema is the original one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATCH_TOLERANCE_MS = 5000


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "rfe", PROJECT_ROOT / "scripts" / "run_fusion_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def overlap_fraction(reference: list[tuple[str, int]], candidate: list[tuple[str, int]]) -> float:
    """Fraction of reference engagements matched in candidate within tolerance."""
    by_match: dict[str, list[int]] = {}
    for match_id, ts in candidate:
        by_match.setdefault(match_id, []).append(ts)
    hit = 0
    for match_id, ts in reference:
        stamps = by_match.get(match_id, ())
        if any(abs(ts - other) <= MATCH_TOLERANCE_MS for other in stamps):
            hit += 1
    return hit / max(1, len(reference))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-ms", type=int, default=None,
                        help="override TF2_KILL_CLUSTER_GAP_MS (default keeps 18000)")
    parser.add_argument("--diameter", type=float, default=None,
                        help="override CLUSTER_MAX_DIAMETER (default keeps 4000)")
    parser.add_argument("--n-matches", type=int, default=553)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference-refs", type=Path, default=None,
                        help="refs JSON from the default run, for overlap")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", default=None,
                        help="core/presets.py preset applied before core.config is imported (default: CFG defaults)")
    args = parser.parse_args(argv)

    if args.preset:
        if "core.config" in sys.modules:
            raise SystemExit("core.config was imported before --preset could be applied")
        os.environ["LOL_CFG_PRESET"] = str(args.preset)
    # an inherited LOL_CFG_PRESET applies too (core.config reads it at import), so it is recorded either way
    preset_in_effect = os.environ.get("LOL_CFG_PRESET", "").strip() or None
    from core.config import CACHE_DIR, cfg

    # the override must reach the detector: no disk cache, no worker processes
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.FIGHT_INDEX_NUM_WORKERS = 1
    cfg.DUMP_FIGHTS = False
    if args.gap_ms is not None:
        cfg.TF2_KILL_CLUSTER_GAP_MS = int(args.gap_ms)
    if args.diameter is not None:
        cfg.CLUSTER_MAX_DIAMETER = float(args.diameter)

    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy

    mids = [p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json")]
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    refs = build_fight_index(cache_match_ids=mids)
    keys = [(r.match_id, int(r.t_start_ts)) for r in refs]
    print(f"gap_ms={getattr(cfg, 'TF2_KILL_CLUSTER_GAP_MS')} "
          f"diameter={getattr(cfg, 'CLUSTER_MAX_DIAMETER')} | refs={len(refs)}")

    X, y, names, used = build_tabular_Xy(refs, feature_set="full")
    groups = np.array([r.match_id for r in used])
    rfe = load_fusion_module()
    pred = rfe.oof_predictions(X, y, groups)

    results = {
        "gap_ms": int(getattr(cfg, "TF2_KILL_CLUSTER_GAP_MS")),
        "diameter": float(getattr(cfg, "CLUSTER_MAX_DIAMETER")),
        "n_matches": len(mids),
        "n_refs": len(refs),
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, pred)),
        "refs": [{"match_id": m, "t_start_ts": t} for m, t in keys],
    }
    if preset_in_effect:
        results["preset"] = preset_in_effect
        results["effective_cfg"] = {k: getattr(cfg, k, None) for k in (
            "TF2_KILL_CLUSTER_GAP_MS", "CLUSTER_MAX_DIAMETER", "TF2_VALIDITY_RADIUS", "TF2_ENGAGE_PRE_KILL_MS",
            "TF2_MIN_PER_TEAM", "FIGHT_HORIZON_SEC", "LABEL_TYPE", "LABEL_TIE_STRATEGY", "TIME_NORM_ABSOLUTE",
            "ANCHORS_CAUSAL", "TAB_FRAME_AGE_FEATURE")}
    if args.reference_refs and args.reference_refs.exists():
        reference = [
            (r["match_id"], int(r["t_start_ts"]))
            for r in json.loads(args.reference_refs.read_text(encoding="utf-8"))["refs"]
        ]
        results["overlap_of_default_refs"] = overlap_fraction(reference, keys)
        print(f"overlap of default engagements: {results['overlap_of_default_refs']:.3f}")
    print(f"rows={len(y)} positives={y.mean():.3f} AUC={results['auc']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
