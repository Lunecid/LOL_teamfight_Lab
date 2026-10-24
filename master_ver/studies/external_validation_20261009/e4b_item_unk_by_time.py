"""Post-hoc (DECISION_LOG_KO.md 10-09 21:59): V on 16.13 nominal rows with and without unseen current items, inside
5-minute game-time bins, so that the item-UNK gap of e4_v_shift_diagnosis.py (D) is not a game-time artifact.
No training.  Writes reports/external_validation_20261009/V_ITEM_UNK_BY_TIME_20261009.json once.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from studies.external_validation_20261009 import e4_v_shift_diagnosis as D  # noqa: E402

OUT = ROOT / "reports/external_validation_20261009/V_ITEM_UNK_BY_TIME_20261009.json"


def main() -> int:
    from studies.external_validation_20261009.e2a_unk import _vocab_sets
    from studies.q_production_20261005 import qeval

    if OUT.exists():
        raise FileExistsError(OUT)
    vocab = _vocab_sets()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "status": "post-hoc exploration", "sets": {}}
    for name in ("kr_16_13", "na1_16_13"):
        s = D.load_set(name)
        item = D.item_unseen_per_query(s, vocab)[s["q"]] > 0
        out = {}
        for lo, hi in D.BINS:
            in_bin = (s["minutes"] >= lo) & (s["minutes"] < hi)
            entry = {"rows": int(in_bin.sum()), "item_unseen_share": float(item[in_bin].mean()) if in_bin.any() else None}
            for label, mask in (("item_unseen", in_bin & item), ("no_item_unseen", in_bin & ~item)):
                if mask.sum() >= 300 and len(np.unique(s["y"][mask])) == 2:
                    ev = qeval.evaluate(s["y"][mask], s["p"][mask], s["m"][mask])["match_weighted"]
                    entry[label] = {"rows": int(mask.sum()), "auc": ev["auc"], "calibration_slope": ev["calibration_slope"]}
                else:
                    entry[label] = {"rows": int(mask.sum())}
            out[D.bin_label(lo, hi)] = entry
        report["sets"][name] = out
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, bins in report["sets"].items():
        print("==", name)
        for b, e in bins.items():
            a, n = e["item_unseen"], e["no_item_unseen"]
            fmt = lambda x: f"{x['rows']:6d} auc {x['auc']:.3f} slope {x['calibration_slope']:.2f}" if "auc" in x else f"{x['rows']:6d}"
            print(f"  {b:9s} share {e['item_unseen_share']:.3f} | unseen {fmt(a)} | none {fmt(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
