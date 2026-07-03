"""score.py — Score human annotations against the teamfight_v2 detector.

Metrics:
  1. Annotator vs detector (per annotator, micro-averaged over matches):
     greedy temporal-IoU matching (default threshold 0.5) ->
     precision / recall / F1, mean IoU of matched pairs, mean |onset delta|.
     Recall is additionally reported restricted to KILL-CONTAINING annotated
     intervals — the detector is kill-conditioned by design, so annotated
     intervals without any CHAMPION_KILL are out of scope (reported separately
     as `killless_annotated`).
  2. Inter-annotator agreement (pairwise): IoU-matched F1 and Cohen's kappa
     on a 1-second binary grid over the match duration.

Inputs:
  --study_dir   directory produced by prepare.py (matches/ + detector/)
  --annotations one or more annotations_<id>.json files exported by viewer.html

Usage:
    python -m analysis.annotation_study.score \
        --study_dir outputs/annotation_study \
        --annotations annotations_annot1.json annotations_annot2.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Interval = Tuple[int, int]  # (start_ms, end_ms)


def temporal_iou(a: Interval, b: Interval) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def greedy_match(ref: Sequence[Interval], cand: Sequence[Interval],
                 iou_thr: float) -> List[Tuple[int, int, float]]:
    """One-to-one greedy matching by descending IoU. Returns (ref_i, cand_j, iou)."""
    pairs = [(i, j, temporal_iou(r, c))
             for i, r in enumerate(ref) for j, c in enumerate(cand)]
    pairs = [p for p in pairs if p[2] >= iou_thr]
    pairs.sort(key=lambda p: p[2], reverse=True)
    used_r, used_c, out = set(), set(), []
    for i, j, iou in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j)
        out.append((i, j, iou))
    return out


def prf(n_matched: int, n_ref: int, n_cand: int) -> Dict[str, float]:
    p = n_matched / n_cand if n_cand else 0.0
    r = n_matched / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def cohens_kappa_grid(a: Sequence[Interval], b: Sequence[Interval],
                      duration_ms: int, step_ms: int = 1000) -> float:
    """Cohen's kappa between two interval sets on a binary time grid."""
    n = max(1, duration_ms // step_ms)

    def mask(iv: Sequence[Interval]) -> List[bool]:
        m = [False] * n
        for s, e in iv:
            for k in range(max(0, s // step_ms), min(n, (e + step_ms - 1) // step_ms)):
                m[k] = True
        return m

    ma, mb = mask(a), mask(b)
    po = sum(1 for x, y in zip(ma, mb) if x == y) / n
    pa1, pb1 = sum(ma) / n, sum(mb) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1.0 else 1.0


def load_annotations(path: Path) -> Tuple[str, Dict[str, List[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("annotator", path.stem)), data.get("matches", {})


def load_detector(study_dir: Path, match_id: str) -> List[dict]:
    p = study_dir / "detector" / f"{match_id}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("fights", [])


def load_payload_meta(study_dir: Path, match_id: str) -> Tuple[int, List[int]]:
    """Returns (duration_ms, kill timestamps)."""
    p = study_dir / "matches" / f"{match_id}.json"
    if not p.exists():
        return 0, []
    d = json.loads(p.read_text(encoding="utf-8"))
    return int(d.get("duration_ms", 0)), [int(k["ts"]) for k in d.get("kills", [])]


def detector_interval(f: dict) -> Interval:
    s = int(f.get("engage_ts", 0))
    e = max(int(f.get("last_kill_ts", s)), s + 1000)
    return (s, e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score annotations vs detector")
    ap.add_argument("--study_dir", type=str, default="outputs/annotation_study")
    ap.add_argument("--annotations", type=str, nargs="+", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=str, default=None,
                    help="report JSON path (default: <study_dir>/score_report.json)")
    args = ap.parse_args()

    study_dir = Path(args.study_dir)
    annotators = [load_annotations(Path(p)) for p in args.annotations]

    report: Dict[str, dict] = {"iou_threshold": args.iou, "vs_detector": {}, "inter_annotator": {}}

    # ---- 1) each annotator vs detector ----
    for name, per_match in annotators:
        tot_ref = tot_ref_kill = tot_cand = 0
        n_match_all = n_match_kill = 0
        ious: List[float] = []
        onset_err: List[float] = []
        killless = 0

        for match_id, alist in per_match.items():
            det = [detector_interval(f) for f in load_detector(study_dir, match_id)]
            _, kill_ts = load_payload_meta(study_dir, match_id)
            human_all = [(int(a["start_ms"]), int(a["end_ms"])) for a in alist]
            # kill-containing subset (the detector's declared scope)
            human_kill = [iv for iv in human_all
                          if any(iv[0] <= k <= iv[1] for k in kill_ts)]
            killless += len(human_all) - len(human_kill)

            m_all = greedy_match(human_all, det, args.iou)
            m_kill = greedy_match(human_kill, det, args.iou)
            tot_ref += len(human_all)
            tot_ref_kill += len(human_kill)
            tot_cand += len(det)
            n_match_all += len(m_all)
            n_match_kill += len(m_kill)
            ious.extend(iou for _, _, iou in m_all)
            onset_err.extend(abs(human_all[i][0] - det[j][0]) / 1000.0
                             for i, j, _ in m_all)

        entry = {
            "n_annotated": tot_ref,
            "n_detected": tot_cand,
            "n_matched": n_match_all,
            **prf(n_match_all, tot_ref, tot_cand),
            "recall_kill_scope": round(n_match_kill / tot_ref_kill, 4) if tot_ref_kill else 0.0,
            "killless_annotated": killless,
            "mean_matched_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "mean_onset_err_sec": round(sum(onset_err) / len(onset_err), 2) if onset_err else 0.0,
        }
        report["vs_detector"][name] = entry

    # ---- 2) pairwise inter-annotator agreement ----
    for (na, ma), (nb, mb) in itertools.combinations(annotators, 2):
        common = sorted(set(ma) & set(mb))
        n_ref = n_cand = n_matched = 0
        kappas: List[float] = []
        for match_id in common:
            ia = [(int(a["start_ms"]), int(a["end_ms"])) for a in ma[match_id]]
            ib = [(int(a["start_ms"]), int(a["end_ms"])) for a in mb[match_id]]
            n_matched += len(greedy_match(ia, ib, args.iou))
            n_ref += len(ia); n_cand += len(ib)
            dur, _ = load_payload_meta(study_dir, match_id)
            if dur > 0:
                kappas.append(cohens_kappa_grid(ia, ib, dur))
        report["inter_annotator"][f"{na}|{nb}"] = {
            "n_common_matches": len(common),
            **prf(n_matched, n_ref, n_cand),
            "mean_kappa_1s_grid": round(sum(kappas) / len(kappas), 4) if kappas else 0.0,
        }

    out_path = Path(args.out) if args.out else study_dir / "score_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- console summary ----
    print(f"\n=== Annotator vs detector (IoU >= {args.iou}) ===")
    for name, e in report["vs_detector"].items():
        print(f"  {name}: P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f} "
              f"| R(kill-scope)={e['recall_kill_scope']:.3f} "
              f"| onset err {e['mean_onset_err_sec']:.1f}s "
              f"| killless annotated {e['killless_annotated']}")
    if report["inter_annotator"]:
        print("=== Inter-annotator ===")
        for pair, e in report["inter_annotator"].items():
            print(f"  {pair}: F1={e['f1']:.3f} kappa={e['mean_kappa_1s_grid']:.3f} "
                  f"({e['n_common_matches']} common matches)")
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
