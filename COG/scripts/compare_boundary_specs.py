"""Compare two fight-boundary pipeline runs side by side (for example a pilot sample vs the full corpus).

    python scripts/compare_boundary_specs.py DIR_A DIR_B --label-a pilot --label-b full

Each directory is an ``--out-dir`` of ``scripts/run_fight_boundary_pipeline.py`` (or a copy
under ``config/fight_boundary``).  ``spec_<scope>.json`` is required; ``drift.json`` and
``details_<scope>.json`` are used when present.  Output is markdown: the boundary table
over pooled + per-patch scopes, the drift verdict of DIR_B, and the pooled temporal and
spatial details of both runs (region crossovers, lane anisotropy, region transitions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
    sys.stdout.reconfigure(encoding="utf-8")


def _load(p: Path) -> Optional[dict]:
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _scopes(d: Path) -> list:
    patches = sorted(p.stem[len("spec_"):] for p in d.glob("spec_*.json") if p.stem != "spec_pooled")
    return ["pooled"] + patches


def _row(s: dict) -> str:
    gci = f"{s['gap_ci_s'][0]:.1f}~{s['gap_ci_s'][1]:.1f}" if s.get("gap_ci_s") else "-"
    dci = f"{s['diameter_ci_u'][0]:,.0f}~{s['diameter_ci_u'][1]:,.0f}" if s.get("diameter_ci_u") else "-"
    pl = f"{s['gap_plateau_s'][0]:g}~{s['gap_plateau_s'][1]:g}" if s.get("gap_plateau_s") else "-"
    depth = f"{s['gap_depth']:.2f}" if s.get("gap_depth") is not None else "-"
    mass = f"{100 * s['diameter_mass_near']:.1f}%" if s.get("diameter_mass_near") is not None else "-"
    return (f"{s['n_matches']:,} | {s['n_intervals']:,} | {s['gap_s']:.1f} | {gci} | {depth} | {pl} | "
            f"{s['diameter_u']:,.0f} | {dci} | {mass} | {s['n_pairs_in_window']:,}")


def _r(x, nd=0):
    if x is None:
        return "-"
    return f"{x:,.{nd}f}"


def _details(d: Path, label: str) -> None:
    det = _load(d / "details_pooled.json")
    if not det:
        return
    T, S = det["temporal"], det["spatial"]
    k = T["kde"]
    print(f"\n### {label}: pooled temporal")
    print(f"modes {k['mode1_s']:.1f} s / {k['mode2_s']:.1f} s, valley {k['valley_s']:.2f} s, depth {k['depth']:.3f}, "
          f"n {k['n']:,}, bandwidth {k['bandwidth']}")
    sweep = {bw: round(v["valley_s"], 2) for bw, v in T["bandwidth_sweep"].items() if v.get("ok")}
    print("bandwidth sweep:", sweep)
    mix = T["mixture"]
    print(f"mixture crossing {mix['crossing_s']:.2f} s (fit n {mix.get('n_fit', mix.get('n')):,})")
    b = T["bootstrap"]["valley_s"]
    print(f"bootstrap valley p2.5 {b['p2.5']:.2f} / p50 {b['p50']:.2f} / p97.5 {b['p97.5']:.2f} s "
          f"(n_boot {T['bootstrap'].get('n_boot')}, plateau {T.get('plateau_s')})")
    print("ARI vs valley:", {g: round(v, 3) for g, v in T["ari_vs_valley"].items()})

    print(f"\n### {label}: pooled spatial (window {S['gap_s']:.1f} s, {S['n_pairs_in_window']:,} pairs)")
    bs = S["bootstrap"] or {}
    print(f"crossover {_r(S['crossover_u'])} u, bootstrap {_r(bs.get('p2.5'))}~{_r(bs.get('p97.5'))}, "
          f"mass within ±10% {100 * (S.get('mass_near_boundary') or 0):.1f}%, "
          f"share ≤2,000 {100 * S['share_le_2000']:.1f}%, ≤4,000 {100 * S['share_le_4000']:.1f}%")
    print("sharing curve (bin centre u, rate, n):", [(int(x), round(y, 3), n) for x, y, n in S["curve"]])
    print("\n| region | crossover (u) | 95% CI | pairs |")
    print("|---|---|---|---|")
    for reg, v in S["by_region"].items():
        bb = v.get("bootstrap") or {}
        print(f"| {reg} | {_r(v['crossover_u'])} | {_r(bb.get('p2.5'))}~{_r(bb.get('p97.5'))} | {v['n']:,} |")
    print("\n| lane | along (u) | n | across (u) | n | ratio |")
    print("|---|---|---|---|---|---|")
    for lane, v in S["lane_anisotropy"].items():
        al, ac = v["along"], v["across"]

        def cx(e):
            if e["crossover_u"] is not None:
                return _r(e["crossover_u"])
            lb = e.get("last_bin")
            return f"> {lb['u']:,.0f} ({lb['shared']:.2f})" if lb else "-"

        ratio = (f"{al['crossover_u'] / ac['crossover_u']:.2f}"
                 if al["crossover_u"] and ac["crossover_u"] else "-")
        print(f"| {lane} | {cx(al)} | {al['n']:,} | {cx(ac)} | {ac['n']:,} | {ratio} |")
    rt = S["region_transition"]
    print("\nregion transition:", {kk: round(vv, 3) for kk, vv in rt.items() if vv is not None})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--no-details", action="store_true", help="print only the spec table and verdict")
    args = ap.parse_args()

    scopes = [s for s in _scopes(args.dir_b) if s in _scopes(args.dir_a)]
    print("| scope | run | matches | intervals | G (s) | G 95% CI | depth | plateau | D (u) | D 95% CI | mass ±10% D | pairs in window |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in scopes:
        for lab, d in ((args.label_a, args.dir_a), (args.label_b, args.dir_b)):
            s = _load(d / f"spec_{name}.json")
            if s:
                print(f"| {name} | {lab} | {_row(s)} |")

    for lab, d in ((args.label_a, args.dir_a), (args.label_b, args.dir_b)):
        drift = _load(d / "drift.json")
        if drift:
            print(f"\n{lab} verdict: **{drift['verdict']}**")
            for p, r in drift["patches"].items():
                print(f"- {p}: G {r['gap_s']:.1f} s inside={r['gap_inside_pooled_plateau']}; "
                      f"D {r['diameter_u']:,.0f} u rel {r['diameter_rel_diff']:+.3f} inside={r['diameter_inside']}; "
                      f"R coverage inside={r.get('radius_inside')}")

    if not args.no_details:
        for lab, d in ((args.label_a, args.dir_a), (args.label_b, args.dir_b)):
            _details(d, lab)


if __name__ == "__main__":
    main()
