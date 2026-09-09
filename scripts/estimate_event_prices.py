"""Recover the team gold the game pays per event from the corpus frames (analysis/event_prices.py).

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/estimate_event_prices.py \\
        --n-matches-per-patch 8000 --seed 7 \\
        --output D:/LOL_Project/fusion_2615/features/event_prices.json \\
        --table config/game_rules/event_prices.json

Per patch and pooled: OLS of the per-team, per-frame gold change on that team's event counts
inside the frame interval (kill bounty paid, kills, assists, plates, towers by tier,
inhibitors, first-tower bonus, dragons, elder, baron, herald, voidgrubs, atakhan, ward kills)
plus lane and jungle CS changes.  The coefficient of an event is the average total gold the
team received for it.  Writes the fits (coefficients, standard errors, counts, R^2) and a
rounded price table used by the ``market_event`` label.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Wiki values (wiki.leagueoflegends.com, read 2026-09-09) for the cross-check column.  The
# pages state current (V26.01) values and, in patch histories, the pre-V26.01 values that
# were in force at 15.14-15.16; "per member" amounts are multiplied by 5 for a team total.
WIKI_TEAM_TOTAL = {
    "plates": "120 local (current page; pre-26.01 value not listed)",
    "tower_outer_turret": "50 global x5 = 250 (+300 first-turret bonus; current page)",
    "tower_inner_turret": "25 global x5 = 125 (current page)",
    "tower_base_turret": "25 global x5 = 125 (current page)",
    "tower_nexus_turret": "50 global x5 = 250 (current page)",
    "inhibitor": "50 to the last hitter (current page)",
    "dragon": "25 kill gold pre-26.01 (per-member status unclear)",
    "elder_dragon": "250 global pre-26.01 (per-member status unclear)",
    "baron_nashor": "25 kill + 300 global pre-26.01 (per-member status unclear)",
    "rifthearld": "",
    "riftherald": "100 local (V13.20+)",
    "horde": "20 local + 50 global pre-26.01 (per grub; per-member status unclear)",
    "atakhan": "50 / +25 (bounty box; form unspecified)",
    "ward_kills": "",
    "first_tower": "300 bonus (current page)",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches-per-patch", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--table", type=Path, default=None, help="write the rounded price table here (JSON)")
    args = ap.parse_args(argv)

    from data.cache_io import load_match_cache
    from analysis.kill_pairs import list_cache_match_ids, group_by_patch, sample_ids
    from analysis.event_prices import REGRESSORS, fit_prices, match_rows, price_table

    mids = list_cache_match_ids()
    by_patch = group_by_patch(mids, progress=lambda s: print(s, flush=True))
    fits = {}
    X_all, Y_all = [], []
    t0 = time.time()
    for patch in sorted(by_patch):
        ids = sample_ids(by_patch[patch], args.n_matches_per_patch, args.seed)
        Xs, Ys = [], []
        for i, mid in enumerate(ids, 1):
            pack = load_match_cache(mid)
            if not pack:
                continue
            X, Y = match_rows(pack)
            if X is not None and len(Y):
                Xs.append(X); Ys.append(Y)
            if i % 2000 == 0:
                print(f"  [{patch}] {i}/{len(ids)} matches, {sum(len(y) for y in Ys):,} intervals, {time.time() - t0:.0f}s", flush=True)
        X = np.concatenate(Xs); Y = np.concatenate(Ys)
        fits[patch] = fit_prices(X, Y)
        fits[patch]["n_matches"] = len(ids)
        X_all.append(X); Y_all.append(Y)
        print(f"[{patch}] n_intervals {len(Y):,} R2 {fits[patch]['r2']:.3f}", flush=True)
    X = np.concatenate(X_all); Y = np.concatenate(Y_all)
    fits["pooled"] = fit_prices(X, Y)
    fits["pooled"]["n_matches"] = int(sum(f["n_matches"] for p, f in fits.items() if p != "pooled"))

    print("\n| event | " + " | ".join(sorted(p for p in fits if p != "pooled")) + " | pooled (SE) | count | wiki |")
    print("|---|" + "---|" * (len(fits) + 2))
    for name in REGRESSORS:
        cells = [f"{fits[p]['coef'][name]:.1f}" for p in sorted(fits) if p != "pooled"]
        f = fits["pooled"]
        print(f"| {name} | " + " | ".join(cells) + f" | {f['coef'][name]:.1f} ({f['se'][name]:.1f}) | {f['count'][name]:,.0f} | {WIKI_TEAM_TOTAL.get(name, '')} |")
    print(f"| intercept (passive gold / min) | " + " | ".join(f"{fits[p]['intercept']:.1f}" for p in sorted(fits) if p != "pooled")
          + f" | {fits['pooled']['intercept']:.1f} ({fits['pooled']['intercept_se']:.1f}) | | ~20.4 g / 10 s x 5 = 612 |")

    out = {"regressors": REGRESSORS, "fits": fits, "wiki_cross_check": WIKI_TEAM_TOTAL,
           "seed": args.seed, "n_matches_per_patch": args.n_matches_per_patch, "elapsed_s": round(time.time() - t0)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", args.output)
    if args.table:
        table = price_table(fits["pooled"])
        per_patch = {p: price_table(f) for p, f in fits.items() if p != "pooled"}
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(json.dumps({
            "unit": "team gold per event (local + global, killer + assisters), regression on frame gold",
            "patches": sorted(p for p in fits if p != "pooled"),
            "pooled": table, "per_patch": per_patch,
            "source": str(args.output), "note": "kill gold is read from the event (bounty + shutdownBounty); "
                                                 "'assists' is gold per assist beyond the bounty",
        }, indent=1), encoding="utf-8")
        print("wrote", args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
