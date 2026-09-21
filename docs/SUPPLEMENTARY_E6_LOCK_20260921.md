# Supplementary E6 — Holdout lock (pre-score)

**locked_at:** 2026-09-21T12:16:40.632455+00:00  
**rules_sha256:** `04cfa692ec8bd2ff1365a464e6553af8248252fef1737a00dbbec09305126931`  
**status:** `LOCKED_AWAITING_KEY`  
**API probe:** `FAIL`  

## Blocked

Riot API probe failed (see probe.stderr_tail). Refresh RIOT_API_KEY then re-run.

Next: refresh key → `python scripts/collect_current_season.py --key-file C:\Users\todtj\.secrets\riot.env --output-root D:\LOL_Project\data\raw\e6_holdout_20260921 --platform kr --min-api-patch 16.16 --max-complete-matches 500 --once`

## Eligibility

- Collect `kr` Master+ matches with `api_patch >= 16.16`.
- Exclude already-exposed patches: `15.14, 15.15, 15.16, 16.13, 16.14, 16.15`.
- Target n=500 after sha256(match_id|20260921) ranking.

## Scoring (only after collection lock finalize)

Frozen V/q/PT score-only; no refit; match-cluster ΔBrier.

## Not confirmatory if

- Re-using MAIN/EXT remainder after seeing those patches' results.
- Choosing matches after peeking at scores.

Output root: `D:\LOL_Project\data\raw\e6_holdout_20260921`
