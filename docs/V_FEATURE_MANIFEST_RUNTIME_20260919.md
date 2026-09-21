# Runtime StateV2 feature manifest (Step 1)

Generated: 2026-09-19T23:17:26+09:00
Source: `C:/Users/todtj/PycharmProjects/LOL_teamfight/outputs/v_redesign_feature_manifest_20260919.json` (local outputs; may be gitignored)

- Raw width: **362**
- Expanded (drop `snapshot_age_s`): **361** (numeric 351 + categorical 10)
- Groups: `{'clock': 2, 'quality': 1, 'player_snapshot': 90, 'champion': 10, 'player_event': 70, 'team_event': 94, 'team_time_interaction': 94}`
- `state_value_v2.py` hash16: `418ff787b5cddd2a`

## vs reference reconstruction

- Reference status: `PROPOSAL_AND_RECONSTRUCTED_REFERENCE_NOT_A_FITTED_RUNTIME_MANIFEST`
- Order exact match: **True**
- Only in runtime: 0 (head: [])
- Only in reference: 0 (head: [])

## Status

**Step 1 PASSED:** runtime names/order match collaborator reference reconstruction exactly (361 = 351 numeric + 10 categorical; 94 team×time interactions present for Core267 ablation).

## Next

- Typed adapters (CAT-1) and history builder (HIST-1 / ENG-1) per [V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md](V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md).
- Phase A: Expanded361 / Core267 current-frame Logistic · LGBM · MLP.
