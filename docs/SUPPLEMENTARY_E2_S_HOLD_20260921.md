# Supplementary E2 — S_hold mechanical probe (§5.2)

**generated:** 2026-09-21T08:18:06.332987+00:00  
**task:** .ai/tasks/T022.md  
**CACHE_MAIN:** `D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13` (npz=210000)  
**bundle_sha16:** `ac459cc4397630a9`  

## Construction

Events/frames with timestamp ≤ `q_pre`; query = `endpoint_h90`; snapshot = last held frame. Frozen fit85 V. Not a no-fight potential outcome.

## Subsample (`subsample`)

status=`OK`; n_ok=2709; miss=0; fail=0; arith_identity=1.0; wall_s=50.5.

| Quantity | mean | mean_abs | p_pos |
|---|---:|---:|---:|
| delta_V | -0.00155 | 0.11334 | 0.497 |
| d_clock | 0.00106 | 0.01268 | 0.576 |
| d_update | -0.00261 | 0.11304 | 0.498 |

Y vs 1[d_update>0] flip=0.0354; opposite-sign clock/update=0.514; mean hold snapshot_age_s=112.2.
Rebuild vs stored X: pre p99 maxabs=0.000e+00, post p99=0.000e+00.

Arithmetic decomposition under chosen update order only. S_hold is not a no-fight potential outcome; held frames may be OOD.

## Full TEST T (`full`)

status=`OK`; n_ok=32981; miss=0; fail=0; arith_identity=1.0; wall_s=610.9.

| Quantity | mean | mean_abs | p_pos |
|---|---:|---:|---:|
| delta_V | -0.00175 | 0.11537 | 0.496 |
| d_clock | 0.00084 | 0.01318 | 0.554 |
| d_update | -0.00258 | 0.11500 | 0.494 |

Y vs 1[d_update>0] flip=0.0360; opposite-sign clock/update=0.507; mean hold snapshot_age_s=111.8.
Rebuild vs stored X: pre p99 maxabs=0.000e+00, post p99=0.000e+00.

Arithmetic decomposition under chosen update order only. S_hold is not a no-fight potential outcome; held frames may be OOD.

