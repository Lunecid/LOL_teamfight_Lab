# Supplementary E4 — Reference-arm class census

**generated:** 2026-09-22T09:36:21.221197+00:00  
**cache:** `D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13`  
**sample_mode:** full_cache_meta (n_matches_request=0)  
**wall_s:** 4986.3  

Census of detector scale class only (reference arm / empty overrides on v3.3). Not the exchange-value labeled subset. AUC/cut sensitivity not recomputed. T=min(cluster_blue,cluster_red)>=4; S=2..3; P=0..1 (via n_min). May differ slightly from stored corpus row counts (566,104 valid; T 113,901).

## Scale

| metric | value |
|---|---:|
| n_matches_listed | 210000 |
| n_matches_loaded | 210000 |
| n_miss | 0 |
| n_detect_err | 0 |
| n_engagements | 566452 |

## Cohort (scale class)

| class | count | share |
|---|---:|---:|
| T | 113901 | 20.11% |
| S | 349248 | 61.66% |
| P | 103303 | 18.24% |

## n_min histogram

| n_min | count | share |
|---|---:|---:|
| 1 | 103244 | 18.23% |
| 2 | 235034 | 41.49% |
| 3 | 114214 | 20.16% |
| 4 | 74346 | 13.12% |
| 5 | 39555 | 6.98% |
| other | 59 | 0.01% |

## Diagonal (blue==red among 1..5 joints recorded)

| cell | count | share of engagements |
|---|---:|---:|
| 2v2 | 88949 | 15.70% |
| 4v4 | 34650 | 6.12% |
| 5v5 | 39555 | 6.98% |

## Params (reference arm)

```
{
  "G": 13700,
  "D": 4264.0,
  "R": 1600.0,
  "B": 15000,
  "I": 3000.0,
  "MR": 2000.0,
  "MD": 60000,
  "SHOPEX": false
}
```
