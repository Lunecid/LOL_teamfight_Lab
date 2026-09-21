# RR4 — small-|ΔV| sensitivity (binary SVI retained)

Generated: 2026-09-20T15:55:43+09:00

Primary target stays \(Y=1[\Delta V>0]\) because that is the research question (direction), not because the triad “proves” direction is predictable. λ-grid uses DEV quiet \(s_Q(z)\) from RR3 fallback; absolute cutoffs are auxiliary only.

- TEST n=32981; L>120s rows=3031 (9.2%) — outside RR3 equal-L match support; \(s_Q\) there is a coarser fallback reference.

## \(s_Q\) fallback composition (all TEST rows)

| Level used | n | share |
|---|---:|---:|
| `p_t_L` | 4562 | 13.8% |
| `p_t` | 7018 | 21.3% |
| `p` | 21401 | 64.9% |

Most rows use **p-only** quiet scale, not full \(p,t,L\). Do not describe \(s_Q\) as if every fight had dense same-length quiet support.

## Direction / mean / scale triad by \(p_{\mathrm{pre}}\)

Role: show the three objects differ. Direction predictability is evidenced by RR1/RR2 holdout ΔBrier, not by large \(E[|\Delta V|]\) alone.

| p bin | n | P(ΔV>0) | E[ΔV] | E[|ΔV|] |
|---|---:|---:|---:|---:|
| [0.0,0.1] | 5400 | 0.3703 | 0.0117 | 0.0380 |
| [0.1,0.2] | 3004 | 0.3903 | 0.0187 | 0.1111 |
| [0.2,0.3] | 2621 | 0.4432 | 0.0209 | 0.1533 |
| [0.3,0.4] | 2598 | 0.4544 | 0.0084 | 0.1716 |
| [0.4,0.5] | 2679 | 0.4913 | 0.0053 | 0.1829 |
| [0.5,0.6] | 2744 | 0.5145 | -0.0019 | 0.1839 |
| [0.6,0.7] | 2589 | 0.5268 | -0.0147 | 0.1771 |
| [0.7,0.8] | 2733 | 0.5492 | -0.0209 | 0.1516 |
| [0.8,0.9] | 2918 | 0.5773 | -0.0276 | 0.1185 |
| [0.9,1.0] | 5695 | 0.6257 | -0.0154 | 0.0416 |

## λ · s_Q sensitivity (post-hoc) — all-T

| λ | n | cov | p_pos | E[|ΔV|] | ΔBrier all-T | CI95 |
|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 32981 | 1.000 | 0.4977 | 0.1123 | -0.0037 | [-0.0046, -0.0028] |
| 0.25 | 30876 | 0.936 | 0.4960 | 0.1199 | -0.0042 | [-0.0051, -0.0032] |
| 0.5 | 29210 | 0.886 | 0.4947 | 0.1265 | -0.0041 | [-0.0050, -0.0031] |
| 1.0 | 26297 | 0.797 | 0.4929 | 0.1394 | -0.0042 | [-0.0051, -0.0032] |

## λ · s_Q — B40 within each λ-slice (exploratory; not assumed equal to all-T)

| λ | B40 n | ΔBrier B40 | CI95 |
|---:|---:|---:|---|
| 0.0 | 5423 | -0.0021 | [-0.0042, -0.0001] |
| 0.25 | 5283 | -0.0023 | [-0.0044, -0.0003] |
| 0.5 | 5132 | -0.0028 | [-0.0049, -0.0006] |
| 1.0 | 4861 | -0.0029 | [-0.0052, -0.0007] |

## Absolute |ΔV| cutoffs (auxiliary, different definition)

| cut | n | cov | ΔBrier(q−PT_flex) | CI95 |
|---:|---:|---:|---:|---|
| 0.0 | 32981 | 1.000 | -0.0037 | [-0.0047, -0.0029] |
| 0.005 | 30072 | 0.912 | -0.0039 | [-0.0048, -0.0030] |
| 0.01 | 28072 | 0.851 | -0.0039 | [-0.0049, -0.0030] |
| 0.02 | 25354 | 0.769 | -0.0038 | [-0.0047, -0.0028] |

## Reading

- Under tested small-change exclusion rules, **all-T** q−PT_flex improvement **does not disappear**.
- That is **not** “small ΔV labels are fine” and **not** a deployable large-move selector.
- Do **not** select λ by TEST performance; do **not** infer B40 from all-T.
- Artifacts: `outputs/review_response_rr4_margin_20260920/`
