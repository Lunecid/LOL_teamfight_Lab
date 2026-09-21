# Migration table — prior ToG draft vs journal freeze

**Prior draft:** `docs/tog_manuscript/` (CoG extension; largely frozen prose as of 2026-09-14)  
**Target:** journal verification paper under `JOURNAL_FINISH_LOCK_20260920.md`

## One-line difference

| | Prior draft | Freeze paper |
|---|---|---|
| **Primary target** | \(y=\) engagement winner (`market_event`) | \(Y_{\mathrm{SVI}}=\mathbf{1}[\Delta\widehat V>0]\) |
| **Primary contrast** | Learners / scoreboard / observation ceiling on winner AUC | Frozen \(q\) vs flexed \(p,t\) (**PT_flex**), then transfer |
| **Role of \(V\)** | Stakes / stacking null (older logistic metrics) | Frozen match WP evaluator that **defines** ΔV/SVI |
| **Headline number** | Winner AUC ≈0.67 / CoG 0.675 | All-T ΔBrier(\(q\)−PT_flex) ≈ −0.00373; EXT lift lost |

## Section-by-section

| Prior section | Reuse? | Action |
|---|---|---|
| Definition (engagement G/D, gates, kill-less) | **Yes, Methods** | Keep as engagement construction; not the primary prediction \(y\) |
| Background (LoL / Match-V5) | **Yes** | Light edit only |
| Label (`market_event`, gold) | **Partial** | Demote to secondary correspondence / CoG continuity — **not** freeze \(Y\) |
| Prediction (winner task, leak audit, G×D) | **Archive / demote** | Do not reuse as Results spine; leak/definition sensitivity may stay as Methods appendix lineage |
| Learners (deep tabular zoo, many `\pending`) | **Do not complete for this version** | Freeze forbids q zoo reopen; leave as deferred CoG thread |
| Limit (old logistic \(V\), stacking AUC 0.583) | **Do not copy numbers** | Replace with fit85 \(V\to W\) + freeze \(q\) Results |
| Intro contributions (ceiling / stacking null) | **Rewrite** | Replace with freeze thesis paragraph |
| Related Work | **Partial** | Keep Maymin / encounter priors; add direction-vs-mean / proper-score object language (not “finance on LoL”) |
| Limitations / future causal ATT | **Rewrite** | EXT adapter deferred; quiet≠ATT; Forbidden list |

## Numbers that must not migrate as freeze headlines

| Prior figure | Why blocked |
|---|---|
| \(V\) AUC 0.846 / Brier 0.160 | Superseded by fit85 / RR6a (~0.854 / 0.1552) |
| Direction AUC 0.583 | Not frozen `logit_state` \(q\) on SVI |
| Winner AUC 0.675 / ~0.67 as “the” predictability bound | Different estimand |

## What this migration is not

- Not a license to fill `sec_learners` pendings.
- Not a numeric swap (replace 0.67 with 0.64) without changing the task statement.
- Not a claim that the prior draft was wrong for CoG — it answers a **different** question.
