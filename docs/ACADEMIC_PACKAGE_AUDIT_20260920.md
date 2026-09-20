# Academic package audit — journal-finish readiness

**Date:** 2026-09-20  
**Branch / tip:** `feature/fight-boundary-pipeline` (`bbde058` journal-freeze Forbidden list)  
**Skills applied:** `academic-pipeline` (stage detect + AI failure modes), `scientific-writing` (evidence binding / reporting route), `research-paper-writing` (5-dim paper review), `academic-paper-reviewer` (5-seat readiness), `academic-paper` (write-stage gaps), `zotero-cli` (library coverage), project rule `manuscript-evidence.mdc`  
**Not run as prose rewrite:** `academic-humanizer`, `grammar-checker`, `style-guide` — deferred until freeze Results/Discussion exist (nothing to humanize without changing claims)

**Verdict:** Evidence package is **ready to write**. Current `docs/tog_manuscript/` is **not** the freeze paper. Stopping V/q performance racing remains **justified**.

### Author adjudication (2026-09-20, post-audit review)

Simulated panel grades (e.g. “Major Revision”) in this file are **work-priority diagnostics**, not forecasts of IEEE ToG decisions and not independent validation. Use concrete claim↔file mismatches; do not treat skill “PASS” labels as substitutes for human verification of sentences.

**TRIPOD+AI:** selective reporting *checklist* for prediction-time availability, splits, discrimination/calibration, external evaluation, and uncertainty — **not** a clinical primary standard for this game study. Prefer venue instructions first.

**CORP:** report as **score decomposition on an evaluation sample**, not as a causal “mechanism.”

**Exploratory / prior TEST exposure / RR1 execution honesty:** required in **Methods/Results draft 1** (not a late P1 appendix).

**Writing package started:** [journal_manuscript_v1/](journal_manuscript_v1/README.md) (outline, claim map, Methods, Results, Discussion). Prior `tog_manuscript/` preserved.

---

## 0. Pipeline stage detection (`academic-pipeline`)

| Check | Finding |
|---|---|
| Materials | Locks + RR0–RRX CLOSED + lit bridge CLOSED + CoG-era ToG tex |
| Manuscript | Partial CoG-extension draft; **no** freeze Results/Discussion; `\pending{abstract}`; `sec_conclusion.tex` missing |
| Entry point | **Stage 2 WRITE** (not Stage 2.5 integrity on a freeze draft — that draft does not exist yet) |
| Mode | Mid-entry: research + verification packs done → draft freeze manuscript |
| Next gate | After first freeze Results/Discussion draft → Stage 2.5 integrity (claim↔artifact + failure modes) |

```
━━━ Academic intake complete ━━━
Evidence: RR0–RRX CLOSED · JOURNAL FREEZE locked
Manuscript: CoG spine (winner AUC / observation ceiling) — NOT freeze thesis
Decision if PDF submitted today: incomplete CoG spine — rewrite onto freeze draft (not a ToG outcome forecast)
Recommended next: WRITE Results+Discussion from JOURNAL_FINISH_LOCK thesis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 1. Evidence package health (`scientific-writing` + locks)

### 1.1 Reporting route (non-scoring)

`select_reporting_guidelines.py`:

| Declared design | Role of bundled selector output |
|---|---|
| `prediction_model` | Selector may list TRIPOD+AI — treat as **optional item bank**, not study-type classification |
| `observational` | STROBE topics only if describing corpus construction |

**Author rule:** Follow IEEE ToG author instructions. Optionally borrow TRIPOD+AI items on prediction-time information, splits, discrimination, calibration, external evaluation, and uncertainty. Do **not** frame this paper as a clinical prediction-model report. Do **not** force CONSORT/CARE.

### 1.2 Claim registry status (conceptual)

| Claim family | Evidence IDs (locks/artifacts) | Status |
|---|---|---|
| Direction predictability beyond \(p,t\) (15.16) | RR12: ΔBrier(q−PT_flex) −0.00373, CI excludes 0; AUC 0.6403 | **Supported** (exploratory post-TEST disclosure required) |
| B40 lift | RR12/RR6a −0.00214; CI upper near 0 | **Supported as exploratory** — do not oversell |
| CORP score decomposition (MAIN) | RR6a: ΔDSC outweighs ΔMCB on that sample | **Supported as decomposition** (not causal mechanism) |
| EXT lift loss | RRX KR/NA1 16.13 ΔBrier > 0; DSC still > PT, MCB dominates | **Supported** |
| Measurement correspondence | RR3/RR5/RR6b | **Supported as correspondence**, not ATT / not \(q\) accuracy |
| Public-telemetry AUC ceiling | — | **Forbidden** (no evidence) |
| Recalibration fixes EXT | — | **Forbidden** (no adapter) |

### 1.3 Doc inconsistency to resolve in writing (not new experiments)

| Doc | Tension |
|---|---|
| `J_RQ1_SCOPE_LOCK` | Still says RQ1 **검증 미완료** |
| `JOURNAL_FINISH_LOCK` + matrix | Says RR packs CLOSED · write Results next |

**Resolution for manuscript:** Treat J-RQ1 as **answered under frozen fit85 + closed RR stack**, with residual limitations explicit. Update or annotate `J_RQ1_SCOPE_LOCK` so it does not contradict the freeze (documentation hygiene, P1).

`Q_RESULT_SCOPE_LOCK` still centers **PT_linear** as primary in one paragraph; freeze primary is **PT_flex**. Keep PT_linear as continuity only.

---

## 2. Manuscript vs freeze (`research-paper-writing` + manuscript audit)

### 2.1 Structural mismatch (P0)

Current title/abstract plan and `sec_prediction` still define:

> predict whether blue **wins the engagement** (`market_event`),

not

> predict \(\mathbf{1}[\Delta\widehat V>0]\) beyond flexed \(p,t\), and report where that lift fails.

Freeze markers in `.tex`: **essentially absent** (no PT_flex / CORP / SVI / fit85 / 16.13 primary Results).  
`\pending` markers remain dense especially in `sec_learners.tex` (~46) — CoG deep-learner debt, **orthogonal** to freeze (do not reopen as q zoo).

Stale numbers that must not appear as freeze headlines:

| Manuscript | Freeze lock |
|---|---|
| \(V\) AUC 0.846 / Brier 0.160 | fit85 ~0.854 / 0.1552 |
| Direction AUC 0.583 (stacking / \|ΔV\|-weighted) | frozen \(q\) AUC 0.6403 |
| Winner AUC ~0.67 / CoG 0.675 as primary RQ | Secondary lineage only |

### 2.2 Five-dimension self-review (package + current tex)

| Dimension | Package | Current manuscript | Gate |
|---|---|---|---|
| **1 Contribution** | Verification + measurement upgrade of CoG purpose — **pass** if framed that way | Still sells winner-ceiling / tool stacking null — **needs revision** | Rewrite Intro contributions |
| **2 Writing clarity** | Locks clear | Dual story risk; abstract pending | Needs freeze spine outline |
| **3 Experimental strength** | Limited but honest lift; EXT negative — **pass for verification venue** | Wrong primary estimand | Do not chase AUC |
| **4 Evaluation completeness** | RR1–RRX closed for declared asks | Missing CORP/EXT/B40 in tex | Write Results from matrix |
| **5 Method soundness** | Frozen \(V\) / dual-stage / exploratory disclosure — sound | Learners zoo pendings tempt frame-lock | Keep freeze Forbidden |

---

## 3. Simulated panel (`academic-paper-reviewer`)

**Advisory only — not an IEEE ToG decision forecast.** Use for rewrite priorities; verify every sentence against artifacts.

| Seat | Actionable themes |
|---|---|
| Journal-Fit | Show CoG→ToG increment explicitly; the label “verification” does not create fit |
| Methodology | Disclose exploratory / prior TEST exposure in draft 1; primary = all-T vs PT_flex; keep B40 visible |
| Domain | RR5/RR6b = correspondence; Maymin as prior not “first” |
| Perspective | Absolute lift is small — honest decision-context framing |
| Devil’s Advocate | Estimated-label dependence + post-TEST cherry risk must stay visible |

**“Stop racing and write?”** → **Yes.** Evidence answers the RQ; next AUC hunt is a new question (EXT adapter → thesis/v2).

---

## 4. AI research failure-mode checklist (`academic-pipeline` Stage 2.5 preview)

Applied to the **evidence package** (pre-draft). Blocking semantics preview for the first freeze draft.

| Mode | Verdict | Notes |
|---|---|---|
| 1 Implementation bug | **MOSTLY CLEARED** for named close-outs | CORP score-gap fixed (`948b36a`); RRX common-valid + feature-order (`3654d4e`); RR5b game-end vs censored clarified. Residual: keep RR1 execution addendum deviations visible. |
| 2 Hallucinated citation | **N/A until freeze prose cites** | Local bibs hold Maymin / Gneiting–Raftery / Dimitriadis CORP. **Zotero library search returned 0** for those queries — sync/import before citation-integrity pass. |
| 3 Hallucinated results | **LOW risk if numbers only from RR docs** | Manuscript still holds superseded \(V\)/\(q\) figures — high risk if copied into freeze Results without re-reading artifacts. |
| 4 Shortcut reliance | **FLAG** | \(q\) may partly rediscover \(p_{\mathrm{pre}}\) structure; PT_flex is the right stress test. EXT failure argues against “solved generalization.” |
| 5 Bug-as-insight | **WATCH** | “DSC high but MCB kills EXT” is diagnostic, not a novel “insight” that recalibration will fix. |
| 6 Methodology fabrication | **LOW if Methods cite manifests** | Risk: describing Q_CAL sigmoid as done when identity was selected; describing PT_linear as the flexed primary. |
| 7 Frame-lock | **ACTIVE RISK IN TEX** | CoG winner-ceiling frame still dominates `tog_manuscript`. Freeze deliberately unlocks by rewriting the spine — do not let learners `\pending` re-lock the paper into model racing. |

**Integrity preview:** would **block Accept** of current PDF; would **not** block starting Stage 2 WRITE of freeze Results.

---

## 5. Literature / Zotero (`zotero-cli` + local packs)

| Source | Status |
|---|---|
| `docs/literature/.../econometrics_for_lol.bib` | Has GneitingRaftery2007, DimitriadisGneitingJordan2021, Maymin2021 |
| `docs/references/definition_refs.bib` | Has `maymin2021smart` (VERIFIED lineage for CoG-era tex) |
| Live Zotero (`zotero-cli search`) | **0 hits** for Maymin / CORP — library not populated or not the project corpus |

**Action before Stage 2.5 citation check:** import lit-pack + definition_refs into Zotero (or point integrity checks at local `.bib` only). Do not invent DOIs.

---

## 6. Skills deferred (correctly)

| Skill | Why deferred |
|---|---|
| `academic-humanizer` | No freeze Results prose yet; humanizing CoG spine would polish the wrong paper |
| `grammar-checker` / `style-guide` | Same; run on first freeze English draft |
| `deep-research` full | Lit application already locked; only gap-fill if Related Work needs a missing primary source |
| `academic-paper` full write | Ready to dispatch **outline → Results/Discussion** on user confirm — not auto-started here |
| Clinical checklists | Explicitly out of scope per `manuscript-evidence.mdc` |

---

## 7. Revision roadmap (writing only — no new experiments)

### P0

1. Rewrite RQs / contributions / title stress toward **JOURNAL_FINISH_LOCK** thesis paragraph.  
2. New Results: RR12 (PT_flex primary) + RR6a CORP + RRX EXT; quarantine 0.846/0.160/0.583.  
3. Retarget prediction estimand to SVI; keep definition/label as measurement methods.  
4. Discussion: beyond \(p,t\); B40 honesty; EXT MCB-dominated failure; Forbidden denials.  
5. Abstract + `sec_conclusion.tex`.

### P1

6. Related Work: Maymin + object-separation econometrics (not “finance on LoL”).  
7. Disclose EXPLORATORY / prior TEST exposure + RR1 addendum.  
8. Align `J_RQ1_SCOPE_LOCK` / `Q_RESULT_SCOPE_LOCK` wording with freeze.  
9. Demote `sec_learners` pendings for this version.

### P2

10. Sync Zotero from local bibs.  
11. Resolve non-blocking `\pending` (authors, availability).  
12. After draft: scientific-writing claim audit + humanizer + grammar pass.

---

## 8. Recommended next user decision

Per `academic-pipeline` checkpoint (FULL):

1. **Proceed to WRITE** — outline + Results/Discussion from freeze thesis (recommended).  
2. **Doc hygiene only** — reconcile J_RQ1 / Q_RESULT scope locks with JOURNAL_FINISH first.  
3. **Pause** — keep modeling freeze; no manuscript work yet.

Do **not** reopen V/q performance search as a condition for (1).
