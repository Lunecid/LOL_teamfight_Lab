# Venue checklist — IEEE Transactions on Games (ToG)

*Prepared 2026-09-21 (Claude session, T013). Method: venue-templates skill, verification-first. Every rule below carries its source and the date it was read; bundled scaffolds are not treated as official templates.*

## Compliance note

```text
Target:            IEEE Transactions on Games, full (regular) paper, initial submission
Official sources:  https://transactions.games/submit/submission-guidelines  (read 2026-09-21 via WebFetch; page undated)
                   https://cis.ieee.org/publications/t-games/tciaig-information-for-authors  (HTTP 418 to the fetcher on 2026-09-21;
                   read with a browser on 2026-09-14 per docs/DATA_AVAILABILITY.md §6)
                   https://ieeeauthorcenter.ieee.org/  (template hub named by the guidelines page; not fetched today)
Submission system: Manuscript Central, https://mc.manuscriptcentral.com/tg-ieee
Main-text limit:   10 pages, two-column IEEE format, INCLUDING references and (optional) biographies; supplementary material excluded
Over-length:       USD 200 per page beyond 10 (full paper)
Other types:       letter 4 pages; short paper 6 pages; survey 15 pages; immersive article 6–14 pages
Abstract:          150–200 words, one paragraph, no abbreviations, no references
Keywords:          2–5 required
Review model:      double-anonymous for manuscripts submitted on/after 2025-01-01 ("fully anonymized": names, affiliations, acknowledgments removed)
Figures:           line drawings and photos in black and white unless colour is specifically requested; .tiff/.eps/.ps; TeX/LaTeX or Word
Statements:        the guidelines page does not mandate data-availability, ethics or code statements (docs/DATA_AVAILABILITY.md §6 reached the same reading on 2026-09-14)
```

The repo's own earlier reading (`docs/DATA_AVAILABILITY.md` §6, 2026-09-14) agrees on the double-anonymous rule and on supplementary material being outside the page limit, and records that one CIS page still says "single-blind"; the contradiction is to be raised with the editorial office. No numeric page limit had been recorded in the repo before today; the 10-page figure above is from the guidelines page read on 2026-09-21 and should be re-checked on the submission day.

## What this means for the current drafts

| Rule | Current state (commit `42bb2c6`) | Action |
|---|---|---|
| 10 pages incl. references | Drafts 01–04 total ≈ 9 900 words of Markdown (Methods 5 236, Results 2 169, Discussion 872, Outline 1 598); the earlier CoG-extension `docs/tog_manuscript/main.tex` built to 41 pages | The journal paper needs a hard budget. Proposal in `00_WRITING_DOSSIER.md`: Intro 700 w, Related Work 600 w, Methods 2 300 w, Results 2 000 w (incl. 4 tables + 2 figures), Discussion 800 w, Conclusion/Limitations 400 w, references ≈ 40 entries ≈ 0.8 page. Everything else → supplementary material (RR3/RR4/RR5/RR6b detail tables, per-role census, lineage constants table). |
| Abstract 150–200 words, no abbreviations | `05_ABSTRACT_DRAFT.md` variant A is 196 words and spells out every term (no "SVI", "PT_flex", "B40") | Keep abbreviations out of the abstract; define them in the Introduction. |
| 2–5 keywords | `05_ABSTRACT_DRAFT.md` proposes 5 | — |
| Double-anonymous | Drafts contain no author names; Methods cites repo paths and commit hashes (`21391b2`, `69130a4`) and lineage docs by file name | Before submission: replace repo paths/commit hashes with "supplementary code archive §x" in the manuscript text (keep them in the supplement); do not cite the rejected CoG submission number or its reviews; cite the accepted 4-page CoG paper in third person only if its authorship does not de-anonymize (author decision; see `docs/DATA_AVAILABILITY.md` §6 item 3). |
| Black-and-white figures | `figures/fig1_pipeline.svg`, `fig2_delta_brier_bins_TS.svg`, `fig3_cohort_roles.svg` use black/grey plus marker shape (circle vs square), so identity is never colour-alone | Export to EPS/TIFF at camera-ready (T014); keep shape encoding. |
| Template | `docs/tog_manuscript/main.tex` already uses `\documentclass[journal]{IEEEtran}` with `IEEEtran` bibliography style and the `\pending{}` macro | Reuse that skeleton for the journal paper: new `main_journal.tex` under `docs/journal_manuscript_v1/latex/` (T014), section files ported from 02–04, bib = `docs/references/definition_refs.bib` + `docs/literature/LOL_ECONOMETRICS_LITERATURE_20260920/econometrics_for_lol.bib` (merge; de-duplicate `Maymin2021` vs `maymin2021smart`). |
| Supplementary material | Not yet assembled | Supplement S1 = detector constants table (Methods §1), S2 = RR verification tables (RR3/RR4/RR5/RR6b), S3 = per-role census and the two-evaluator-path table, S4 = S-cohort sensitivity (Table 2b), S5 = anonymised code archive (per DATA_AVAILABILITY §7). |
| Extending previously published work | The guidelines ask authors to state the relation to earlier work in the text while keeping anonymity (DATA_AVAILABILITY §6) | One sentence in the Introduction: "A four-page conference version [ref] predicted the gold engagement winner; this article changes the target and adds the verification stack." Wording depends on the anonymity decision above. |

## LaTeX porting plan (T014, Cursor; no new numbers)

1. `docs/journal_manuscript_v1/latex/main_journal.tex`: copy the preamble of `docs/tog_manuscript/main.tex` (IEEEtran journal, cite, amsmath, booktabs, hyperref, cleveref, `\pending`), new title, anonymised author block, `\IEEEkeywords` from `05_ABSTRACT_DRAFT.md`.
2. Section files: `sec_intro.tex` (from `06_INTRODUCTION_BRIEF.md` once the author confirms), `sec_related.tex` (from `07_RELATED_WORK_BRIEF.md`), `sec_methods.tex` (from `02_METHODS_CORE.md` §1–§6; §7–§9 to supplement), `sec_results.tex` (from `03_RESULTS.md`), `sec_discussion.tex` (from `04_DISCUSSION.md`), `sec_conclusion.tex` (from `08_CONCLUSION_AVAILABILITY_BRIEF.md`), `sec_availability.tex` (adapt `docs/tog_manuscript/sec_availability.tex`).
3. Figures: convert the three SVGs to EPS/PDF with `rsvg-convert` or Inkscape; `\includegraphics` at column width (fig1, fig3 double-column; fig2 double-column).
4. Build with `latexmk -pdf`; run `python scripts/validate_format.py --file main_journal.pdf --max-pages 10 --content-pages 10 --source-url https://transactions.games/submit/submission-guidelines --check page-count,fonts` from the venue-templates skill directory (page count and embedded fonts only; margins and excluded sections are checked by eye).
5. `grep -n "\\\\pending" latex/*.tex` must list only author items; none may ship.

## Final compliance checklist (to tick on submission day)

- [ ] Exact venue, paper type (full), stage (initial) confirmed; guidelines page re-read and dated
- [ ] 10-page limit incl. references verified on the compiled PDF
- [ ] Abstract 150–200 words, one paragraph, no abbreviations; 2–5 keywords
- [ ] Manuscript fully anonymised: no names, affiliations, acknowledgments, repo URLs, commit hashes, or references to the rejected conference submission; PDF metadata cleared
- [ ] Figures readable in black and white; captions state n, matches, weights, and "no interval" where applicable
- [ ] Supplementary archive prepared (code anonymised via `scripts/prepare_anonymous_release.py` as described in `docs/DATA_AVAILABILITY.md` §7)
- [ ] Data terms statement present (Riot API terms; derived data only), see `08_CONCLUSION_AVAILABILITY_BRIEF.md`
- [ ] AI-assistance disclosure decided by the author (ToG page silent; IEEE policy to be checked on submission day)
- [ ] Every `\pending{}` resolved
