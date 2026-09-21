# Conclusion, Limitations, Availability and Statements — brief and drafts (T013, 2026-09-21)

*Status: proposal for author confirmation. Method: research-paper-writing conclusion guide (restate problem → strongest evidence → insight → limitation → next step) and the academic-paper skill's mandatory inclusions (data availability, ethics, contributions, conflicts, funding, AI-use disclosure). The ToG guidelines page (read 2026-09-21) does not mandate these statements; IEEE-wide policy should be checked on submission day. Numbers are copied from `03_RESULTS.md`; availability facts from `docs/DATA_AVAILABILITY.md` and `docs/tog_manuscript/sec_availability.tex`.*

## Conclusion — draft (≈ 230 words)

This article asked how much of a fight's consequence is visible in public state before the fight starts, and answered it with a target linked to the match: the direction of change in estimated win probability across the engagement interval under a frozen evaluator. Within teamfights and within skirmishes, a regularized logistic model on pre-fight state lowers the match-weighted Brier score relative to a spline baseline in win probability and time, by about 0.0037 and 0.0034 on the held-out patch, with match-cluster bootstrap intervals that exclude zero. The gain is small in absolute terms: for reference, a constant prediction at the cohort base rate scores a Brier of about 0.25 on this near-even outcome. That reference is the marginal (base-rate) uncertainty of the label under this evaluator, not a bound on how predictable the outcome is; it does not license any claim about an irreducible limit of prediction from public telemetry. The finding is that pre-fight public state carries a measurable but modest amount of information about where a fight will move the match, once the two things everyone already knows, the current win probability and the clock, are taken out.

The verification stack bounds the claim. The evaluator's own quality is reported separately; fight intervals move the estimate more than matched quiet intervals; the label agrees with material and objective outcomes where those are decided; and the lift survives filters on small changes. The lift does not survive score-only transfer to a later patch and another region for teamfights, where the discrimination component still favours the model but miscalibration dominates.

Cohort-specific models are the right unit: pooling teamfights and skirmishes helps neither. Extending the design with a validated external adapter, and testing whether the label's direction can be read from finer participation counts, are the next steps; both are outside this version and are listed as preregistration-ready plans in the working notes.

## Limitations — consolidated (from Discussion §6, Methods §0, contract §6)

1. **Exploratory status.** The flexed-baseline, CORP, external and scale-split panels were produced after earlier exposure of the 15.16 test patch; intervals are match-bootstrap contrasts under a fixed evaluator and selection, not preregistered tests and not model-fitting uncertainty.
2. **Model-defined label.** The direction label is the sign of a frozen evaluator's change; absolute fight value is not observed, and every predictability claim is conditional on that evaluator and on the endpoint rule.
3. **Cohort scope.** Picks (a side with at most one participant, about 18 % of engagements) are excluded a priori; no result exists for them under this evaluator. The two cohorts differ in size, positive rate and balanced-state share, so their absolute scores are not comparable and are not compared.
4. **Calibrator variant.** For skirmishes the two-stage selection rule would have chosen a sigmoid for every model; the contract-literal identity is reported as main and the variant as sensitivity, with the same sign.
5. **External evaluation.** Score-only, no intervals, no adapter; the evaluator's Brier on external pre-states does not validate external labels by itself.
6. **Detector constants.** Sensitivity of secondary constants was measured on conference-era or pilot corpora and not re-run under the current corpus.
7. **Learners.** One logistic specification fixed a priori; the tree candidate used different inputs; no architecture ranking is claimed.

## Data and code availability — draft (adapted from `docs/tog_manuscript/sec_availability.tex`; double-anonymous)

The complete pipeline (engagement detector, constant estimation, label construction, feature extraction, the evaluator and predictor training scripts, and every evaluation and audit script) will be released under the MIT License together with the definition specifications, the fitted event-price table, the map anchors and a configuration preset that sets every constant of this article at once, and the derived, per-engagement data behind every reported number will be released with it. Raw match records are obtained from the Riot Games API under its Terms and Conditions and General Policies and are not redistributed; the article reports derived quantities only. What is available now, at submission, is limited to the derived result tables provided as supplementary material and an anonymised copy of the code \pending{anon-url}; a third party cannot yet reproduce every number end-to-end from this material alone (see open items below). The public repository, release tag and archival DOI will be identified after review \pending{release-doi}. The data licence for the derived files is \pending{data-licence}.

Open items that must close before the full pipeline release can be described as complete (carried from `docs/DATA_AVAILABILITY.md` §8 and `docs/tog_manuscript/sec_availability.tex`): the form of the anonymised copy (archive vs link) to be agreed with the editorial office; the learner scripts that live only on branch `codex/engagement-state-value` must be merged before the release tag (until then they are not part of the released tree); and the derived-data licence. The wording above distinguishes what is planned for release from what a reviewer can currently access.

## Statements — skeletons (author fills)

- **Author contributions (CRediT).** Conceptualization: ___; Methodology: ___; Software: ___; Validation: ___; Formal analysis: ___; Investigation: ___; Data curation: ___; Writing – original draft: ___; Writing – review & editing: ___; Visualization: ___; Supervision: ___; Project administration: ___; Funding acquisition: ___. (Omit from the anonymised submission; add at camera-ready.)
- **Funding.** ___ (camera-ready only).
- **Conflicts of interest.** The authors declare no competing interests. (Confirm.)
- **Ethics.** The study uses publicly available match records obtained through the Riot Games API under its terms; no human-subject interaction, no personal data beyond public game identifiers, which are not reported. (Confirm wording against `docs/DATA_AVAILABILITY.md` §3.)
- **AI-assistance disclosure (proposed).** Parts of the analysis code, documentation and manuscript drafts were prepared with the assistance of large-language-model tools operated by the authors; all numbers were produced by the released scripts, all references were verified against the sources listed, and the authors take full responsibility for the content. (Author decides whether and where to place it; check IEEE policy on submission day.)

## Claim–evidence map (conclusion)

| Claim | Evidence | Status |
|---|---|---|
| ≈ 0.0037 (T) and ≈ 0.0034 (S) lift, intervals exclude 0 | Results §3.1 | supported |
| "small; base-rate constant Brier ≈ 0.25 in this cohort" | UNC = base-rate Brier ȳ(1−ȳ) ≈ 0.25 (Results §3.3); not an irreducible-uncertainty bound | supported (as a reference scale, not a predictability limit) |
| Fight intervals move the estimate more than matched quiet intervals | Results §2.2 (0.0633 [0.0612, 0.0654]) | supported (scale contrast, not causal) |
| Label agrees with material/objective outcomes where decided | Results §2.3 (0.904; 0.604 vs 0.396) | correspondence only |
| Lift survives small-change filters | Results §4 | post-hoc |
| External T lift does not hold; discrimination still favours model | Results §5 (no intervals) | ordering only |
| Pooling helps neither | Results §3.4 | secondary |
