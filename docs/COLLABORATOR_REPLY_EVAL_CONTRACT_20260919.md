# Draft reply — collaborator evaluation-contract review (2026-09-19)

Thank you — we agree with the diagnosis and will revise before expanding models.

**Keep:** SVI as primary target (occurrence of upward \(\widehat{V}\) revision, not fight win); binary label; continuous \(\Delta V\) not required as primary; research goal unchanged.

**Change (now locked in the three protocol docs):**

1. **Corpus ≠ eval sample.** Main prediction table: all of \(p_{\mathrm{pre}}\), \(b(p)\), PT, \(q\) on the **same 15.16 T rows** (\(n=32{,}981\)). Pooled T only for measurement/concordance. We will not juxtapose pooled \(p_{\mathrm{pre}}\) AUC with 15.16 ΔBrier as one “lift.”

2. **Primary contrast = \(q-\mathrm{PT}\)** (overall + B40). \(b(p)\) secondary; raw \(p_{\mathrm{pre}}\) diagnostic only (different estimand). Research question reframed as *how much beyond PT*, not a pre-committed positive result.

3. **Material:** split concordance vs cross-target retrain vs calibrated kill→SVI substitute. Soften language to “correspondence with observed outcomes.” Fix decided-set denominators/weights so agree+disagree=1; separate ties.

4. **Quiet:** keep same-match, add matching on \(p_{\mathrm{pre}}\), clock, duration; report signed \(\Delta V\) and \(P(+)\) as well as mean \(|\Delta V|\) ratio (statistic named explicitly).

5. **Models:** lean slate first (logistic / LightGBM / MLP + TabM). FT/TabNet/SAINT only if needed for reviewer response; SAINT inference contract to be specified. Matched-352 scope ≠ “graphs invalid.”

6. **Transfer:** freeze \(\widehat{V}\), \(q\), PT, \(b(p)\), preproc, calibrators; report \(\widehat{V}\to W\) and \(q\to\mathrm{SVI}\) separately.

7. **Epistemic:** exploratory after prior exposure; this comparison’s rules frozen before sealed reopen — not confirmatory.

We will execute in that order (eval contract → lean select → concordance/quiet → transfer), not by growing the architecture list first.
