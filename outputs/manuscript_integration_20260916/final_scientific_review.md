# Final scientific review — correction-pass checkpoint

Date: 2026-09-16. Scope at this checkpoint: corrected `docs/tog_delta_v_20260916/manuscript.md` only, SHA256 `2A3C35178E1A9066802088E61D30F54FD9A9E19A7D6BCAD6C3BBC5F81DE4BA47`. Other bundle files were still being corrected and are not accepted here. Targeted recheck against initial findings and revision-request items 1–15; no experiments, manuscript edits, or full-source reread.

## Remaining findings

1. **Medium, scientific acceptance blocker — definition sensitivity inherits untested neural comparisons.** `manuscript.md:98-102` contribution 4 jointly attributes preserved residual-MLP-versus-LightGBM and other-engagement neural comparisons to both horizon sensitivity and TRAIN-only definition constants. The definition-dev study evaluates frozen PT/logistic/LightGBM and refit logistic/LightGBM, not MLP arms. Split the sentence: neural comparison patterns are supported by horizon sensitivity; boundary sensitivity supports its own fixed/refit tabular comparisons and changed-population counts. Source: `outputs/definition_dev_20260916/REPORT.md`, model tables and paired contrasts; manuscript §8.4 itself.

2. **Medium, remaining qualification — globally unresolved-difference rationale persists.** `manuscript.md:850-851` says “differences of the size seen here are not resolved by this comparison” immediately after describing residual-MLP differences that are resolved. Remove that blanket rationale or limit it explicitly to the logistic/LightGBM/plain-MLP contrasts whose intervals contain zero. Source: manuscript §8.2 tables and corrected §9 at 844–848; revision-request item 1.

## Resolved in the inspected manuscript

- Frame diagnostics: all-row 35.145%, T 35.27%, same-pre/post-frame 2,224/32,981 (6.74%) separated; elapsed-time contributions retained (§5.4, §8.6, §9, §10.3).
- Principal equivalence/composition headlines: no-detected-benefit and inconclusive contrasts replace tie/immaterial/absence claims; untested graphs and alternative representations remain open (abstract, §1.3, §8.2/8.5, §9, conclusion), except remaining qualification above.
- Horizon ranking: exact ranking changes named, cross-target easier-label inference removed, follow-up-only TEST tuning boundary preserved (§8.3).
- Definition robustness: external larger changes, N/external refit deterioration, original TEST dependence and no equivalence retained (abstract, §8.4, limitations, conclusion).
- Mechanism: mean absolute contributions distinguished from beta, structures 19.3% comparator corrected to 9.6% overall/7.9% >2pp, measured error bounds corrected, binary indicator identity used, champion transformed block corrected to 1,382 of 1,733 columns (§8.6).
- B40: LightGBM inconclusive versus exploratory logistic gain now preserved throughout; unsupported N mechanism removed (§8.1, §9).
- Statistical uncertainty: metric match bootstrap separated from SHAP row resampling and boundary-estimator intervals (§7.3, §10.6).
- Three historical label lineages now distinguished (§1.4).
- Predeclared exploratory protocol language replaces unverified preregistration; deterministic logistic fit distinguished from three-seed stochastic families (§8 preface, §8.2).
- Core label/split/OOF/retrospective/external limitations remain present in the targeted pass.

## Pending scope

Matrix-specific issue provenance, current README/map status, collaborator consistency, ledger table fixes and the revision receipt require later bundle review. Root owns exhaustive numeric/local-link/hash gates. Two small manuscript wording corrections above remain before scientific acceptance; no new experiment is required.

## Correction-pass checkpoint — matrix and collaborator specification

Inspected corrected `reviewer_response_matrix.md` SHA256 `95839D3AC3EDFBB5C91D7E60047D14C6570812DB1534E30ABB0EBB85488E0277` and `collaborator_specification.md` SHA256 `BF081904785670A174AA189EE0C743CB7399404502A0FC2EC710589D9D63BD7A`. Targeted review of previously identified semantic/status/attribution defects only; ledger/README remained under active correction.

- Initial findings 9–12 are resolved in the matrix: Schubert is correctly distinguished from the five unaudited audience references (line 79); new consolidated IDs and historical crosswalk are explicit (36–69); logistic candidate fits are separated from three-seed stochastic fits (94); Kim comparison is an author-proposed extension and objective ablation is our outgoing local result (183–184).
- Repeated tie/equal-Brier/null/ranking/frame mistakes are corrected in the matrix's R1-2, R2-2/R2-3 and CE-2/CE-3 rows. B40 logistic evidence, multiplicity and attribution interval limitations are explicit. Completed/partial/open states continue to distinguish executed analyses from unexecuted architectures and semantic validation.
- The collaborator specification now separates interval types, conditions non-detection and horizon patterns correctly, distinguishes same-frame from last-kill staleness, corrects 1,382 champion columns and mechanism quantities, reports worsening definition cells, and keeps Kim comparison as a proposed extension rather than a collaborator request. Known independent findings from the initial pass are resolved there.
- No additional confirmed material blocker found in this targeted pass. Root's planned SHAP bound correction (<4e-16) still applies across the bundle.

Provenance precision note for root: matrix CE-2/CE-3 now carry `(incoming)` labels for baseline comparison and explanation. If the original incoming email only asked for more balanced states, label these as the authors' operationalization of that concern rather than exact incoming requests. The incoming source was not independently re-read in this checkpoint, so this note is conditional, not a verified additional finding.

The two manuscript wording findings recorded above remain pending root integration; final disposition awaits those edits and final bundle gates.

## Final disposition — after root integration

**Accepted as a scientifically qualified working draft within this review's scope. No unresolved material scientific finding remains from the recorded review. This is not submission-ready acceptance.**

Targeted closure verified after `final_wording_fixes.py`:

- Manuscript contribution 4 now separates neural/tabular horizon comparisons from PT/logistic/LightGBM-only definition sensitivity; it explicitly says the latter did not test neural robustness.
- The blanket “differences of the size seen here are not resolved” rationale has been removed; named resolved and inconclusive contrasts remain distinct.
- SHAP reconstruction claims now use a valid <4e-16 / within 4e-16 bound rather than the too-tight 3.3e-16 bound.
- Definition robustness is no longer an all-metric 0.0002 bound. Manuscript/specification/ledger distinguish specific Brier changes from main-TEST AUC changes of about 0.00075 and larger external Brier changes, with the retained root metric-scope check cited in the ledger.
- Matrix CE-2 and CE-3 explicitly say they are the authors' operationalization of the incoming balanced-state concern, closing the conditional attribution note.

Final reviewed SHA256 values:

| File under docs/tog_delta_v_20260916/ | SHA256 |
|---|---|
| manuscript.md | 17C613ABA1D19682EB90AFF0F210441238886FD33CB9C373891CB7360FE33310 |
| collaborator_specification.md | FB0628D20493709CA8E0A085734A8673C2FB94C5CCD2B36240BB84B2D7B03B57 |
| reviewer_response_matrix.md | 28208A9B6B28779187A5CC154CD21EFA1E835A5C60D06ACA04C727E9A6F9B894 |
| claim_evidence_ledger.md | 98BA09B6D8C944EB6D4204A7DE9E7418DC7F32438D8598A36B3DEC8B3943F0A6 |

The checkpoint history above is preserved; its pending findings are superseded by this disposition. This final step was a targeted closure review, not a new exhaustive audit of every number or experiment. Root retains ownership of document/link/numeric/preservation gates and final acceptance receipt; the README link-parser repair was still pending at dispatch and is outside these four hashes. Existing scientific limitations remain explicit: exploratory TEST-exposed evidence, model-defined label without independent semantic validation, unexecuted model families, and no demonstrated live operation. No experiments or original scientific sources were modified during this review.


Final hash refresh: verified punctuation-only changes at manuscript line 659 and collaborator specification line 752; numerical values and scientific scope are unchanged. The final hash table above now records the current files. Root reports the README link repair and document gate passed 702/702; that gate was not independently rerun by this reviewer. Scientific working-draft acceptance remains unchanged.
