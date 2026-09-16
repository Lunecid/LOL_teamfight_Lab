# Citation audit for DeltaV manuscript integration

Checked 2026-09-16. Scope is limited to the six central references requested by the
manuscript integration. Bibliographic facts below were checked against the paper
itself or an author/institution repository/page. The cited papers are precedents
for particular methods or problem framings; none validates the complete DeltaV
design.

## Verified references and bounded roles

### Kim, Lee, and Chung (CoG 2020)

**Bibliographic fact.** Dong-Hee Kim, Changwoo Lee, and Ki-Seok Chung, “A
Confidence-Calibrated MOBA Game Winner Predictor,” *IEEE Conference on Games
(CoG)*, 2020, pp. 622–625, DOI
[10.1109/CoG47356.2020.9231878](https://doi.org/10.1109/CoG47356.2020.9231878).

**Primary sources.** [Conference paper PDF](https://ieee-cog.org/2020/papers/paper_221.pdf);
[Hanyang author/institution record](https://scholarworks.bwise.kr/hanyang/handle/2021.sw.hanyang/3695).

The paper studies League of Legends winner probability and proposes calibration
that models input-dependent, data-related uncertainty. Its reported method is a
neural density/MLP model with a data-uncertainty-aware loss and Monte Carlo
softmax calibration; the paper evaluates ECE, MCE, and NLL in addition to
accuracy. This supports citing Kim et al. for the motivation to assess probability
calibration and uncertainty.

**DeltaV boundary.** The manuscript’s `V` is a raw logistic/state-value model.
Raw, sigmoid, or isotonic calibration in this project does not reproduce Kim et
al.’s neural architecture or uncertainty-aware loss. Their calibration numbers,
feature vector, player population, and model performance must not be presented
as DeltaV results or as evidence that the project implemented their method.

### Maymin (JQAS 2021)

**Bibliographic fact.** Philip Z. Maymin, “Smart kills and worthless deaths:
eSports analytics for League of Legends,” *Journal of Quantitative Analysis in
Sports*, 17(1), 2021, pp. 11–27, DOI
[10.1515/jqas-2019-0096](https://doi.org/10.1515/jqas-2019-0096).

**Primary sources.** [Author publication page](https://philipmaymin.com/academic-papers);
[paper PDF](https://d-nb.info/1367424143/34).

Maymin’s “In-Game Win Probability” section fits a logistic regression using
elapsed minutes and both teams’ cumulative kills, towers, and large monsters,
with the completed game outcome as the label. To reduce within-game row
dependence, the analysis samples one random minute from each game for that
modeling step. The paper then evaluates events by changes in its estimated win
probability. This is the closest direct precedent among the six for a simple
time/state logistic LoL win-probability model and for using a frozen probability
estimate to describe event-associated change.

**DeltaV boundary.** The resemblance is methodological lineage only. It does not
validate this project’s feature schema, patch/cohort population, engagement
label, `V`, `q`, endpoint, calibration result, or causal interpretation. Do not
claim reproduction of Maymin’s high-frequency instrumentation, millions-game
corpus, or smart-kill/worthless-death framework. A DeltaV change remains a
model-derived probability difference over the project’s defined observation
window.

### Hodge et al. (IEEE ToG, issue 2021; online 2019)

**Bibliographic fact.** Victoria J. Hodge, Sam Devlin, Nick Sephton, Florian
Block, Peter I. Cowling, and Anders Drachen, “Win Prediction in Multiplayer
Esports: Live Professional Match Prediction,” *IEEE Transactions on Games*,
13(4), 2021, pp. 368–379, DOI
[10.1109/TG.2019.2948469](https://doi.org/10.1109/TG.2019.2948469).

**Primary source.** [University of Leeds/White Rose author repository record and
accepted manuscript](https://eprints.whiterose.ac.uk/id/eprint/152931/).

The paper develops and deploys live professional DotA2 win-prediction models.
Its running protocol uses a five-minute sliding state window and separate
minute-indexed models once that window is available, with chronological
evaluation and a tournament deployment. It is therefore evidence for live
runtime MOBA prediction and for reporting data-availability, temporal, and
domain limits.

**DeltaV boundary.** Hodge et al. is DotA2 work with a different model protocol;
it does not establish the project’s LoL `V`, exact features, calibration, `q`,
`DeltaV`, or an engagement effect. Their reported prediction accuracies (including
the abstract’s “up to 85%” live result) are not project results and must not be
transferred to this manuscript.

### Halfaker et al. (WWW 2015)

**Bibliographic fact.** Aaron Halfaker, Oliver Keyes, Daniel Kluver, Jacob
Thebault-Spieker, Tien Nguyen, Kenneth Shores, Anuradha Uduwage, and Morten
Warncke-Wang, “User session identification based on strong regularities in
inter-activity time,” *Proceedings of the 24th International Conference on World
Wide Web*, 2015, pp. 410–418, DOI
[10.1145/2736277.2741117](https://doi.org/10.1145/2736277.2741117).

**Primary sources.** [University of Minnesota author/institution record](https://experts.umn.edu/en/publications/user-session-identification-based-on-strong-regularities-in-inter/);
[author arXiv copy and full text](https://arxiv.org/abs/1411.2878).

The method constructs inter-activity-time distributions, inspects their
log-scaled histogram, fits within-session and between-session components with an
EM-fitted Gaussian mixture on log times, and selects a threshold at the
likelihood intersection. The paper gives about one hour as a useful rule of
thumb, while showing that fitted intersections vary across logs (examples range
from roughly 29 to 115 minutes).

**DeltaV boundary.** `G = 13.7 s` is this project’s pooled-data KDE-valley
estimate. It is not a value reported by Halfaker et al., and the KDE valley and
Halfaker’s Gaussian-mixture intersection are different estimators applied to
different populations. The citation supports distribution inspection and
mixture-based temporal separation only; it does not justify the exact engagement
window, late-game behavior, or a causal claim.

### Lundberg and Lee (NeurIPS 2017)

**Bibliographic fact.** Scott M. Lundberg and Su-In Lee, “A Unified Approach to
Interpreting Model Predictions,” *Advances in Neural Information Processing
Systems 30 (NeurIPS 2017)*, pp. 4765–4774.

**Primary sources.** [Official NeurIPS paper PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf);
[official proceedings abstract](https://papers.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

The paper introduces SHAP as a unified additive feature-attribution framework.
For a selected model/output, Shapley-based attributions are defined relative to
an explanation mapping (including the conditional-expectation formulation) and
sum to the represented model output under the stated assumptions. Feature
dependence, background data, and the mapping from features to the explanation
space affect interpretation.

**DeltaV boundary.** The project’s qSHAP uses seven fixed groups and includes
fixed `p_pre` as an input group. It explains the fitted `q` prediction under that
chosen grouping and interventional background-masking mapping. The project's
masking implementation is not a conditional-distribution estimator. It does not explain `V` or
`DeltaV` directly, establish causal engagement mechanisms, or show that a feature
causes a teamfight win. The manuscript should call these model-output
attributions and document the grouping and background choices.

### Schubert, Drachen, and Mahlmann (2016)

**Bibliographic fact.** Matthias Schubert, Anders Drachen, and Tobias Mahlmann,
“Esports Analytics Through Encounter Detection,” *MIT Sloan Sports Analytics
Conference, Research Papers Competition*, 2016, paper 1458.

**Primary sources.** [Author-hosted paper PDF](https://andersdrachen.com/wp-content/uploads/2014/07/esportsanalytics_ssac.pdf);
[archived official conference page](https://web.archive.org/web/20170626141008/http://www.sloansportsconference.com/content/esports-analytics-through-encounter-detection/);
[archived official conference PDF](https://web.archive.org/web/20171017210220/http://www.sloansportsconference.com/wp-content/uploads/2016/02/1458.pdf).

The paper defines Dota encounter components from spatio-temporal links among
units, using team/opponent relations and distance/range rules, and associates
encounter summaries with outcome and win-probability prediction. Its reported
data contain many encounters without a kill: 18,744 of 23,110 encounters
(81.1%) had no kill event. This supports citing it as a spatial/temporal
encounter-segmentation and outcome-modeling precedent.

**DeltaV boundary.** The paper’s Dota encounter definition is not the project’s
kill-conditioned localized engagement population. It does not justify the
project’s exact time/distance gates, `G`, `D`, `B`, or `R`, and does not establish
that an engagement causes a positive `DeltaV` or a win. Keep no-kill encounters
and label-population differences explicit when using the citation.

## Version and date assumptions

- Hodge et al. was accepted in 2019, published online on 2019-11-19, and appears
  in *IEEE Transactions on Games* volume 13, issue 4 (2021). Use the 2021 issue
  citation with the DOI; mention the online 2019 date only when explaining a
  metadata discrepancy.
- Maymin’s paper was published online on 2020-09-21 and assigned to the 2021
  JQAS volume/issue. Use 2021 for the journal citation.
- Halfaker et al. is a 2015 WWW publication. The arXiv author copy is useful for
  method verification but is not the publication date to cite.
- Lundberg and Lee is the NeurIPS 2017 proceedings paper, pp. 4765–4774; the
  official proceedings PDF is the checked source.
- Schubert et al. is a 2016 conference research paper. The old Sloan page/PDF
  are available through the archived official URLs; the author-hosted PDF is a
  second direct copy. No journal DOI or issue metadata was used.
- Kim et al. is the IEEE CoG 2020 paper. The supplied conference PDF is the
  checked primary text; the Hanyang record supplies the page and DOI metadata.

## Manuscript wording contract

Use these narrow claims in the integration:

1. Maymin supports the general time/state logistic win-probability lineage for
   LoL. `V` remains this project’s raw logistic model, not Kim et al.’s neural
   uncertainty-aware architecture.
2. Halfaker supports inspecting inter-activity distributions and separating
   temporal components. `G = 13.7 s` is the project’s KDE-valley estimate, not
   Halfaker’s approximately one-hour mixture-intersection rule.
3. Lundberg and Lee support SHAP as model-output attribution. Seven-group qSHAP
   with fixed `p_pre` is an implementation choice and is not causal evidence.
4. `DeltaV` (whether written as `p_post - p_pre` or as `V(S_e)-V(S_s)` in the
   manuscript’s notation) is an observed model-probability change over a defined
   window. It is not an engagement causal effect or an observed counterfactual.
5. Schubert et al. support encounter segmentation as a Dota spatial/temporal
   precedent. Their encounter population and label are different, including a
   large no-kill component.

The local integration contracts corresponding to these boundaries are
`docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md` (especially §§1–3) and
`docs/DEFINITION_EVIDENCE_REGISTER_20260914.md` (especially §§2–5). The original
candidate metadata audit is
`C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript/references_audit.md`.

## Implementation-affecting uncertainty

The only material bibliographic ambiguity is publication dating: Hodge has a
2019 online record and 2021 issue assignment, while Maymin has a 2020 online
record and 2021 issue assignment. The method boundaries above are not affected.
For SHAP, the exact interpretation depends on the project’s group definitions,
background rows, and conditional/independence assumptions; these choices should
remain explicit in the manuscript. The sources do not provide a basis for
generalizing across patches, regions, endpoints, or causal engagement effects.
