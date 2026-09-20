# Related Work — brief, source table and draft paragraphs (T013, 2026-09-21)

*Status: proposal. Method: research-paper-writing skill (related-work guide: 2–4 topics, paradigm → limitation → our distinction) and the citation-verification rule that no reference may be invented. Every key below already exists in one of two BibTeX files in the repository; nothing is added from memory. Keys marked* **[verify]** *exist in the bib but were not fetched by the reference audit (see `docs/tog_manuscript/references_audit.md` §5); keys marked* **[unwired]** *are in the econometrics bib but not yet cited in any `.tex`.*

## Source pools (both verified to exist in the repo)

| Pool | File | Entries | Status |
|---|---|---|---|
| A. Definition / esports / learners / calibration | `docs/references/definition_refs.bib` | 105 (all carry a VERIFIED/UNVERIFIED marker; 0 duplicates per `references_audit.md` §1) | wired into `docs/tog_manuscript/main.tex` |
| B. Forecast evaluation / direction of change / cluster inference | `docs/literature/LOL_ECONOMETRICS_LITERATURE_20260920/econometrics_for_lol.bib` | 17 | not wired; `Maymin2021` duplicates pool A `maymin2021smart` (keep A's key) |

Named-but-unbibbed works listed in `docs/RELATED_FORECAST_VALUE_LIT_20260919.md` (Chong & Hendry; Fair & Shiller; Clark & McCracken; Lock & Nettleton; iWinRNFL; WPA/LI glossaries; a 2026 Serie A preprint) **must not be cited** unless a verified entry is added first (`[CITATION NEEDED]` rule).

## Topic design (four topics, each ends with our distinction)

| Topic | Paradigm to summarise | Limitation tied to our challenge | Our distinction | Keys |
|---|---|---|---|---|
| T1. Win-probability models in MOBAs | Match winner from picks/player history; in-match win probability from per-minute state; live professional prediction; calibrated MOBA predictors | They estimate *match* outcome; none defines a fight-level target, and the state evaluator is the product, not a fixed instrument | We freeze the evaluator and use it only to define a fight-interval direction label | costa2021, hitar2023, hodge2021win, silva2018continuous, bisberg2022, kim2020confidence, jalovaara2024win |
| T2. Encounter and teamfight detection / prediction | Encounters from unit proximity and damage (Dota replays); OpenDota's kill-window heuristic; camera- and death-based teamfight prediction seconds ahead; teamfight features for match prediction | Targets are whether/when a fight happens or who dies; the consequence of the fight for the match is not measured | We take the detected unit as given (data-derived constants) and measure its consequence under a frozen evaluator | schubert2016encounter, opendota2018processteamfights, tot2021camera, katona2019time, ke2022, halfaker2015session |
| T3. Valuing events by win-probability change | "Smart kills / worthless deaths" (LoL); item value by win probability added; action values in soccer; event-study scale versus signed mean | Event valuation assumes the evaluator; direction and mean of the change are treated as one object; no test of predictability *before* the event beyond p and t | We separate direction, mean and scale (RR4 triad), and test pre-event predictability beyond flexible p,t baselines; Maymin 2021 is the direct LoL prior, so we do not claim the first win-probability-change study | maymin2021smart, jalovaara2024win, DecroosEtAl2019 [unwired], BrownWarner1985 [unwired], ChristoffersenDiebold2006 [unwired] |
| T4. Proper scores, calibration and paired forecast comparison | Strictly proper scoring rules; CORP reliability and Brier decomposition; Platt/temperature/binned calibration; forecast-comparison and conditional-predictive-ability tests; cluster-robust inference | Standard tests assume time-series or i.i.d. losses; fight rows cluster within matches; recalibration on the evaluation sample is diagnosis, not a model | We use CORP as a decomposition only, paired match-cluster bootstrap for ΔBrier (not renamed as a Diebold–Mariano or Giacomini–White test), and identity calibrators chosen on a separate patch | GneitingRaftery2007, DimitriadisGneitingJordan2021, platt2000probabilities, guo2017calibration, naeini2015obtaining, DieboldMariano1995, GiacominiWhite2006, CameronMiller2015, efron1993bootstrap, HansenLundeNason2011 [unwired, optional] |
| (T5, optional, one sentence) Tabular learners | Trees vs deep tabular models | Not our question; the logistic q is a fixed specification | State that no architecture search was run (Discussion §5) | grinsztajn2022tree, gorishniy2021revisiting, shwartzziv2022tabular, ke2017lightgbm |

## Guardrails from the repository (must survive into prose)

From `docs/ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md` "Forbidden slides from lit" and `docs/LIT_RESULT_BRIDGE_RR46_20260920.md`:

- Do not equate ΔV with financial returns or market efficiency; do not write "martingale ⇒ sign unpredictable".
- Do not write "large E|ΔV| ⇒ direction is predictable".
- Do not call the match-bootstrap ΔBrier a Diebold–Mariano or Giacomini–White test; cite them only as motivation for paired, conditional comparison.
- Do not claim the first LoL win-probability-change evaluation (Maymin 2021 is direct prior).
- CORP MCB must be BS − BS_iso; no TEST-fit recalibration presented as a frozen-model upgrade.
- Do not infer 16.x external failure modes from 15.16 CORP.

## Draft paragraphs (≈ 620 words; author confirmation required)

**Win-probability models in MOBAs.** Match-outcome prediction in League of Legends and Dota 2 spans pre-game inputs such as picks and player history [costa2021; hitar2023], in-match state summarised per minute [silva2018continuous], live professional matches [hodge2021win], and graph models over a league's season [bisberg2022]. Calibration of such predictors has been studied directly [kim2020confidence], and learned evaluators have been used to value items by the win probability they add [jalovaara2024win]. In all of this work the evaluator is the product. We use one as an instrument: it is trained on match outcomes, frozen, and thereafter serves only to define what a fight did to the estimated win probability.

**Encounter and teamfight detection.** Encounters in Dota 2 were first detected from unit proximity and damage on replay data, where most encounters contained no kill [schubert2016encounter]; the public OpenDota processor opens a fight window fifteen seconds before a hero death and closes it after fifteen seconds without one [opendota2018processteamfights]. Later work predicted deaths or team fights seconds ahead from camera and state features [katona2019time; tot2021camera] and fed detected fights into match prediction [ke2022]. These targets ask whether or when a fight happens, or who dies in it. Our detector follows the same lineage, with the kill-gap and cluster-diameter constants estimated from the data rather than set by hand, and the fight is then treated as the unit whose consequence is measured, not predicted.

**Valuing events by win-probability change.** The idea that an in-game event is worth the win probability it adds is established in League of Legends, where kills and deaths were classified as smart or worthless by a logistic win-probability model [maymin2021smart], and in soccer action valuation [DecroosEtAl2019]. Event studies in finance separate the scale of a change from its signed mean [BrownWarner1985], and direction-of-change forecasting shows that the sign of a change can be predictable while its mean is not, or the reverse [ChristoffersenDiebold2006]. We keep the three objects apart, direction, mean and scale, and ask a question those valuation studies do not: whether the sign of a fight's change is predictable from public state before the fight, beyond what the current win probability and the clock already imply. Maymin's study is the direct prior for win-probability change in this game; ours differs in the unit (the fight interval), the frozen evaluator, and the test against flexible baselines.

**Proper scores, calibration and paired comparison.** We report Brier score and log loss, both strictly proper [GneitingRaftery2007], and decompose the Brier score into miscalibration, discrimination and uncertainty with the CORP construction [DimitriadisGneitingJordan2021], used here as a diagnostic on the evaluation sample and not as a fitted recalibrator. Post-hoc calibrators such as Platt scaling and temperature scaling [platt2000probabilities; guo2017calibration] and binned calibration measures [naeini2015obtaining] are standard; we fit a positive-slope sigmoid on a separate calibration patch and keep it only when it improves a selection patch, which for the reported models it did not. Paired comparisons of forecasts have a long history in econometrics [DieboldMariano1995; GiacominiWhite2006]; because engagements cluster within matches, we use a paired match-cluster bootstrap [efron1993bootstrap; CameronMiller2015] and do not present its intervals as those tests.

**Tabular learners (one sentence, optional).** Tree ensembles and deep tabular models trade places across benchmarks [grinsztajn2022tree; gorishniy2021revisiting; shwartzziv2022tabular]; this article fixes a logistic specification a priori and runs no architecture search, so it makes no claim about learners.

## Checklist (related-work guide)

- Strongest competitors covered: Maymin 2021 (direct prior), Hodge 2021 / Hitar 2023 (ToG match prediction), Schubert 2016 / Ke 2022 (fight detection). ✓
- Each topic connected to our setting with a technical distinction, not a marketing one. ✓
- Citation coverage for core claims: object separation (Christoffersen–Diebold), CORP (Dimitriadis et al.), cluster bootstrap (Cameron–Miller, Efron). ✓
- Open: page ranges for `gorishniy2021revisiting` (references_audit §5); five pool-A sources could not be fetched by the audit and keep their [verify] status; the author decides whether pool B entries marked [unwired] enter the journal bib (they are needed for T3/T4).
