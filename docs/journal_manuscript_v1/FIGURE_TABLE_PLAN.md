# Figure and table plan (T013, 2026-09-21)

*Method: dataviz skill (form first, colour last, validated palette, legend + shape encoding, table view) and artifact-diagramming (draw the mechanism, label the arrows). ToG prints line art in black and white unless colour is requested, so every figure uses black/grey plus marker shape. Three figures are already drawn as hand-authored SVG under `figures/` from values recorded in `docs/` (no `outputs/` access, no new numbers). Figures that need `outputs/` are listed for T014.*

## Main-text figures (target: 3; ToG 10-page budget)

| # | File | Form | Claim it carries | Data source (all in docs/) | Status |
|---|---|---|---|---|---|
| Fig. 1 | `figures/fig1_pipeline.svg` | mechanism diagram | Measurement → prediction → verification chain; where the evaluator is frozen; which rows get fold evaluators; what is evaluated per cohort | Methods §1–§6 (text only) | drawn; caption below |
| Fig. 2 | `figures/fig2_delta_brier_bins_TS.svg` | dot plot (A: 20 narrow p_pre bins, point estimates; B: four B40 bins with 95 % CI), T = black circle, S = grey open square | The lift over PT_flex is present across the p_pre range in both cohorts and is smaller near 0.5; intervals shown only where they exist | A: `REVIEW_RESPONSE_RR12_RESULTS_20260920.md` narrow-bin table; `…_S_qS_id.md` narrow-bin table. B: `…RR12_RESULTS_20260920.json` `balance_bins`; `…_S_qS_id.json` `balance_bins` | drawn; palette checked with validate_palette.js (light mode): adjacent-pair separation ΔE 40 and contrast pass; the chroma-floor check fails by design because the figure is greyscale for print, and identity is carried by marker shape (circle vs square) as the required secondary encoding |
| Fig. 3 | `figures/fig3_cohort_roles.svg` | sample-flow diagram + drawn role table | From 210 000 matches to the T and S rows used in each role; picks excluded a priori | `lineage_20260915/README.md` L82–88; `cohort_manifest.json`; `SCALE_SPLIT_RR0_MANIFEST_20260920.json` | drawn |

### Captions (draft)

- **Fig. 1.** Measurement and prediction chain. Engagements are detected on the public timeline, the frozen evaluator V̂ scores the state 15 s before the first kill and the state at the endpoint e_h (capped 90 s after the last kill), and the sign of the difference is the label. A regularized logistic model on pre-fight state is compared with a spline baseline in pre-fight win probability and time, separately for teamfights T and skirmishes S. TRAIN rows are labelled by fold evaluators; all other rows by the frozen bundle. Verification checks (bottom) bound what the label measures; none changes the estimand.
- **Fig. 2.** ΔBrier(q − PT_flex) by pre-fight win-probability bin on the 15.16 test patch, match-weighted (w = 1/n_m) within each bin. (A) Twenty bins of width 0.05, point estimates only (diagnostic; no intervals were computed per narrow bin). (B) The four bins inside the balanced slice B40 with 95 % match-cluster bootstrap intervals (2 000 draws, seed 7). T: 32 981 engagements; S: 101 205 (identity calibrator). Negative values favour q. The two cohorts share axes for reading convenience; the paper does not compare them.
- **Fig. 3.** Sample flow. 210 000 matches (KR, patches 15.14–15.16) yield 566 452 detected engagements, 566 104 valid at h90; classes by the smaller side's participation n_min; picks (n_min ≤ 1) are excluded a priori. Rows per role for T and S with match counts; EXT S match counts are not recorded.

## Main-text tables (target: 4)

| # | Content | Source | Notes |
|---|---|---|---|
| Table 1 | Data roles and cohorts (compact version of Methods Table 5b: role, patch, T rows/matches, S rows/matches, label source) | Methods §5 | Fig. 3 may replace it if space is short (keep one) |
| Table 2 | Primary panel: T block and S block; q, PT_flex, PT_linear; n, matches, Brier, log loss, AUC; primary ΔBrier with CI per block; PT_linear continuity; S sensitivity row | Results §3.1 | one primary per block, bold |
| Table 3 | Within-cohort slices and secondary contrasts: B40 per cohort (ΔBrier, H); transfer and pooling (4 rows) | Results §3.2, §3.4 | all identity; secondary label |
| Table 4 | External score-only, T rows and S rows: n, V_pre Brier, ΔBrier(q − PT_flex), ΔMCB, ΔDSC; "no interval" in caption | Results §5 | T and S listed, not contrasted |

## Supplementary figures/tables (S1–S6)

| # | Content | Source | Needs outputs/? |
|---|---|---|---|
| S1 | Detector constants table (value, source class, basis) | Methods §1 constants table | no |
| S2 | Two evaluator paths and per-fold OOF census | Methods §5; `Q_NEWV_FIT85_OOF_META_SLIM_20260920.json` | no |
| S3 | V̂→W time-band ledger + CORP by band (V_to_W_timeline, eng_pre, eng_post) | `TEST_BAND_LEDGER_MLP_FIT85`, `RR6A` rows | no (md tables) |
| S4 | Verification tables: RR3 arms and coverage bins; RR4 λ and cut-off tables and triad; RR5a axes; RR5b objectives; RR6b horizons | RR3/RR4/RR5/RR6b md | no |
| S5 | S-cohort sensitivity (RR12 selection variant) and the four S RR12 variants; SCALE_SPLIT manifest digests | `SCALE_SPLIT_TvsS_RESULTS` Tables 1b/2b; manifest | no |
| S6 (figure) | ΔV̂ distribution and P(SVI = 1) by p_pre bin and by time band (triad) | `NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md` L29–46; RR4 triad | no (md tables) |
| S7 (figure, optional) | Per-minute V̂ performance and density (already rendered PNGs) | `docs/figures/mlp_fit85_test_per_minute_*.png` | no (pre-rendered; re-render in B/W for print) |
| S8 (figure, optional) | Reliability (CORP) diagram for q and PT_flex on T and S | needs prediction tables | **yes** → T014 |

## Rules applied (dataviz / diagramming)

- One axis per panel; no dual axes. Categorical identity (T vs S) by fixed hue order plus shape; legend present; text in ink colour, never series colour.
- Palette `#111111` (T) / `#7a7a7a` (S) run through the skill's `validate_palette.js` in light mode: separation and contrast pass; the chroma floor fails because the palette is deliberately greyscale (ToG prints in black and white), so identity is also encoded by marker shape; both figures remain legible in greyscale print.
- Every figure states n, matches, weighting and interval status in its caption; narrow-bin panel says "no intervals".
- Diagrams label arrows with what moves (states, scores, labels); the frozen/never-refit property is written on the box that carries it.

## T014 items (Cursor, needs outputs/)

1. Reliability diagrams (CORP) for q and PT_flex, T and S, from `outputs/review_response_rr12_20260920*/prediction_table.npz` (script under `scripts/ss20260921_fig_reliability.py`; values must match RR6A/Table 3 MCB/DSC).
2. Export the three SVGs to EPS/PDF for IEEEtran; check greyscale rendering.
3. Optional: re-render `docs/figures/mlp_fit85_*` per-minute plots in black and white for S7.
