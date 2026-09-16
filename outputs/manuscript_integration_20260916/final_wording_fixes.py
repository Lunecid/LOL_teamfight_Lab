from pathlib import Path

root = Path(__file__).resolve().parents[2]
bundle = root / 'docs/tog_delta_v_20260916'

def replace(name, old, new):
    p = bundle / name
    text = p.read_text(encoding='utf-8')
    assert text.count(old) == 1, (name, old, text.count(old))
    p.write_text(text.replace(old, new), encoding='utf-8')

replace('manuscript.md', '''4. **Sensitivity results that bound the definition's influence.** Endpoint caps of 60/90/120 s (§8.3) and
   engagement constants re-estimated from the training patch alone (§8.4) preserve the particular broader
   comparisons named in those sections — LightGBM over the baseline, the residual MLP behind LightGBM, trees ahead
   on other engagements — while exact point-estimate rankings do move and several cells change measurably
   (§8.4 reports which ones worsen). The populations these variations induce differ by amounts we report
   separately rather than as a single "effect".''', '''4. **Sensitivity results with experiment-specific scope.** The 60/90/120 s endpoint study (§8.3) reports
   the tabular and neural comparisons, including changed point-estimate rankings. The TRAIN-only engagement
   definition study (§8.4) evaluates PT, logistic and LightGBM arms; it does not test neural-model robustness.
   Its main teamfight refit-versus-frozen contrast is inconclusive, while other engagements and some external
   cells worsen. Changed matches, removed rows and added rows are reported separately.''')
replace('manuscript.md', '''input contract, the same reinforced baseline and the same match-bootstrap protocol, because differences of the
size seen here are not resolved by this comparison.''', '''input contract, the same reinforced baseline and the same match-bootstrap protocol.''')
replace('manuscript.md', '''so a blanket "sealed metrics stay within 0.0002" is false outside the main-TEST cells.''', '''These are Brier comparisons across different row populations, not a bound on every metric: even main-TEST
AUC changes reach about 0.00075 in the recorded frozen-model cells.''')
replace('manuscript.md', '''re-derivation moves `D` outside the pooled interval; on main-TEST cells the frozen metrics move by about 0.0002 or
less (T LightGBM 0.228391 → 0.22837), but external frozen cells move by up to roughly 0.0007, the refit-versus-frozen''', '''re-derivation moves `D` outside the pooled interval; the main-TEST T frozen LightGBM Brier changes from
0.228391 to 0.22837, while the external KR 16.15 N frozen Brier changes by roughly 0.0007. These Brier examples
do not bound AUC changes. The refit-versus-frozen''')
replace('collaborator_specification.md', '''so "sealed metrics stay within 0.0002" holds only for the main-TEST cells.''', '''These are Brier comparisons across different row populations, not a bound on all metrics: main-TEST AUC
changes reach about 0.00075 in the recorded frozen-model cells.''')
replace('claim_evidence_ledger.md', '''Main-TEST frozen metrics move by 0.0000–0.0002 between the old and dev populations; **that bound does not hold on external cells**, and these are descriptive comparisons across two different row populations, not a paired effect''', '''Main-TEST frozen Brier changes in recorded all/B40/B45 cells are below 0.00007; AUC changes reach about 0.00075 (root check: `outputs/manuscript_integration_20260916/definition_metric_scope_check.json`). External Brier changes can be larger. These are descriptive comparisons across different row populations, not a paired effect or an all-metric bound''')
replace('claim_evidence_ledger.md', '''True only for main-TEST cells; external frozen cells reach ≈0.0007, and N/KR 16.13 T refit contrasts are worse with intervals excluding zero''', '''Not an all-metric bound even on main TEST (AUC changes reach ≈0.00075); external frozen Brier changes reach ≈0.0007, and N/KR 16.13 T refit contrasts are worse with intervals excluding zero''')
replace('reviewer_response_matrix.md', '*(incoming)* Does a model beat the prior estimate in balanced states?', '*(our operationalization of the incoming balanced-state concern)* Does a model beat the prior estimate in balanced states?')
replace('reviewer_response_matrix.md', '*(incoming)* Explain balanced states, not only score them', '*(our operationalization of the incoming balanced-state concern)* Explain balanced states, not only score them')

for name in ['manuscript.md', 'collaborator_specification.md']:
    p = bundle / name
    text = p.read_text(encoding='utf-8').replace('≤3.3e-16', '<4e-16').replace('to 3.3e-16', 'within 4e-16')
    p.write_text(text, encoding='utf-8')
replace('claim_evidence_ledger.md', 'max abs additivity error 3.33e-16; `Σφ + base` equals the parent sealed prediction to 3.33e-16', 'max abs additivity error ≈3.33e-16; `Σφ + base` reproduces the parent sealed prediction with error <4e-16')
print('Final bounded scientific wording corrections applied.')
