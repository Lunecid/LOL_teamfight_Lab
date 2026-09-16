# Definitions and evidence: cohort / role-aware q experiment (2026-09-15)

Each entry: definition, status (fixed input / specification rule / implementation choice / empirical evidence) and where the evidence is.

| Term | Definition | Status | Evidence |
|---|---|---|---|
| Engagement population E | every detected engagement row of the frozen full-corpus labels (566,452 main; 54,686 external); comparisons use rows valid at the horizon | fixed input | cohorts/cohort_manifest.json join |
| Participation count | stored v3.3 det_cluster_blue/red: kill participants plus in-radius (3,000 u) interaction actors incl. shop events; merged engagements keep the earlier candidate counts | fixed input (v3.3 detector) | manuscript sec_definition.tex scale subsection; gameplay/fights.py; detector_train_fixture.json |
| Negative count | -1 = int(0 or -1); read as 0 only when re-detection proves a raw integer 0 at the same (s, L) | specification rule; empirical proof | cohorts/negative_count_provenance.json |
| Teamfight T | min(cluster_blue, cluster_red) >= 4 (manifest scale.teamfight_min) | fixed convention (v3.3), not a universal threshold | corpus_shards_v33/manifest.json |
| Non-teamfight N | known scale and min < 4; fine classes pick (<= 1) and skirmish (2..3) are diagnostics | specification rule | cohort_manifest.json |
| Unknown scale | negative count not proven zero; excluded from T/N, kept in E (none observed) | specification rule | cohort_manifest.json |
| Label Y_h | 1[V(endpoint_h) - V(s - 1 ms) > 0], same adapter; TRAIN held-out-fold V, others final V; h90 primary | fixed input (frozen full run) | outputs/full_corpus_training_20260915/labels |
| q | probability of generated Y from pre-only inputs (StateV2 at q_pre minus snapshot age, plus p_pre_V) | fixed input schema | full run q_pre_only_schema.json |
| Pooled q | frozen full-run choice per horizon, fit on E | fixed reference | frozen_manifest.json pooled_reference |
| Specialist q | unchanged candidate pool fit on cohort TRAIN rows, calibrated on cohort Q_CAL, chosen on cohort Q_SELECT | specification rule | selection/q_specialist_*.json |
| Oracle-cohort routing | specialist chosen by the post-cutoff cohort label; not deployable | reporting device only | eval/results_A.json |
| Weak role annotation | cached role_slots of eligible 15.14 TRAIN teams (teamPosition-derived with silent ID fill); detectable fills excluded | specification fallback clause (raw main detail unavailable) | role_supervision_provenance.json |
| Meta-role classifier | champion one-hot + unordered spell multi-hot -> multinomial logistic regression C=1 | implementation choice fixed by the specification | role_models_manifest.json |
| Team role posterior W | marginals over 120 one-to-one assignments with weights prod p_i(role) | modeling assumption (one player per role) | cr20260915_common.team_role_posterior; tests |
| Role uncertainty summaries | per team: assignment entropy, mean/min max marginal, champion missing/unseen counts, spell missing/unseen counts | implementation choice | protocol.json interpretation_choices |
| Role representation | global features + role-weighted blue/red participant features, blue-minus-red differences, difference x time_minutes (gold/xp/level), uncertainty summaries | specification rule | cr20260915_common.build_role_matrix |
| Draft control | participant representation + per-slot champion one-hot + per-slot unordered spell multi-hot (cohort TRAIN encoders) | specification rule; unscaled indicators are an implementation choice | selection/arms_*_h90.json |
| Full-feature participant LightGBM | economic-model parameters on all 352 numeric pre inputs | new model (identified as such) | selection/arms_*_h90.json |
| Role-block ablation | role ridge refit without one role group (51 columns), same calibration family as the preselected full role ridge | model ablation, not causal importance | selection/ablations_T_h90.json |
| Group Shapley | exact interventional Shapley over 7 groups on the final calibrated probability; TRAIN background | descriptive explanation | shap/shap_summary.json |
| Role reliability | final role model vs external raw teamPosition (post-freeze) | empirical evidence (transfer to later patches) | eval/role_reliability_external_raw.json |

## Hypotheses versus evidence

- Hypothesis: separate T and N q pools capture scale-specific structure. Evidence: section 4 of REPORT.md (paired, identical rows). Cohort choice itself is not a live input.
- Hypothesis: role-organized participant information adds predictive signal beyond slot order and beyond direct draft indicators. Evidence: section 6 arm-winner pairs (controlled within cohort and base) and T ablations; redundancy with global totals is expected.
- Not claimed: causal lane importance, validated role accuracy on the main corpus, a deployable routing model, or a confirmatory TEST result.

Verification: validation.json (42 checks, failed: none).