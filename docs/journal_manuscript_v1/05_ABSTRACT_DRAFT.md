# Abstract — DRAFT for author confirmation (T013, 2026-09-21)

*Status: proposal. Nothing here is final until the author confirms. Numbers are copied from `03_RESULTS.md` (T block from RR12; S block from `SCALE_SPLIT_TvsS_RESULTS_20260920.json` `table2a_identity`). Venue rule (ToG guidelines, read 2026-09-21): 150–200 words, one paragraph, no abbreviations, no references; 2–5 keywords.*

## Pre-writing answers (research-paper-writing skill)

1. **Problem and why no settled solution.** Before a fight in League of Legends, how much of its outcome is already in public match state? The conference line predicted an engagement's gold winner; that label is local and does not say whether the fight moved the match. No established target links a fight to the match outcome under a fixed evaluator.
2. **Contribution.** (a) A match-linked target: the direction of change in estimated match win probability across the engagement interval under a frozen evaluator. (b) A verification stack that separates evaluator quality from direction predictability and tests the lift against flexible functions of initial win probability and time, on two engagement cohorts.
3. **Why it works.** The evaluator is trained on match outcomes, frozen, and never refit for the prediction task; the direction label is therefore fixed before any predictor sees it, and every contrast is paired on identical rows.
4. **Advantage / insight.** Pre-fight state carries a small but interval-excluding proper-score lift beyond initial edge and time in both cohorts; the lift is bounded by score decomposition and by external score-only checks.

## Variant A (196 words; includes the external-transfer sentence pending `\pending{ext-TS-sentence}`)

Before a fight starts in League of Legends, how much of its consequence is already visible in public match state? Earlier work predicted which team wins an engagement; here the target is the direction of change in estimated match win probability across the engagement interval, measured by a frozen evaluator trained on match outcomes. On 210 000 matches from three patches we detect engagements with a data-derived definition, form two cohorts by the smaller side's participation (teamfights with at least four champions; skirmishes with two or three), and ask whether a pre-fight model predicts the direction of change beyond flexible functions of initial win probability and game time. Within each cohort, a regularized logistic model on pre-fight state lowers the Brier score relative to that baseline by about 0.0037 (teamfights) and 0.0034 (skirmishes) on a held-out patch, with match-cluster bootstrap intervals excluding zero; the teamfight model carries part of its signal to skirmishes, and pooling the cohorts helps neither. On later patches and another region the teamfight lift does not hold while the skirmish lift does, without intervals. Score decomposition, quiet-interval contrasts and next-objective correspondence bound what the label measures. The two cohorts are reported separately and are not ranked against each other.

## Variant B (184 words; external sentence reduced to the teamfight finding only)

Before a fight starts in League of Legends, how much of its consequence is already visible in public match state? Earlier work predicted which team wins an engagement; here the target is the direction of change in estimated match win probability across the engagement interval, measured by a frozen evaluator trained on match outcomes. On 210 000 matches from three patches we detect engagements with a data-derived definition, form two cohorts by the smaller side's participation (teamfights with at least four champions; skirmishes with two or three), and ask whether a pre-fight model predicts the direction of change beyond flexible functions of initial win probability and game time. Within each cohort, a regularized logistic model on pre-fight state lowers the Brier score relative to that baseline by about 0.0037 (teamfights) and 0.0034 (skirmishes) on a held-out patch, with match-cluster bootstrap intervals excluding zero; the teamfight model carries part of its signal to skirmishes, and pooling the cohorts helps neither. On later patches and another region the teamfight lift does not hold. Score decomposition, quiet-interval contrasts and next-objective correspondence bound what the label measures. The two cohorts are reported separately and are not ranked against each other.

## Keywords (ToG asks for 2–5)

League of Legends; esports analytics; win probability; proper scoring rules; engagement detection

## 국문 초록 (독립 작성, 학위논문·국내 발표용; 저널 제출본에는 포함하지 않음)

리그 오브 레전드에서 교전이 시작되기 전, 공개된 경기 상태에는 그 교전의 결과가 얼마나 담겨 있는가. 선행 학회 연구는 교전의 승자를 예측했지만, 그 라벨은 교전 국소적이며 경기 승패와 연결되지 않는다. 본 연구는 경기 승패로 학습한 뒤 동결한 평가기로 교전 구간 전후의 추정 승률 변화 **방향**을 정의하고, 이 방향을 교전 전 공개 상태로 얼마나 예측할 수 있는지를 묻는다. 세 패치 210 000경기에서 자료 기반 정의로 교전을 검출하고, 작은 쪽 참여 인원에 따라 한타(4인 이상)와 소규모 교전(2–3인) 두 코호트로 나누었다. 각 코호트 안에서 교전 전 상태로 학습한 정규화 로지스틱 모델은 초기 승률과 시간의 유연한 함수를 기준선으로 할 때 검증 패치에서 Brier 점수를 약 0.0037(한타)과 0.0034(소규모 교전) 낮추었고, 경기 단위 부트스트랩 구간은 0을 포함하지 않았다. 한타 모델은 소규모 교전으로 신호의 일부를 옮기지만, 두 코호트를 합쳐 학습하면 어느 쪽에도 이득이 없다. 이후 패치와 다른 지역에서는 한타의 이득이 유지되지 않았다(구간 없음). 점수 분해, 비교전 구간 대조, 다음 오브젝트와의 대응은 라벨이 측정하는 것의 범위를 한정한다. 두 코호트는 따로 보고하며 서로 비교하지 않는다.

## Claim–evidence map for the abstract

| Sentence | Evidence | Status |
|---|---|---|
| 210 000 matches, three patches; data-derived definition | `02_METHODS_CORE.md` §1, §5 | supported |
| Two cohorts by smaller side's participation | Methods §1 (`cohort`, `fine` flags; lineage `scale_classes`) | supported |
| Brier lift ≈ 0.0037 (T) and ≈ 0.0034 (S), intervals exclude zero | `03_RESULTS.md` §3.1: −0.00373 [−0.00461, −0.00279]; −0.00335 [−0.00379, −0.00288] | supported (exploratory tag in Methods §0) |
| Teamfight model carries part of its signal to skirmishes | §3.4: q_T→S − PT_flex_S −0.00141 [−0.00186, −0.00094] | supported (secondary) |
| Pooling helps neither | §3.4: q_S − q_TS −0.00024 [−0.00039, −0.00009]; q_TS − q −(+0.00071) [+0.00018, +0.00126] | supported (secondary) |
| Teamfight lift does not hold externally | §5: +0.0026 / +0.0040, no interval | supported as point ordering; wording "does not hold" already used in Results |
| Skirmish lift does hold externally (variant A only) | §5 S rows −0.0018 / −0.0020, no interval | pending author (`\pending{ext-TS-sentence}`) |
| Cohorts not ranked | Discussion §5; contract §6 | required guard |

## Writing-quality check (writing-anti-ai + venue rules)

- One paragraph; no "In this paper we…", no "crucial/delve/landscape"; two sentence lengths alternate; no em dashes; no abbreviations (ToG rule).
- Direct-statement score: the abstract states results as measurements, not as significance claims ("intervals excluding zero" rather than "significant").
- Author decision needed: variant A vs B (depends on `\pending{ext-TS-sentence}`), and whether "Brier score" needs the qualifier "match-weighted" (adds 1 word; recommended in Methods, optional here).
