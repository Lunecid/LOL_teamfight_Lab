# Champion-class position-assumption audit (2026-09-16)

## Decision

The frozen champion-class arm uses **participant-order slots**. It assigns the
names `TOP/JG/MID/BOT/SUP` to those slots by protocol convention; the code does
not establish that each slot is the corresponding Riot role in each match.
Position-pair and lane/role interpretations must therefore be qualified as
assumption-conditioned proxies. Existing class frequencies can still be
reported as frequencies by participant-order slot.

## Relevant files and data flow

- `worktrees/engagement-state-value/gameplay/state_value_v2.py:L38-L45` builds
  `participant_order = sorted(team_map, key=(team_id, participant_id))` and
  replaces incoming `meta.role_slots` with that order. `L63-L78` passes the
  sanitized pack to the legacy builder. The V2 contract test
  `tests/test_state_value_v2_contract.py:L36-L42` confirms that deleting,
  permuting, or mutating role metadata does not change V2 values.
- `worktrees/engagement-state-value/gameplay/state_value.py:L38-L56,L127-L136`
  consumes the role-slot map supplied by its caller. V2 has already replaced
  that map, so this downstream `slotN` representation is also participant
  order, not original role order.
- Original cache metadata does contain a role-slot map:
  `worktrees/engagement-state-value/data/cache_io.py:L421-L437` writes
  `meta.role_slots = get_role_slots_from_detail(detail)`. The generator
  `worktrees/engagement-state-value/core/roles.py:L40-L86` derives it from
  `teamPosition`/`individualPosition`, then fills missing roles by remaining
  participant IDs. That is a weak metadata proxy with possible silent fills.
- `scripts/fc20260915_extract.py:L165-L184` computes `order_differs` by
  comparing cached `role_slots` order with V2 `participant_order`. It is one
  match-level reorder indicator; it is not per-slot role accuracy.
- `scripts/cc20260916_common.py:L53-L64` declares `POSITIONS` and hard-codes
  blue/red slot blocks. `L151-L165,L183-L249` constructs all class-count,
  class-aggregate, same-team, and cross-team features from the ten slot
  columns. No `role_slots`, `teamPosition`, or per-match role join is read.
  `scripts/cc20260916_protocol.py:L64-L87` records this mapping as a feature
  specification, not as a validated role label.
- The frozen state definition is explicit at
  `outputs/full_corpus_training_20260915/protocol.json:L363-L367`:
  participant order is team then participant ID and has no positions.

## Stored full-corpus aggregate

The following are sums of the stored `participant_order_differs_from_role_slots`
fields in `outputs/full_corpus_training_20260915/eval/counts_and_exclusions.json`:

| Stored population | Matches | `order_differs` | Rate |
|---|---:|---:|---:|
| TRAIN folds 0–4 | 74,673 | 1,255 | 1.68% |
| VALIDATION Q/V roles | 74,748 | 1,251 | 1.67% |
| TEST | 60,579 | 1,042 | 1.72% |
| Main total | 210,000 | 3,548 | 1.69% |

Evidence rows: TRAIN folds are at `counts_and_exclusions.json:L201-L214,
L240-L253,L279-L292,L318-L331,L357-L370`; Q/V validation rows are at
`L5-L19,L45-L58,L123-L136,L162-L175`; TEST is at `L84-L97`. The evaluator
definition is `scripts/fc20260915_evaluate.py:L352-L379`. These rates show that
the two orders disagree in a small but nonzero fraction of stored matches;
they do not show which individual slots are correct when the orders agree.
The post-freeze external rows have 3 differences in KR 16.13 and 0 in the
other stored external partitions (`counts_and_exclusions.json:L397-L411,
L438-L452,L473-L487,L508-L523`); those zeros are not role-accuracy evidence.

## Existing role-validation limits

- `outputs/cohort_role_training_20260915/role_supervision_provenance.json:L1-L36`
  labels main TRAIN supervision as cached `meta.role_slots`, explicitly weak
  and not raw `teamPosition`. It records 149,346 teams, 149,162 teams used,
  184 detectable participant-ID-fill teams, and 745,810 participants. The
  stored `73,251` identity-order matches are another order statistic, not role
  truth.
- `scripts/cr20260915_draft.py:L1-L16,L45-L109` documents the same weak-label
  source, StateV2 participant order, and undetectable whole-team/trailing-slot
  fills. Its raw audit covered only two distinct TRAIN matches; the provenance
  file reports the examples at `role_supervision_provenance.json:L37-L40,
  L138-L140,L240-L242` and says accuracy was not validated at `L252-L256`.
- Later external diagnostics use Riot `teamPosition`, which the report calls a
  role assignment rather than observed spatial lane:
  `outputs/cohort_role_training_20260915/REPORT.md:L283-L292,L504-L515`.
  Their posterior/classifier results are transfer diagnostics for later
  patches/regions, not in-distribution evidence that frozen main slots equal
  roles. The raw diagnostic source and invalid-team handling are recorded in
  `outputs/cohort_role_training_20260915/eval/role_reliability_external_raw.json:L1-L3`.

## Claims supported and unsupported

Supported: V2 slots are deterministic team/participant-ID order; the class arm
uses those slots; the stored main-corpus reorder rates above are reproducible;
and cached role supervision is a weak proxy with documented fill and coverage
limits.

Unsupported by the current evidence: every slot 0/5 is TOP, slot 1/6 is JG,
etc.; participant ID itself guarantees role; `cc_matchup_TOP` is a same-role
matchup; or an observed class effect can be attributed to lane/role semantics.
Agreement of the two orderings, including the 98%+ matches where
`order_differs=0`, cannot prove any of those statements.

## Required manuscript/spec wording and minimal validation

In `docs/CLAUDE_CHAMPION_CLASS_20260916.md:L20-L22`, describe the reported
class percentages as **participant-order-slot** distributions and state that
the TOP/JG/MID/BOT/SUP names are the frozen mapping assumption. In the feature
description at `L37-L49`, and in `docs/CHAMPION_CLASS_FINDINGS_20260916.md:L33-L38`,
call pair features fixed slot-pair features or assumption-conditioned proxies;
remove the assertion that “slot = position” and avoid unqualified
“mid/jungle/bottom” interpretations. Existing frozen results do not need a
new fit for this correction.

If the manuscript must retain semantic role claims, the minimal additional
check is a read-only raw-detail audit over the frozen main 15.14–15.16
population: for every StateV2 slot, join its participant ID to raw
`teamPosition`, and report slot-wise agreement plus missing/duplicate roles,
ID-fill cases, and match/team coverage. The two TRAIN examples and later
external diagnostics cannot substitute for that in-distribution audit. If
main raw detail is unavailable, retain the assumption-qualified wording and
mark the semantic mapping unvalidated; do not infer it from participant IDs or
from `order_differs=0`.
