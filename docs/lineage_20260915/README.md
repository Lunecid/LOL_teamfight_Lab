# Lineage copies (2026-09-15 label and cohort build) — reference only, not executed here

**Origin:** branch `codex/research-snapshot-20260917`, commit `e0ec3d0` (2026-09-17). Files below are **byte-identical copies** (`git show e0ec3d0:<origin path>`), kept so that the journal manuscript's evidence trace points to files inside this branch. Nothing here is run by this branch's scripts; the code copies are provenance, not a dependency.

| Copy | Origin path (at `e0ec3d0`) | sha256[:16] | What it establishes |
|---|---|---|---|
| `LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md` | `docs/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md` | `65bcf943edbe8b58` | h60/h90/h120 endpoint rule with worked timelines |
| `CLAUDE_COHORT_ROLE_TRAIN_20260915.md` | `docs/CLAUDE_COHORT_ROLE_TRAIN_20260915.md` | `d262ae99052e0d76` | T / N cohort specification |
| `CLAUDE_FULL_CORPUS_TRAIN_20260915.md` | `docs/CLAUDE_FULL_CORPUS_TRAIN_20260915.md` | `9c6bd95a34dee06d` | patch roles, VALIDATION role hash (V_CAL / V_SELECT / Q_CAL / Q_SELECT) |
| `cohort_manifest.json` | `outputs/cohort_role_training_20260915/cohorts/cohort_manifest.json` | `e39a33df3afad820` | per-set, per-sub_role row and match counts (`T`, `N`, `pick`, `skirmish`, `T_matches`, `N_matches`, `valid`) |
| `engagement_labels_v3_rules.py` | `scripts/engagement_labels_v3_rules.py` | `ffa87493b4afddbb` | `endpoint_rule`, `endpoint_validity` (canonical implementation) |
| `cr20260915_common.py` | `scripts/cr20260915_common.py` | `04d4a2eb9808888d` | `scale_classes` (cohort / fine codes from stored participation counts) |

## Provenance excerpts (verbatim, from files not copied)

**Protocol strings — `e0ec3d0:scripts/fc20260915_protocol.py` L146–150**
```
times='s = first kill - 15000 ms; q_pre = s - 1 ms; L = last kill',
endpoint='endpoint_h = min(L + 1000h, next raw CHAMPION_KILL strictly after L - 1, next eligible (stored next_start) '
         'engagement start - 1 (absent if next_start >= game end), game_end - 1) for h = 60, 90, 120 s',
validity='endpoint >= L, endpoint > q_pre, endpoint <= last frame, q_pre within observed frames, next_start > L, '
         'StateV2 builds at both queries; failures excluded with reason lineage',
```

**Fold / role hash — `e0ec3d0:scripts/fc20260915_common.py` L47–51, L65–77**
```
FOLD_TAG = 'full-v-oof-20260915:'
VAL_TAG = 'full-val-20260915:'
N_FOLDS = 5
VAL_ROLES = ('V_CAL', 'V_SELECT', 'Q_CAL', 'Q_SELECT')

def h8(tag, match_id):
    return int(hashlib.sha256((tag + str(match_id)).encode('utf-8')).hexdigest()[:8], 16)

def train_fold(match_id):
    """TRAIN match fold = int(sha256('full-v-oof-20260915:'+match_id).hexdigest()[:8],16) % 5."""
    return h8(FOLD_TAG, match_id) % N_FOLDS

def validation_role(match_id):
    """VALIDATION role = int(sha256('full-val-20260915:'+match_id).hexdigest()[:8],16) % 4."""
    return VAL_ROLES[h8(VAL_TAG, match_id) % 4]
```

**Label driver — `e0ec3d0:scripts/fc20260915_extract.py` L259, L287–289, L322–346 (abridged to the decisive lines)**
```
E['q_pre'].append(s - 1)
K_ms = s + 15_000
nk = R.next_kill_after(kills, L)
ns_eff = R.effective_next_start(int(x['next_start']), end_sem)
e_ms, reasons, cand = R.endpoint_rule(L, h, nk, ns_eff, end_sem)
val = R.endpoint_validity(e_ms, L, s - 1, support_start, last_frame)
inval = [k for k, ok in val.items() if not ok]
if int(x['next_start']) <= L:
    inval.insert(0, 'same_match_overlap_next_start_le_L')
...
post = _state(states, builder, int(e_ms))
if post.snapshot_ms > e_ms:
    raise ValueError('future snapshot')
...
E[f'valid_h{h}'].append(int(not inval))
```

**ΔV and Y — `e0ec3d0:scripts/fc20260915_labels.py` L78–81**
```
for h in HS:
    v = E[f'valid_h{h}'] == 1
    delta = np.where(v, p_post[h] - p_pre, np.nan)
    Y = np.where(v, (delta > 0).astype(np.int8), -1).astype(np.int8)
```

**Cohort file columns — `e0ec3d0:scripts/cr20260915_cohorts.py` L441–444 (`save_set`)**
```
arrays = dict(match=..., s=..., L=..., sub_role=..., valid_h60=..., valid_h90=..., valid_h120=...,
              cluster_blue=cb, cluster_red=cr, present_blue=pb, present_red=pr, ..., n_min=n_min,
              scale_known=known.astype(np.int8), cohort=cohort.astype(np.int8), fine=fine.astype(np.int8),
              teamfight_cut3_DIAG=..., teamfight_cut5_DIAG=..., presence_n_min_DIAG=pres_min, ...,
              role=np.asarray('COHORT_MEMBERSHIP_POST_CUTOFF_NOT_A_PREDICTOR'))
```

**Valid rows and the 348 exclusions — `e0ec3d0:docs/tog_delta_v_20260916/manuscript.md` L200–208**
```
| Patch role | Raw matches | Detected engagements | Valid ΔV rows | T rows | N rows |
| TRAIN 15.14 | 74,673 | 199,480 | 199,358 | 39,605 | 159,753 |
| VALIDATION 15.15 | 74,748 | 203,292 | 203,170 | 41,315 | 161,855 |
| TEST 15.16 | 60,579 | 163,680 | 163,576 | 32,981 | 130,595 |
| **Total** | **210,000** | **566,452** | **566,104** | **113,901** | **452,203** |
The same valid-row masks apply at the 60, 90 and 120 s caps. The 348 excluded rows are engagements whose next
eligible onset falls at or before the current last kill.
```
Korean source, `e0ec3d0:docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md` L54: "유효 행은 h60/h90/h120에서 동일하다. 제외 348행은 다음 engagement 시작이 현재 마지막 킬 이하인 중첩 사례다."

**Endpoint rule in manuscript form — `e0ec3d0:docs/tog_delta_v_20260916/manuscript.md` L319–328**
```
Let `K` be the first kill, `s = K − 15 s`, `t_pre = s − 1 ms`, `L` the last kill of the engagement, `J` the first
subsequent champion kill anywhere on the map, `S_next` the next eligible engagement's onset, and `T_end` match
termination. Missing candidates are treated as infinite.
e(h)  = min{ L + h,  J − 1 ms,  S_next − 1 ms,  T_end − 1 ms }
ΔV(h) = V(S(e(h))) − V(S(t_pre))
Y(h)  = 1[ ΔV(h) > 0 ]
h     = 90 s primary; 60 s and 120 s as sensitivity
```

**Realised endpoint statistics, main TEST h90 — `e0ec3d0:docs/tog_delta_v_20260916/manuscript.md` L351–358**
```
| Observed follow-up after last kill, mean / median | 45.493 s / 39.440 s |
| Reaching the 90 s cap | 14.286% |
| Stopped by the next global kill | 129,036 / 163,576 (≈78.9%) |
| No new observation frame after `L`, all 163,576 valid rows | 35.145% |
| |ΔV| ≤ 0.005 / 0.01 / 0.02 | 7.504% / 12.696% / 20.905% |
| Sign disagreement h60↔h90 / h90↔h120 / h60↔h120 | 1.545% / 0.608% / 1.909% |
```
(These statistics were computed under the 2026-09-15 logistic V; the endpoint times themselves do not depend on V.)

**113,901 vs 109,829 — `e0ec3d0:outputs/cohort_role_training_20260915/REPORT.md` L75**
"Main corpus at cut 4: 113,901 teamfight rows of 566,452 detected engagements. The manuscript's 109,829 teamfights refer to the market_event-labelled population (532,547 rows, draws dropped) …"

## Derived counts used by the manuscript (arithmetic on `cohort_manifest.json` fields; not themselves recorded)
- Pooled T matches = Σ `by_sub_role.*.h90.T_matches` over TRAIN fold0–4 (5,732 + 5,858 + 5,725 + 5,845 + 5,829 = 28,989), VALIDATION (7,550 + 7,469 + 7,448 + 7,632 = 30,099) and TEST (24,020) = **83,108**.
- T rows valid at h90 = 100%: Σ fold `h90.T` = 39,605 = `sets.MAIN_TRAIN.T`; VALIDATION 41,315; TEST 32,981 — all 348 invalid rows are N.
