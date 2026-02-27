# Investigation: Teamfight Duration & Data Quality Anomalies

Root-cause analysis for three data quality issues identified in the teamfight dataset.

---

## 6.1 Duration > 60,000ms (838 fights, max 111,497ms)

**Severity:** P3 — Low
**Verdict:** Intended behavior

### Root Cause

Traced to `gameplay/fights.py:1250`:

```python
horizon_end_ts = int(max(fight_end_ts, engage_ts_val + horizon_ms))
```

The `max()` ensures the label window always includes all kills in the cluster. The duration formula:

```
duration = horizon_end_ts - engage_ts
         = max(last_kill_ts - engage_ts, 60,000ms)
         = max(cluster_span + TF2_ENGAGE_PRE_KILL_MS, 60,000ms)
         = max(cluster_span + 10,000ms, 60,000ms)
```

**Duration > 60s ⟺ cluster_span > 50s.** Kill clusters form when consecutive kills are within `TF2_KILL_CLUSTER_GAP_MS = 18s`, but the total cluster can span much longer.

The duration cap at `gameplay/fights.py:1253` rejects fights where `fight_end_ts - engage_ts > MAX_MERGED_FIGHT_DURATION_MS (120,000ms)`, so max theoretical duration is 120s. The observed max of 111,497ms (~112s) is within this bound.

### Example: How a 111s fight forms

```
Kill 1: t=0s     ─┐
Kill 2: t=16s     │  gap=16s < 18s → same cluster
Kill 3: t=33s     │  gap=17s < 18s → same cluster
Kill 4: t=49s     │  ...
Kill 5: t=65s     │
Kill 6: t=82s     │
Kill 7: t=101s   ─┘  cluster_span = 101s

engage_ts = Kill1 - 10s = -10s (relative)
duration  = 101s + 10s  = 111s ✓
```

### Recommendation

- **Keep in training data.** These represent legitimate extended teamfights (baron dance, base siege, prolonged objective contests).
- For models assuming fixed 60s windows: normalize by actual duration or add `duration_ms` as a feature.
- No code change required.

---

## 6.2 Engagement Overlap (11,037 cases, 3.67%)

**Severity:** P1 — High (sequential models only)
**Verdict:** Intended for i.i.d.; problematic for sequential models

### Root Cause

Traced to `gameplay/fight_postmerge.py:113-125`:

```python
if float(location_radius) > 0:
    if distance_2d((pcx, pcy), (ccx, ccy)) > float(location_radius):
        kept.append(f)    # ← overlap ALLOWED if spatially distant
        continue
```

The post-merge overlap enforcement has a **location-based exception**: when two fights overlap temporally but are spatially separated by more than `cluster_max_diameter` (4000 map units), both fights are kept. This is by design — simultaneous fights in different map locations (e.g., toplane skirmish + botlane dive) are genuinely independent events.

Additionally, the overlap check only compares against the immediately previous kept fight (`kept[-1]`), so transitive overlaps between non-adjacent fights can pass through.

### Impact

| Model Type | Impact | Action Required |
|-----------|--------|-----------------|
| i.i.d. (LightGBM) | None | No action |
| Sequential (RNN, Transformer) | Label leakage between timesteps | Clipping or masking |

### Recommendation

For sequential models, implement one of:
- **(a) Clipping:** `fight[i].label_end_ts = min(fight[i].label_end_ts, fight[i+1].engage_ts)`
- **(b) Masking:** skip overlapping pairs during sequence training loss computation
- **(c) Merge:** combine overlapping fights into a single extended event

Document this explicitly in the paper's data section.

---

## 6.3 t_start < START_OFFSET_MIN (3,598 cases, 1.20%)

**Severity:** P2 — Medium
**Verdict:** Bug — intent-vs-implementation mismatch

### Root Cause

`START_OFFSET_MIN = 2` is defined in `core/config.py:446` but is **never enforced** anywhere in the detection pipeline.

The actual time filter in `gameplay/fights.py:1222-1225`:

```python
ctx_ms = int(config.fight_context_min) * 60000   # = 1 * 60000 = 60,000ms
if engage_ts_val - ctx_ms < t_min_ms:
    diag["rejected_startctx"] += 1
    continue
```

This only requires `engage_ts >= t_min + 60,000ms` (1 minute into game).

`START_OFFSET_MIN` is **only referenced** in `core/diagnostics.py:147` for logging — it is never used as a filter.

```
Searched: grep -r "START_OFFSET_MIN" → 3 results
  core/config.py:446         — definition (START_OFFSET_MIN = 2)
  core/config_legacy.py:262  — legacy definition (START_OFFSET_MIN = 1)
  core/diagnostics.py:147    — logging only (not filtering)
```

### Impact

At 1 minute into the game:
- Champions may not have reached lane
- Feature information is extremely limited
- Context window (`FIGHT_CONTEXT_MIN = 1min`) may not be fully populated
- Model predictions at this game phase are unreliable

### Recommendation

**Option A — Enforce (recommended):**

Add to `gameplay/fights.py` after the context guard (line 1225):

```python
start_offset_ms = int(getattr(cfg, "START_OFFSET_MIN", 2)) * 60000
if engage_ts_val - t_min_ms < start_offset_ms:
    diag["rejected_start_offset"] = diag.get("rejected_start_offset", 0) + 1
    continue
```

**Option B — Post-filter:**

Filter during dataset construction in `data/index_split.py` to preserve cache reproducibility.

**Option C — Align config:**

If 1-minute fights are acceptable, change `START_OFFSET_MIN` from 2 to 1.

---

## Summary

| Issue | Count | Severity | Verdict | Action |
|-------|-------|----------|---------|--------|
| 6.1 Duration > 60s | 838 | P3 Low | Intended | None (document) |
| 6.2 Overlap | 11,037 | P1 High* | Intended (i.i.d.) | Clip/mask for seq models |
| 6.3 Early fights | 3,598 | P2 Medium | Bug | Enforce START_OFFSET_MIN |

\* P1 only for sequential model architectures.

---

## Code References

| File | Line | Relevance |
|------|------|-----------|
| `gameplay/fights.py` | 1250 | horizon_end_ts = max(fight_end_ts, engage_ts + horizon_ms) |
| `gameplay/fights.py` | 1253 | MAX_MERGED_FIGHT_DURATION_MS cap check |
| `gameplay/fights.py` | 1222-1225 | Context guard (only checks FIGHT_CONTEXT_MIN) |
| `gameplay/fight_postmerge.py` | 113-125 | Location-based overlap exception |
| `core/config.py` | 446 | START_OFFSET_MIN = 2 (unused) |
| `core/config.py` | 435 | FIGHT_CONTEXT_MIN = 1 (actual filter) |
| `core/config.py` | 444 | MAX_MERGED_FIGHT_DURATION_MS = 120000 |
