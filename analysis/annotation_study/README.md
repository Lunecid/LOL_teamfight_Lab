# Teamfight Detector Annotation Study (Track A)

Human-annotation validation of the `teamfight_v2` kill-cluster detector:
annotators mark engagement intervals on a **minimap replay reconstructed from
the same Match-V5 telemetry the detector consumes**, blind to the detector's
output. Agreement between humans and the detector then quantifies whether the
algorithm's clustering / boundary / merge decisions match human judgment.

**Scope note.** Because the viewer shows the same telemetry the detector sees,
this study validates the *decision rules* (18 s gap, 4,000 u diameter split,
1,800 u alive-validation, 15 s / 2,000 u merge). It cannot detect information
missing from the API itself (e.g., kill-less standoffs) — those are out of the
paper's kill-conditioned scope by definition and are reported separately
(`killless_annotated`).

## 1. Prepare the dataset

```bash
python -m analysis.annotation_study.prepare --n_matches 50 --seed 7 \
    --timeline_dir <TIMELINE_DIR> --detail_dir <DETAIL_DIR>
```

Output (`outputs/annotation_study/`):

| Path | Contents | Give to annotators? |
|---|---|---|
| `viewer.html` | annotation UI | ✅ |
| `matches/*.json` | minimap replay payloads | ✅ |
| `detector/*.json` | teamfight_v2 intervals | ❌ **never** (blind study) |
| `manifest.json` | sampling record (seed, patches) | — |

## 2. Annotation protocol

Each annotator, independently (no discussion until both finish):

1. Open `viewer.html` in a browser (no server needed), enter a unique
   annotator ID, and load the `matches/*.json` files via the file picker.
2. Watch each match (4×–8× speed; kill ticks on the timeline help navigation).
3. Mark every **combat engagement**: an interval where players from both teams
   actively fight (trading damage/kills), from the moment fighting starts
   (initiation, not the first kill) to the moment it ends (disengage or wipe).
   - `[` = interval start, `]` = interval end + save.
   - Type: **한타 (3v3+)** / **소규모 교전 (2v2~)** / **불확실**.
   - Separate engagements ≥ ~15 s apart or in clearly different locations
     should be separate intervals (mirror your own judgment — do not try to
     guess the algorithm's rules).
4. Progress is auto-saved in the browser (localStorage, keyed by annotator
   ID). When done, click **전체 내보내기** to export
   `annotations_<id>.json`.

Recommended: ≥ 2 annotators × 50 matches, plus a 5-match warm-up (excluded
from scoring) to calibrate the interval-splitting convention.

## 3. Score

```bash
python -m analysis.annotation_study.score \
    --study_dir outputs/annotation_study \
    --annotations annotations_a1.json annotations_a2.json
```

Reports (temporal IoU ≥ 0.5, greedy one-to-one matching):

- **Annotator vs detector** — precision / recall / F1, mean matched IoU,
  mean |onset error| (s), plus `recall_kill_scope` (recall restricted to
  annotated intervals containing ≥ 1 kill — the detector's declared target).
- **Inter-annotator** — pairwise F1 and Cohen's κ on a 1-second grid.
  Low κ here bounds the achievable detector agreement and is itself a
  finding about the ambiguity of the "teamfight" concept.

## Interpreting results

- High `recall_kill_scope` + high precision → the kill-conditioned
  localization matches human judgment on its declared scope.
- `killless_annotated` count → how much combat the kill-conditioned
  definition excludes (paper limitation, quantified).
- Detector-vs-human F1 should be interpreted **relative to
  inter-annotator F1** (the human ceiling), not against 1.0.
