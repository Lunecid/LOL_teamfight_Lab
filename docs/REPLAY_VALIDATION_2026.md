# 2026 full-replay detector validation

This protocol validates `teamfight_v2` against independent full-match League
Client replay review. The detector still consumes Match-V5 telemetry; the
archived `.rofl` is the human evidence and must be watched from start to end.

## Current completed run

- Population: 500 complete KR ranked replays, public patch 26.13 / API patch
  16.13, created from 2026-06-30 through 2026-07-06 UTC.
- Byte integrity: replay, detail, and timeline SHA-256 values reconciled for all
  500 rows; both SQLite databases passed `integrity_check`.
- Exclusions: 12 games shorter than 10 minutes. Eleven of these were below
  1 MiB and are treated as expected short/remake warnings rather than corrupt
  files.
- Gold set: 100 full matches, ten deterministic samples from each of ten
  creation-time bins, seed `20260715`. Detector output was not used for
  sampling.
- Technical detector result: 480 candidate intervals (mean 4.8 per match), no
  execution exception, repeated-run difference, invalid/zero-kill boundary,
  final-interval overlap, diagnostic error, or unknown monster event.
- Human precision, recall, F1, temporal IoU, onset error, and kill-less miss
  rate remain pending. Do not cite the technical pass as detector accuracy.

## Reproduce archive, sample, and detector checks

```powershell
python scripts\validate_replay_goldset.py `
  --replay-db D:\LOL_Project\data\replays\2026_current\_replay_state\replay_manifest.sqlite3 `
  --source-db D:\LOL_Project\data\raw\2026_current\_collector_state\collector.sqlite3 `
  --raw-root D:\LOL_Project\data\raw\2026_current `
  --project-root C:\Users\todtj\PycharmProjects\LOL_teamfight `
  --out D:\LOL_Project\validation\replay_goldset_2026_26_13 `
  --n-matches 100 --time-bins 10 --seed 20260715
```

The output separates blinded material from internal detector evidence:

- Give annotators only `annotator/annotation_form.html`, `queue.json`,
  `replay_queue.csv`, and read-only access to the listed replay files.
- Never give annotators `detector/`, `matches/`, `candidate_intervals.csv`, or
  `technical_test_report.json` before their independent exports are frozen.
- `matches/` is a telemetry reconstruction retained for adjudication and
  scoring support. It is not the primary annotation evidence.

## Human annotation

Use two annotators for all 100 matches. Each marks every engagement from
initiation to disengage/wipe, including engagements with no kill. Annotators
must not discuss boundaries or view detector-selected timestamps until both
exports are frozen.

Afterward, score the two exported JSON files with the existing scorer:

```powershell
python -m analysis.annotation_study.score `
  --study_dir D:\LOL_Project\validation\replay_goldset_2026_26_13 `
  --annotations annotations_A01.json annotations_A02.json
```

Report detector-vs-human precision/recall/F1 and temporal errors beside
inter-annotator F1 and Cohen's kappa. Repeat the same frozen detector and
annotation protocol on patches 15.14-15.16 before making a cross-season
robustness claim.
