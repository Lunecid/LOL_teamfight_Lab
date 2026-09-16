# Integration execution notes

- The first Claude launch (`claude-writer.jsonl`) repeatedly emitted API retry events under the restricted runtime and was interrupted before any deliverable was written. Its log is retained.
- The second launch uses the already-authorized Claude transfer scope from the user context. It exposes file read/write tools only, disables customization/MCP loading, and creates a fresh writing session. Shell execution, training, raw corpus transfer, and external messaging are outside its contract.
- The original 140-file evidence snapshot was captured before document writing. It covers named manuscripts, findings, relevant scripts, aggregate results and manifests, not a full rehash of all raw data and models.
- Independent checks resolved 2,193 selected metric/contrast JSON records and checked source hashes and arithmetic. The combined aggregate checker passed 6,158 checks. This is separate from the 410 saved final checks of six original experimental stages; those original gates were not rerun for a documentation task.
- A first scratch aggregation of cached order differences mistakenly grouped `fold0`–`fold4` into validation. The retained initial file is explicitly named `position_counts_initial_misgrouped.json`. The corrected `position_counts_verified.json` uses fold keys as TRAIN and asserts all three expected split totals; source files were never changed.
- No training was launched by this integration task. No original manuscript, model, data, or experiment output is authorized for modification.
