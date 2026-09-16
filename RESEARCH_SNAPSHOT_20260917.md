# Research snapshot — 2026-09-17

This branch captures the current research implementation and the audited September 16 delta-V working draft. It is a working research snapshot, not a submission-ready paper or a portable turnkey data release.

Start with [the integrated draft index](docs/tog_delta_v_20260916/README.md), [the manuscript](docs/tog_delta_v_20260916/manuscript.md), [reviewer responses](docs/tog_delta_v_20260916/reviewer_response_matrix.md), and [the collaborator specification](docs/tog_delta_v_20260916/collaborator_specification.md).

The snapshot combines tracked files from local research commit `15751e8` with current workspace scripts, tests, protocols and documents. It is committed on top of the existing GitHub master history; the local development history is represented by this snapshot, not rewritten into master.

Selected aggregate results and audit evidence are in `outputs/`. They were explicitly selected despite the normal output-directory ignore rule. Raw matches, split membership lists, record-level timelines, predictions, trained models, media, credentials, assistant conversations and runtime logs are excluded. Aggregate documents may refer to local-only files and Windows paths; these are provenance references, not bundled dependencies. Two older explanatory documents have public-copy match identifiers redacted; original local files were preserved. Some older evidence files containing record-level identifiers are omitted. No excluded artifact is claimed to be reproducible from this Git snapshot alone.

The September 16 acceptance record reports 703 document integrity checks, 6,158 aggregate consistency checks and 140 named source files preserved. Those were local integration checks, not fresh training or an end-to-end test of this exported branch. Test-exposed exploratory evidence, model-defined labels, positional proxies and unexecuted architectures remain qualified in the manuscript. Upload preparation did not rerun experiments.

`GITHUB_SNAPSHOT_MANIFEST.json` records the exported file hashes (excluding itself and Git internals). Existing local experiment artifacts and the original repository were not modified.
